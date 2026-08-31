import numpy as np
import os
import time
import torch

from datetime import datetime
from torch import Tensor
from loss import GlobalMassConservationLoss, LocalMassConservationLoss
from data.dataset_normalizer import DatasetNormalizer
from typing import Callable, Optional, List, Tuple, Union
from utils import physics_utils

from . import Logger
from .metric_utils import RMSE, MAE, NSE, CSI

class ValidationStats:
    def __init__(self,
                 logger: Optional[Logger] = None,
                 normalizer: Optional[DatasetNormalizer] = None,
                 is_normalized: Optional[bool] = None,
                 delta_t: Optional[int] = None):
        self.normalizer = normalizer
        self.is_normalized = is_normalized
        self.delta_t = delta_t
        self.val_start_time = None
        self.val_end_time = None
        self.timestamps = []

        # ======== Water volume stats ========
        self.pred_list = []
        self.target_list = []

        # Overall stats
        self.rmse_list = []
        self.mae_list = []
        self.nse_list = []
        self.csi_list = []

        # Flooded cell stats
        self.rmse_flooded_list = []
        self.mae_flooded_list = []
        self.nse_flooded_list = []

        # ======== Water flow stats ========
        self.edge_pred_list = []
        self.edge_target_list = []

        # Overall stats
        self.edge_rmse_list = []
        self.edge_mae_list = []
        self.edge_nse_list = []

        # ======== Physics-informed stats ========
        self.global_mass_loss_list = []
        self.local_mass_loss_list = []

        self.log = print
        if logger is not None and hasattr(logger, 'log'):
            self.log = logger.log
    
    def start_validate(self):
        self.val_start_time = time.time()

    def end_validate(self):
        self.val_end_time = time.time()

    def get_computation_time(self) -> Tuple[float, float]:
        if self.val_start_time is None or self.val_end_time is None:
            return None, None

        total_time = self.val_end_time - self.val_start_time
        num_timesteps = max(len(self.pred_list), len(self.edge_pred_list))
        inference_time = total_time / num_timesteps
        return total_time, inference_time

    def get_avg_rmse(self) -> float:
        return float(np.mean(self.rmse_list))

    def get_avg_mae(self) -> float:
        return float(np.mean(self.mae_list))

    def get_avg_nse(self) -> float:
        return float(np.mean(self.nse_list))

    def get_avg_edge_rmse(self) -> float:
        return float(np.mean(self.edge_rmse_list))

    def get_avg_edge_mae(self) -> float:
        return float(np.mean(self.edge_mae_list))

    def get_avg_edge_nse(self) -> float:
        return float(np.mean(self.edge_nse_list))

    def get_total_global_mass_loss(self) -> float:
        return float(np.sum(self.global_mass_loss_list))

    def get_total_local_mass_loss(self) -> float:
        return float(np.sum(self.local_mass_loss_list))

    def add_pred_for_timestep(self,
                              pred: Tensor = None,
                              target: Tensor = None,
                              edge_pred: Tensor = None,
                              edge_target: Tensor = None,
                              timestamp: datetime = None):
        if pred is not None:
            assert target is not None, "target must be provided if pred is provided."
            self.pred_list.append(pred)
            self.target_list.append(target)
        if edge_pred is not None:
            assert edge_target is not None, "edge_target must be provided if edge_pred is provided."
            self.edge_pred_list.append(edge_pred)
            self.edge_target_list.append(edge_target)
        if timestamp is not None:
            self.timestamps.append(timestamp)

    def compute_overall_stats(self, water_threshold: Union[float, Tensor] = 0.05):
        def get_metric_list(metric_func: Callable, p: Tensor, t: Tensor, mask: Tensor = None):
            v_metric_func = torch.vmap(metric_func)
            if mask is not None:
                out = v_metric_func(p, t, mask).tolist()
                return out
            out = v_metric_func(p, t).tolist()
            return out

        if len(self.pred_list) > 0 and len(self.target_list) > 0:
            t_pred = torch.stack(self.pred_list, dim=0)
            t_target = torch.stack(self.target_list, dim=0)

            # Per timestep node stats
            self.rmse_list = get_metric_list(RMSE, t_pred, t_target)
            self.mae_list = get_metric_list(MAE, t_pred, t_target)
            self.nse_list = get_metric_list(NSE, t_pred, t_target)

            binary_pred = t_pred > water_threshold
            binary_target = t_target > water_threshold
            self.csi_list = get_metric_list(CSI, binary_pred, binary_target)

            # Flooded area stats
            flooded_mask = binary_pred | binary_target
            self.rmse_flooded_list = get_metric_list(RMSE, t_pred, t_target, mask=flooded_mask)
            self.mae_flooded_list = get_metric_list(MAE, t_pred, t_target, mask=flooded_mask)
            self.nse_flooded_list = get_metric_list(NSE, t_pred, t_target, mask=flooded_mask)

        if len(self.edge_pred_list) > 0 and len(self.edge_target_list) > 0:
            t_edge_pred = torch.stack(self.edge_pred_list, dim=0)
            t_edge_target = torch.stack(self.edge_target_list, dim=0)

            # Per timestep edge stats
            self.edge_rmse_list = get_metric_list(RMSE, t_edge_pred, t_edge_target)
            self.edge_mae_list = get_metric_list(MAE, t_edge_pred, t_edge_target)
            self.edge_nse_list = get_metric_list(NSE, t_edge_pred, t_edge_target)

    def compute_physics_informed_stats_for_timestep(self,
                                                    pred: Tensor,
                                                    prev_node_pred: Tensor,
                                                    prev_edge_pred: Tensor,
                                                    databatch,
                                                    local_mass_nodes: List[int] = None):
        assert self.normalizer is not None and self.is_normalized is not None and self.delta_t is not None, \
            "normalizer, is_normalized, and delta_t must be set before updating physics-informed stats."

        global_mass_loss_func = GlobalMassConservationLoss(mode='test',
                                                           normalizer=self.normalizer,
                                                           is_normalized=self.is_normalized,
                                                           delta_t=self.delta_t)
        total_rainfall = physics_utils.get_total_rainfall(databatch)
        global_mass_loss = global_mass_loss_func(pred, prev_node_pred, prev_edge_pred, total_rainfall, databatch)
        self.global_mass_loss_list.append(global_mass_loss.cpu().item())

        local_mass_loss_func = LocalMassConservationLoss(mode='test',
                                                         normalizer=self.normalizer,
                                                         is_normalized=self.is_normalized,
                                                         delta_t=self.delta_t)
        rainfall = physics_utils.get_rainfall(databatch)
        local_nodes_mask = None
        if local_mass_nodes is not None:
            # Only compute local mass loss for specific nodes
            assert databatch.num_graphs == 1, 'For testing, assume there is only one graph per batch.'
            local_nodes_mask = np.isin(np.arange(databatch.num_nodes), local_mass_nodes)

        local_mass_loss = local_mass_loss_func(pred, prev_node_pred, prev_edge_pred, rainfall, databatch, local_nodes_mask)
        self.local_mass_loss_list.append(local_mass_loss.cpu().item())

    def print_stats_summary(self):
        def print_stat_avg(name: str, values: List[float], newline: bool = False):
            if len(values) > 0:
                newline_str = '\n' if newline else ''
                self.log(f'{newline_str}Average {name}: {np.mean(values):.4e}')

        print_stat_avg('RMSE', self.rmse_list)
        print_stat_avg('MAE', self.mae_list)
        print_stat_avg('NSE', self.nse_list)
        print_stat_avg('CSI', self.csi_list)
        print_stat_avg('RMSE (flooded)', self.rmse_flooded_list)
        print_stat_avg('MAE (flooded)', self.mae_flooded_list)
        print_stat_avg('NSE (flooded)', self.nse_flooded_list)

        print_stat_avg('Edge RMSE', self.edge_rmse_list, newline=True)
        print_stat_avg('Edge MAE', self.edge_mae_list)
        print_stat_avg('Edge NSE', self.edge_nse_list)

        if len(self.global_mass_loss_list) > 0:
            self.log(f'\nTotal Global Mass Conservation Loss: {self.get_total_global_mass_loss():.4e}')
        if len(self.local_mass_loss_list) > 0:
            self.log(f'\nTotal Local Mass Conservation Loss: {self.get_total_local_mass_loss():.4e}')

        if self.val_start_time is not None and self.val_end_time is not None:
            total_time, inference_time = self.get_computation_time()
            self.log(f'\nValidation time: {total_time:.2f} seconds')
            self.log(f'Inference time for one timestep: {inference_time:.4f} seconds')

    def save_stats(self, filepath: str):
        dirname = os.path.dirname(filepath)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        total_time, inference_time = self.get_computation_time()
        stats = {
            'timestamps': self.timestamps,
            'pred': np.array(self.pred_list),
            'target': np.array(self.target_list),
            'edge_pred': np.array(self.edge_pred_list),
            'edge_target': np.array(self.edge_target_list),
            'rmse': np.array(self.rmse_list),
            'mae': np.array(self.mae_list),
            'nse': np.array(self.nse_list),
            'csi': np.array(self.csi_list),
            'rmse_flooded': np.array(self.rmse_flooded_list),
            'mae_flooded': np.array(self.mae_flooded_list),
            'nse_flooded': np.array(self.nse_flooded_list),
            'edge_rmse': np.array(self.edge_rmse_list),
            'edge_mae': np.array(self.edge_mae_list),
            'edge_nse': np.array(self.edge_nse_list),
            'global_mass_loss': np.array(self.global_mass_loss_list),
            'local_mass_loss': np.array(self.local_mass_loss_list),
            'total_time': total_time,
            'inference_time': inference_time,
        }
        np.savez(filepath, **stats)

        self.log(f'Saved metrics to: {filepath}')
