import torch

from utils import file_utils
from data import dataset_factory
from utils import Logger


CONFIG_PATH = "configs/config.yaml"

config = file_utils.read_yaml_file(CONFIG_PATH)

dataset_parameters = config["dataset_parameters"]
training_parameters = dataset_parameters["training"]
loss_parameters = config["loss_func_parameters"]

logger = Logger(log_path=None)

dataset_config = {
    "mode": "train",
    "dataset_summary_file": training_parameters["dataset_summary_file"],
    "event_stats_file": training_parameters["event_stats_file"],
    "root_dir": dataset_parameters["root_dir"],
    "nodes_shp_file": dataset_parameters["nodes_shp_file"],
    "edges_shp_file": dataset_parameters["edges_shp_file"],
    "dem_file": dataset_parameters["dem_file"],
    "features_stats_file": dataset_parameters["features_stats_file"],
    "previous_timesteps": dataset_parameters["previous_timesteps"],
    "normalize": dataset_parameters["normalize"],
    "timestep_interval": dataset_parameters["timestep_interval"],
    "spin_up_time": dataset_parameters["spin_up_time"],
    "time_from_peak": dataset_parameters["time_from_peak"],
    "inflow_boundary_nodes": dataset_parameters["inflow_boundary_nodes"],
    "outflow_boundary_nodes": dataset_parameters["outflow_boundary_nodes"],
    "with_global_mass_loss": loss_parameters["use_global_mass_loss"],
    "with_local_mass_loss": loss_parameters["use_local_mass_loss"],
    "debug": False,
    "logger": logger,
    "force_reload": False,
}

storage_mode = dataset_parameters["storage_mode"]

dataset = dataset_factory(
    storage_mode,
    autoregressive=False,
    **dataset_config,
)

sample = dataset[0]

print("\n" + "=" * 70)
print("SAMPLE STRUCTURE")
print("=" * 70)

print(sample)

print("\nAvailable keys:")
print(sample.keys())

print("\nTensor fields:")

for key in sample.keys():
    value = sample[key]

    if torch.is_tensor(value):
        print(
            f"{key:30s}",
            "shape=",
            tuple(value.shape),
            "dtype=",
            value.dtype,
        )
    else:
        print(
            f"{key:30s}",
            type(value),
        )

print("=" * 70)
