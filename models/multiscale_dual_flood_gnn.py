"""A multiscale extension of DUALFloodGNN.

The model follows the fine-to-coarse-to-fine processor used by mSWE-GNN while
retaining DUALFloodGNN's node-volume and signed edge-flow outputs.  Coarse
graphs are internal latent representations; labels and physics losses remain
defined on the original HEC-RAS graph.
"""

import hashlib
from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn import Module, ModuleList

from utils.model_utils import make_mlp
from utils.multiscale_utils import build_multiscale_hierarchy

from .base_model import BaseModel
from .dual_flood_gnn import NodeEdgeConv


class NodeEdgeProcessor(Module):
    """A stack of hidden-width DUAL node/edge message-passing layers."""

    def __init__(
        self,
        hidden_features: int,
        num_layers: int,
        mlp_layers: int,
        activation: str,
        residual: bool,
        device: str,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("Each graph scale requires at least one GNN layer.")

        self.layers = ModuleList(
            [
                NodeEdgeConv(
                    node_in_channels=hidden_features,
                    edge_in_channels=hidden_features,
                    node_out_channels=hidden_features,
                    edge_out_channels=hidden_features,
                    hidden_size=hidden_features,
                    num_layers=mlp_layers,
                    activation=activation,
                    residual=residual,
                    bias=False,
                    device=device,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        debug_message_passing: bool = False,
        scale_name: str = "",
    ) -> Tuple[Tensor, Tensor]:
        """Run node-edge message passing at one graph scale.

        Debug output is optional and does not change the numerical forward pass.
        """

        # PyG message passing over an empty topology is version-sensitive.
        if edge_index.numel() == 0:
            if debug_message_passing:
                print(f"\n[{scale_name}] EMPTY GRAPH - skipping message passing", flush=True)
            return x, edge_attr

        if debug_message_passing:
            print("\n" + "=" * 70)
            print(f"MESSAGE PASSING: {scale_name}")
            print("=" * 70)
            print("Input node latent shape :", tuple(x.shape))
            print("Input edge latent shape :", tuple(edge_attr.shape))
            print("edge_index shape        :", tuple(edge_index.shape))
            print("Number of GNN layers    :", len(self.layers))

        for layer_id, layer in enumerate(self.layers):
            if debug_message_passing:
                # Only clone in debug mode; normal training has no extra memory cost.
                x_before = x.detach().clone()
                edge_before = edge_attr.detach().clone()

            x, edge_attr = layer(x, edge_index, edge_attr)

            if debug_message_passing:
                node_change = (x.detach() - x_before).abs().mean().item()
                edge_change = (edge_attr.detach() - edge_before).abs().mean().item()

                print(f"\n--- GNN Layer {layer_id + 1} ---")
                print("Node shape after MP :", tuple(x.shape))
                print("Edge shape after MP :", tuple(edge_attr.shape))
                print("Mean |node change|  :", f"{node_change:.6e}")
                print("Mean |edge change|  :", f"{edge_change:.6e}")

                if x.shape[0] > 0:
                    print("Example node latent before:", x_before[0, :5].cpu().tolist())
                    print("Example node latent after :", x[0, :5].detach().cpu().tolist())

                if edge_attr.shape[0] > 0:
                    print("Example edge latent before:", edge_before[0, :5].cpu().tolist())
                    print("Example edge latent after :", edge_attr[0, :5].detach().cpu().tolist())

        if debug_message_passing:
            print("=" * 70, flush=True)

        return x, edge_attr


class MultiScaleDUALFloodGNN(BaseModel):
    """DUALFloodGNN with configurable fine/coarse graph resolutions.

    For ``S`` scales the processor traverses
    ``0, 1, ..., S-1, S-2, ..., 0`` and therefore contains ``2S-1`` graph
    blocks.  Latent nodes and crossing latent edges are mean-pooled.  Model
    outputs always retain the original fine node and edge ordering.
    """

    requires_multiscale_hierarchy = True

    def __init__(
        self,
        input_features: int = None,
        input_edge_features: int = None,
        output_features: int = None,
        output_edge_features: int = None,
        hidden_features: int = 64,
        num_scales: int = 2,
        coarsening_factor: int = 4,
        layers_per_scale: int = 1,
        activation: str = "relu",
        residual: bool = True,
        mlp_layers: int = 2,
        skip_connections: bool = True,
        encoder_layers: int = 2,
        encoder_activation: str = "relu",
        decoder_layers: int = 2,
        decoder_activation: str = "relu",
        **base_model_kwargs,
    ):
        super().__init__(**base_model_kwargs)

        if num_scales < 2:
            raise ValueError("MultiScaleDUALFloodGNN requires at least two scales.")
        if coarsening_factor < 2:
            raise ValueError("coarsening_factor must be at least 2.")
        if hidden_features <= 0:
            raise ValueError("hidden_features must be positive.")
        if layers_per_scale < 1:
            raise ValueError("layers_per_scale must be at least 1.")
        if mlp_layers < 1:
            raise ValueError("mlp_layers must be at least 1.")

        self.num_scales = num_scales
        self.coarsening_factor = coarsening_factor
        self.skip_connections = skip_connections
        self._hierarchy_initialized = False
        self._num_nodes_per_scale: List[int] = []
        self._num_edges_per_scale: List[int] = []

        input_features = input_features or self.input_node_features
        input_edge_features = input_edge_features or self.input_edge_features
        output_features = output_features or self.output_node_features
        output_edge_features = output_edge_features or self.output_edge_features

        self.node_encoder = self._make_projection(
            input_size=input_features,
            output_size=hidden_features,
            hidden_size=hidden_features * 2,
            num_layers=encoder_layers,
            activation=encoder_activation,
        )
        self.edge_encoder = self._make_projection(
            input_size=input_edge_features,
            output_size=hidden_features,
            hidden_size=hidden_features,
            num_layers=encoder_layers,
            activation=encoder_activation,
        )

        self.down_processors = ModuleList(
            [
                NodeEdgeProcessor(
                    hidden_features=hidden_features,
                    num_layers=layers_per_scale,
                    mlp_layers=mlp_layers,
                    activation=activation,
                    residual=residual,
                    device=self.device,
                )
                for _ in range(num_scales - 1)
            ]
        )
        self.bottleneck_processor = NodeEdgeProcessor(
            hidden_features=hidden_features,
            num_layers=layers_per_scale,
            mlp_layers=mlp_layers,
            activation=activation,
            residual=residual,
            device=self.device,
        )
        self.up_processors = ModuleList(
            [
                NodeEdgeProcessor(
                    hidden_features=hidden_features,
                    num_layers=layers_per_scale,
                    mlp_layers=mlp_layers,
                    activation=activation,
                    residual=residual,
                    device=self.device,
                )
                for _ in range(num_scales - 1)
            ]
        )

        # Learned parent-to-child transfer approximates the paper's learned
        # inter-mesh upsampling.  Edge transfer is zero for edges that remain
        # completely inside one coarse cell.
        self.node_upsamplers = ModuleList(
            [
                make_mlp(
                    input_size=hidden_features,
                    output_size=hidden_features,
                    hidden_size=hidden_features,
                    num_layers=2,
                    activation=activation,
                    bias=False,
                    device=self.device,
                )
                for _ in range(num_scales - 1)
            ]
        )
        self.edge_upsamplers = ModuleList(
            [
                make_mlp(
                    input_size=hidden_features,
                    output_size=hidden_features,
                    hidden_size=hidden_features,
                    num_layers=2,
                    activation=activation,
                    bias=False,
                    device=self.device,
                )
                for _ in range(num_scales - 1)
            ]
        )

        fusion_input_size = hidden_features * 2 if skip_connections else hidden_features
        self.node_fusions = ModuleList(
            [
                make_mlp(
                    input_size=fusion_input_size,
                    output_size=hidden_features,
                    hidden_size=hidden_features,
                    num_layers=2,
                    activation=activation,
                    bias=False,
                    device=self.device,
                )
                for _ in range(num_scales - 1)
            ]
        )
        self.edge_fusions = ModuleList(
            [
                make_mlp(
                    input_size=fusion_input_size,
                    output_size=hidden_features,
                    hidden_size=hidden_features,
                    num_layers=2,
                    activation=activation,
                    bias=False,
                    device=self.device,
                )
                for _ in range(num_scales - 1)
            ]
        )

        self.node_decoder = self._make_projection(
            input_size=hidden_features,
            output_size=output_features,
            hidden_size=hidden_features * 2,
            num_layers=decoder_layers,
            activation=decoder_activation,
            activate_single_layer=False,
        )
        self.edge_decoder = self._make_projection(
            input_size=hidden_features,
            output_size=output_edge_features,
            hidden_size=hidden_features,
            num_layers=decoder_layers,
            activation=decoder_activation,
            activate_single_layer=False,
        )

        self.register_buffer(
            "_fine_edge_index",
            torch.empty((2, 0), dtype=torch.long),
            persistent=False,
        )
        # Unlike the topology buffers, this small signature is checkpointed.
        # It prevents a checkpoint from silently loading with a different mesh
        # ordering or coarsening configuration.
        self.register_buffer(
            "_hierarchy_fingerprint",
            torch.zeros(32, dtype=torch.uint8),
            persistent=True,
        )

        for level in range(num_scales - 1):
            self.register_buffer(
                f"_fine_to_coarse_{level}",
                torch.empty(0, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                f"_coarse_edge_index_{level}",
                torch.empty((2, 0), dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                f"_crossing_fine_edge_ids_{level}",
                torch.empty(0, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                f"_crossing_to_coarse_edge_{level}",
                torch.empty(0, dtype=torch.long),
                persistent=False,
            )

    def _make_projection(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int,
        activation: str,
        activate_single_layer: bool = True,
    ) -> Module:
        # In the original model, zero encoder/decoder layers means no module.
        # A multiscale processor always needs hidden-width tensors, so zero is
        # interpreted as a single linear projection without an activation.
        if num_layers < 0:
            raise ValueError("Encoder and decoder layer counts cannot be negative.")
        return make_mlp(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            num_layers=max(1, num_layers),
            activation=(
                activation
                if num_layers > 1 or (num_layers == 1 and activate_single_layer)
                else None
            ),
            bias=False,
            device=self.device,
        )

    def initialize_hierarchy(
        self,
        fine_edge_index: Tensor,
        num_nodes: int,
        boundary_mask: Tensor = None,
    ) -> None:
        """Build and register the static hierarchy from a processed graph."""

        hierarchy = build_multiscale_hierarchy(fine_edge_index=fine_edge_index,
                                               num_nodes=num_nodes,
                                               num_scales=self.num_scales,
                                               coarsening_factor=self.coarsening_factor,
                                               boundary_mask=boundary_mask,
                                               debug_hierarchy=True,
                                               )

        model_device = next(self.parameters()).device
        self._fine_edge_index = fine_edge_index.detach().to(
            device=model_device,
            dtype=torch.long,
        )
        for level, scale in enumerate(hierarchy):
            setattr(
                self,
                f"_fine_to_coarse_{level}",
                scale.fine_to_coarse.to(model_device),
            )
            setattr(
                self,
                f"_coarse_edge_index_{level}",
                scale.coarse_edge_index.to(model_device),
            )
            setattr(
                self,
                f"_crossing_fine_edge_ids_{level}",
                scale.crossing_fine_edge_ids.to(model_device),
            )
            setattr(
                self,
                f"_crossing_to_coarse_edge_{level}",
                scale.crossing_to_coarse_edge.to(model_device),
            )

        self._num_nodes_per_scale = [hierarchy[0].num_fine_nodes]
        self._num_edges_per_scale = [hierarchy[0].num_fine_edges]
        for scale in hierarchy:
            self._num_nodes_per_scale.append(scale.num_coarse_nodes)
            self._num_edges_per_scale.append(scale.num_coarse_edges)
        self._hierarchy_fingerprint = self._compute_hierarchy_fingerprint(
            hierarchy,
            fine_edge_index,
        ).to(model_device)
        self._hierarchy_initialized = True

    def _compute_hierarchy_fingerprint(
        self,
        hierarchy,
        fine_edge_index: Tensor,
    ) -> Tensor:
        """Hash topology and coarsening settings for checkpoint validation."""

        hasher = hashlib.sha256()
        metadata = (
            self.num_scales,
            self.coarsening_factor,
            int(fine_edge_index.shape[1]),
        )
        hasher.update(repr(metadata).encode("utf-8"))

        tensors = [fine_edge_index]
        for scale in hierarchy:
            tensors.extend(
                [
                    scale.fine_to_coarse,
                    scale.coarse_edge_index,
                    scale.crossing_fine_edge_ids,
                    scale.crossing_to_coarse_edge,
                ]
            )
        for tensor in tensors:
            cpu_tensor = tensor.detach().to(device="cpu", dtype=torch.long).contiguous()
            hasher.update(repr(tuple(cpu_tensor.shape)).encode("utf-8"))
            hasher.update(cpu_tensor.numpy().tobytes())

        return torch.tensor(list(hasher.digest()), dtype=torch.uint8)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Reject weights trained with a different static hierarchy."""

        checkpoint_fingerprint = state_dict.get("_hierarchy_fingerprint")
        if checkpoint_fingerprint is not None and self._hierarchy_initialized:
            checkpoint_fingerprint = checkpoint_fingerprint.to(
                self._hierarchy_fingerprint.device
            )
            if not torch.equal(
                checkpoint_fingerprint,
                self._hierarchy_fingerprint,
            ):
                raise RuntimeError(
                    "Checkpoint hierarchy does not match the current processed "
                    "graph, edge ordering, num_scales, or coarsening_factor."
                )
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def get_hierarchy_summary(self) -> dict:
        """Return small, log-friendly topology counts for each scale."""

        if not self._hierarchy_initialized:
            return {}
        return {
            "num_scales": self.num_scales,
            "nodes_per_scale": list(self._num_nodes_per_scale),
            "edges_per_scale": list(self._num_edges_per_scale),
            "coarsening_factor": self.coarsening_factor,
        }

    @staticmethod
    def _scatter_mean(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
        """Dependency-free differentiable mean aggregation along dimension 0."""

        output_shape = (dim_size,) + tuple(src.shape[1:])
        output = src.new_zeros(output_shape)
        if src.shape[0] == 0:
            return output

        output.index_add_(0, index, src)
        count_shape = (dim_size,) + (1,) * (src.ndim - 1)
        counts = src.new_zeros(count_shape)
        counts.index_add_(0, index, src.new_ones((src.shape[0],) + (1,) * (src.ndim - 1)))
        return output / counts.clamp_min(1)

    def _expand_hierarchy_level(
        self,
        level: int,
        batch_size: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, int, int]:
        """Repeat one static hierarchy with correct per-graph offsets."""

        fine_to_coarse = getattr(self, f"_fine_to_coarse_{level}")
        coarse_edge_index = getattr(self, f"_coarse_edge_index_{level}")
        crossing_ids = getattr(self, f"_crossing_fine_edge_ids_{level}")
        crossing_groups = getattr(self, f"_crossing_to_coarse_edge_{level}")

        num_fine_nodes = self._num_nodes_per_scale[level]
        num_coarse_nodes = self._num_nodes_per_scale[level + 1]
        num_fine_edges = self._num_edges_per_scale[level]
        num_coarse_edges = self._num_edges_per_scale[level + 1]

        device = fine_to_coarse.device
        graph_ids = torch.arange(batch_size, device=device, dtype=torch.long)

        expanded_mapping = fine_to_coarse.repeat(batch_size)
        expanded_mapping = expanded_mapping + (graph_ids * num_coarse_nodes).repeat_interleave(num_fine_nodes)

        if num_coarse_edges > 0:
            expanded_coarse_edges = coarse_edge_index.unsqueeze(0)
            expanded_coarse_edges = expanded_coarse_edges + (
                graph_ids * num_coarse_nodes
            ).view(batch_size, 1, 1)
            expanded_coarse_edges = expanded_coarse_edges.permute(1, 0, 2).reshape(
                2, batch_size * num_coarse_edges
            )
        else:
            expanded_coarse_edges = coarse_edge_index.new_empty((2, 0))

        num_crossing_edges = crossing_ids.numel()
        if num_crossing_edges > 0:
            expanded_crossing_ids = crossing_ids.repeat(batch_size)
            expanded_crossing_ids = expanded_crossing_ids + (
                graph_ids * num_fine_edges
            ).repeat_interleave(num_crossing_edges)

            expanded_crossing_groups = crossing_groups.repeat(batch_size)
            expanded_crossing_groups = expanded_crossing_groups + (
                graph_ids * num_coarse_edges
            ).repeat_interleave(num_crossing_edges)
        else:
            expanded_crossing_ids = crossing_ids.new_empty(0)
            expanded_crossing_groups = crossing_groups.new_empty(0)

        return (
            expanded_mapping,
            expanded_coarse_edges,
            expanded_crossing_ids,
            expanded_crossing_groups,
            batch_size * num_coarse_nodes,
            batch_size * num_coarse_edges,
        )

    def _validate_fine_batch(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> int:
        if not self._hierarchy_initialized:
            raise RuntimeError(
                "Multiscale hierarchy is not initialized. Call initialize_hierarchy() "
                "with a processed fine graph before training or inference."
            )

        fine_nodes = self._num_nodes_per_scale[0]
        fine_edges = self._num_edges_per_scale[0]
        if x.shape[0] % fine_nodes != 0:
            raise ValueError(
                f"Expected a whole-graph node batch divisible by {fine_nodes}, "
                f"but received {x.shape[0]} nodes."
            )

        batch_size = x.shape[0] // fine_nodes
        expected_edges = batch_size * fine_edges
        if edge_attr.shape[0] != expected_edges or edge_index.shape != (2, expected_edges):
            raise ValueError(
                f"Expected {expected_edges} fine edges for batch size {batch_size}, "
                f"but received edge_index={tuple(edge_index.shape)} and "
                f"edge_attr={tuple(edge_attr.shape)}."
            )

        graph_ids = torch.arange(
            batch_size,
            device=self._fine_edge_index.device,
            dtype=torch.long,
        )
        expected_edge_index = self._fine_edge_index.unsqueeze(0)
        expected_edge_index = expected_edge_index + (
            graph_ids * fine_nodes
        ).view(batch_size, 1, 1)
        expected_edge_index = expected_edge_index.permute(1, 0, 2).reshape(
            2,
            expected_edges,
        )
        if edge_index.device != expected_edge_index.device or not torch.equal(
            edge_index,
            expected_edge_index,
        ):
            raise ValueError(
                "Fine edge_index or edge ordering differs from the processed "
                "graph used to construct the multiscale hierarchy."
            )
        return batch_size

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        debug_message_passing: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Run one full fine -> coarse -> fine forward pass.

        Set ``debug_message_passing=True`` to print shapes and latent changes at
        every graph scale. Existing training code remains compatible because
        the debug argument defaults to False.
        """

        batch_size = self._validate_fine_batch(x, edge_index, edge_attr)

        # ================================================================
        # Encoder
        # ================================================================
        node_latent = self.node_encoder(x)
        edge_latent = self.edge_encoder(edge_attr)
        current_edge_index = edge_index

        if debug_message_passing:
            print("\n" + "#" * 70)
            print("MULTISCALE MESSAGE PASSING DEBUG")
            print("#" * 70)
            print("\nENCODER")
            print("Batch size        :", batch_size)
            print("Raw node input    :", tuple(x.shape))
            print("Raw edge input    :", tuple(edge_attr.shape))
            print("Encoded nodes     :", tuple(node_latent.shape))
            print("Encoded edges     :", tuple(edge_latent.shape))
            print("Fine edge_index   :", tuple(current_edge_index.shape))

        skip_nodes = []
        skip_edges = []
        skip_edge_indices = []
        expanded_hierarchy = []

        # ================================================================
        # Fine -> coarse
        # ================================================================
        for level, processor in enumerate(self.down_processors):
            node_latent, edge_latent = processor(
                node_latent,
                current_edge_index,
                edge_latent,
                debug_message_passing=debug_message_passing,
                scale_name=f"DOWN SCALE {level}",
            )

            # Save fine/intermediate-scale latent states for the U-Net skip.
            skip_nodes.append(node_latent)
            skip_edges.append(edge_latent)
            skip_edge_indices.append(current_edge_index)

            expanded = self._expand_hierarchy_level(level, batch_size)
            (
                fine_to_coarse,
                coarse_edge_index,
                crossing_ids,
                crossing_groups,
                num_coarse_nodes,
                num_coarse_edges,
            ) = expanded
            expanded_hierarchy.append(expanded)

            if debug_message_passing:
                internal_edge_count = edge_latent.shape[0] - crossing_ids.numel()

                print("\n" + "-" * 70)
                print(f"POOLING SCALE {level} -> {level + 1}")
                print("-" * 70)
                print("Nodes                 :", node_latent.shape[0], "->", num_coarse_nodes)
                print("Input edges           :", edge_latent.shape[0])
                print("Crossing input edges  :", crossing_ids.numel())
                print("Internal edges removed:", internal_edge_count)
                print("Coarse directed edges :", num_coarse_edges)

            # Node pooling: all fine nodes contribute to their parent cluster.
            node_latent = self._scatter_mean(
                node_latent,
                fine_to_coarse,
                dim_size=num_coarse_nodes,
            )

            # Edge pooling: only fine edges crossing between clusters survive.
            # Fine crossing edges mapping to the same directed coarse edge are
            # mean-pooled together.
            edge_latent = self._scatter_mean(
                edge_latent[crossing_ids],
                crossing_groups,
                dim_size=num_coarse_edges,
            )
            current_edge_index = coarse_edge_index

            if debug_message_passing:
                print("Pooled node latent     :", tuple(node_latent.shape))
                print("Pooled edge latent     :", tuple(edge_latent.shape))
                print("Coarse edge_index      :", tuple(current_edge_index.shape))

                if node_latent.shape[0] > 0:
                    print(
                        "Example pooled node     :",
                        node_latent[0, :5].detach().cpu().tolist(),
                    )
                if edge_latent.shape[0] > 0:
                    print(
                        "Example pooled edge     :",
                        edge_latent[0, :5].detach().cpu().tolist(),
                    )

        # ================================================================
        # Coarsest scale / bottleneck
        # ================================================================
        node_latent, edge_latent = self.bottleneck_processor(
            node_latent,
            current_edge_index,
            edge_latent,
            debug_message_passing=debug_message_passing,
            scale_name=f"BOTTLENECK SCALE {self.num_scales - 1}",
        )

        # ================================================================
        # Coarse -> fine
        # ================================================================
        for level in reversed(range(self.num_scales - 1)):
            (
                fine_to_coarse,
                _,
                crossing_ids,
                crossing_groups,
                _,
                _,
            ) = expanded_hierarchy[level]

            if debug_message_passing:
                print("\n" + "-" * 70)
                print(f"UPSAMPLING SCALE {level + 1} -> {level}")
                print("-" * 70)
                print("Current coarse nodes :", tuple(node_latent.shape))
                print("Current coarse edges :", tuple(edge_latent.shape))

            # Each fine node receives its parent coarse-node latent, followed by
            # a learnable projection.
            upsampled_nodes = self.node_upsamplers[level](
                node_latent[fine_to_coarse]
            )

            # Only fine edges that crossed coarse cells have a coarse parent.
            # Internal fine edges start with zero coarse contribution; their
            # fine-scale information is still available through the skip path.
            upsampled_edges = edge_latent.new_zeros(skip_edges[level].shape)
            if crossing_ids.numel() > 0:
                upsampled_edges = upsampled_edges.index_copy(
                    0,
                    crossing_ids,
                    edge_latent[crossing_groups],
                )
            upsampled_edges = self.edge_upsamplers[level](upsampled_edges)

            if debug_message_passing:
                print("Upsampled nodes      :", tuple(upsampled_nodes.shape))
                print("Upsampled edges      :", tuple(upsampled_edges.shape))
                print("Skip nodes           :", tuple(skip_nodes[level].shape))
                print("Skip edges           :", tuple(skip_edges[level].shape))
                print("Crossing fine edges  :", crossing_ids.numel())

            # Fuse the local skip representation with the upsampled broader
            # spatial context.
            if self.skip_connections:
                node_fusion_input = torch.cat(
                    [skip_nodes[level], upsampled_nodes],
                    dim=-1,
                )
                edge_fusion_input = torch.cat(
                    [skip_edges[level], upsampled_edges],
                    dim=-1,
                )
            else:
                node_fusion_input = upsampled_nodes
                edge_fusion_input = upsampled_edges

            node_latent = self.node_fusions[level](node_fusion_input)
            edge_latent = self.edge_fusions[level](edge_fusion_input)

            if debug_message_passing:
                print("Node fusion input    :", tuple(node_fusion_input.shape))
                print("Edge fusion input    :", tuple(edge_fusion_input.shape))
                print("Fused node latent    :", tuple(node_latent.shape))
                print("Fused edge latent    :", tuple(edge_latent.shape))

            # Return to the exact topology saved on the downward path and run
            # message passing again at that resolution.
            current_edge_index = skip_edge_indices[level]
            node_latent, edge_latent = self.up_processors[level](
                node_latent,
                current_edge_index,
                edge_latent,
                debug_message_passing=debug_message_passing,
                scale_name=f"UP SCALE {level}",
            )

        # ================================================================
        # Decoder
        # ================================================================
        node_output = self.node_decoder(node_latent)
        edge_output = self.edge_decoder(edge_latent)

        if debug_message_passing:
            print("\nDECODER")
            print("Final fine node latent:", tuple(node_latent.shape))
            print("Final fine edge latent:", tuple(edge_latent.shape))
            print("Node output           :", tuple(node_output.shape))
            print("Edge output           :", tuple(edge_output.shape))
            print("\n" + "#" * 70)
            print("FORWARD PASS COMPLETE")
            print("#" * 70, flush=True)

        # Keep edge_output signed. Negative face flow is physically meaningful.
        return node_output, edge_output

