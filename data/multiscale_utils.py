"""Utilities for constructing deterministic multiscale graph hierarchies.

The flood-event dataset contains one processed HEC-RAS graph.  A hierarchy is
therefore built once from that graph and reused for every event and timestep.
Only topology is coarsened here; physical targets and physics losses remain on
the original fine graph.
"""

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class ScaleHierarchy:
    """Topology connecting one fine graph level to the next coarse level."""

    fine_to_coarse: Tensor
    coarse_edge_index: Tensor
    crossing_fine_edge_ids: Tensor
    crossing_to_coarse_edge: Tensor
    coarse_boundary_mask: Tensor
    num_fine_nodes: int
    num_coarse_nodes: int
    num_fine_edges: int
    num_coarse_edges: int


def greedy_connected_partition(
    edge_index: Tensor,
    num_nodes: int,
    target_cluster_size: int,
    boundary_mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Group adjacent nodes into deterministic, connected coarse cells.

    The algorithm walks nodes and neighbours in ascending index order.  Nodes
    marked by ``boundary_mask`` become singleton clusters so an appended ghost
    boundary node can never be merged with an ordinary interior cell.

    Returns:
        ``fine_to_coarse`` and a boolean mask for coarse boundary nodes.
    """

    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")
    if target_cluster_size < 2:
        raise ValueError("target_cluster_size must be at least 2.")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")

    edge_index_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    if edge_index_cpu.numel() > 0:
        if edge_index_cpu.min().item() < 0 or edge_index_cpu.max().item() >= num_nodes:
            raise ValueError("edge_index contains a node outside [0, num_nodes).")

    if boundary_mask is None:
        boundary_mask_cpu = torch.zeros(num_nodes, dtype=torch.bool)
    else:
        boundary_mask_cpu = torch.as_tensor(boundary_mask, dtype=torch.bool, device="cpu")
        if boundary_mask_cpu.ndim != 1 or boundary_mask_cpu.numel() != num_nodes:
            raise ValueError("boundary_mask must contain one value per fine node.")

    adjacency = [set() for _ in range(num_nodes)]
    for source, target in edge_index_cpu.t().tolist():
        if source == target:
            continue
        # Coarsening uses undirected connectivity even when hydraulic edges are
        # stored with a physical direction.
        adjacency[source].add(target)
        adjacency[target].add(source)

    assignment = [-1] * num_nodes
    coarse_boundary = []
    cluster_id = 0

    for start_node in range(num_nodes):
        if assignment[start_node] >= 0:
            continue

        if boundary_mask_cpu[start_node]:
            assignment[start_node] = cluster_id
            coarse_boundary.append(True)
            cluster_id += 1
            continue

        queue = deque([start_node])
        queued = {start_node}
        cluster_members = []

        while queue and len(cluster_members) < target_cluster_size:
            node = queue.popleft()
            if assignment[node] >= 0 or boundary_mask_cpu[node]:
                continue

            assignment[node] = cluster_id
            cluster_members.append(node)

            for neighbour in sorted(adjacency[node]):
                if (
                    assignment[neighbour] < 0
                    and not boundary_mask_cpu[neighbour]
                    and neighbour not in queued
                ):
                    queue.append(neighbour)
                    queued.add(neighbour)

        if not cluster_members:
            raise RuntimeError(f"Unable to assign fine node {start_node} to a coarse node.")

        coarse_boundary.append(False)
        cluster_id += 1

    fine_to_coarse = torch.tensor(assignment, dtype=torch.long)
    coarse_boundary_mask = torch.tensor(coarse_boundary, dtype=torch.bool)

    if (fine_to_coarse < 0).any():
        raise RuntimeError("Every fine node must map to exactly one coarse node.")
    if fine_to_coarse.unique().numel() != cluster_id:
        raise RuntimeError("Coarse node identifiers must be contiguous and used.")

    return fine_to_coarse, coarse_boundary_mask


def build_coarse_topology(
    fine_edge_index: Tensor,
    fine_to_coarse: Tensor,
    num_coarse_nodes: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Contract a fine graph while preserving directed coarse-edge order.

    Fine edges internal to one cluster disappear at the coarse level.  Parallel
    crossing edges with the same directed coarse endpoints are coalesced.  The
    returned mappings are later used to mean-pool their latent edge embeddings.
    """

    fine_edge_index_cpu = fine_edge_index.detach().to(device="cpu", dtype=torch.long)
    fine_to_coarse_cpu = fine_to_coarse.detach().to(device="cpu", dtype=torch.long)

    if fine_edge_index_cpu.ndim != 2 or fine_edge_index_cpu.shape[0] != 2:
        raise ValueError("fine_edge_index must have shape [2, num_edges].")
    if fine_to_coarse_cpu.ndim != 1:
        raise ValueError("fine_to_coarse must be one-dimensional.")
    if fine_to_coarse_cpu.numel() == 0 or (fine_to_coarse_cpu < 0).any():
        raise ValueError("fine_to_coarse must assign every fine node.")
    if fine_edge_index_cpu.numel() > 0:
        if fine_edge_index_cpu.min().item() < 0 or fine_edge_index_cpu.max().item() >= fine_to_coarse_cpu.numel():
            raise ValueError("fine_edge_index references a missing fine node.")

    inferred_num_coarse_nodes = int(fine_to_coarse_cpu.max().item()) + 1
    if num_coarse_nodes is None:
        num_coarse_nodes = inferred_num_coarse_nodes
    if num_coarse_nodes != inferred_num_coarse_nodes:
        raise ValueError("num_coarse_nodes does not match fine_to_coarse.")

    coarse_edge_lookup = {}
    coarse_edges = []
    crossing_fine_edge_ids = []
    crossing_to_coarse_edge = []

    for fine_edge_id, (fine_source, fine_target) in enumerate(fine_edge_index_cpu.t().tolist()):
        coarse_source = int(fine_to_coarse_cpu[fine_source])
        coarse_target = int(fine_to_coarse_cpu[fine_target])

        if coarse_source == coarse_target:
            continue

        edge_key = (coarse_source, coarse_target)
        if edge_key not in coarse_edge_lookup:
            coarse_edge_lookup[edge_key] = len(coarse_edges)
            coarse_edges.append(edge_key)

        crossing_fine_edge_ids.append(fine_edge_id)
        crossing_to_coarse_edge.append(coarse_edge_lookup[edge_key])

    if coarse_edges:
        coarse_edge_index = torch.tensor(coarse_edges, dtype=torch.long).t().contiguous()
    else:
        coarse_edge_index = torch.empty((2, 0), dtype=torch.long)

    return (
        coarse_edge_index,
        torch.tensor(crossing_fine_edge_ids, dtype=torch.long),
        torch.tensor(crossing_to_coarse_edge, dtype=torch.long),
    )


def build_multiscale_hierarchy(
    fine_edge_index: Tensor,
    num_nodes: int,
    num_scales: int = 2,
    coarsening_factor: int = 4,
    boundary_mask: Optional[Tensor] = None,
) -> List[ScaleHierarchy]:
    """Build ``num_scales - 1`` deterministic fine-to-coarse mappings."""

    if num_scales < 2:
        raise ValueError("A multiscale graph requires at least two scales.")
    if coarsening_factor < 2:
        raise ValueError("coarsening_factor must be at least 2.")

    current_edge_index = fine_edge_index.detach().to(device="cpu", dtype=torch.long)
    current_num_nodes = int(num_nodes)
    current_boundary_mask = (
        torch.zeros(current_num_nodes, dtype=torch.bool)
        if boundary_mask is None
        else torch.as_tensor(boundary_mask, dtype=torch.bool, device="cpu")
    )
    if current_boundary_mask.ndim != 1 or current_boundary_mask.numel() != current_num_nodes:
        raise ValueError("boundary_mask must contain one value per fine node.")

    hierarchy = []
    for level in range(num_scales - 1):
        fine_to_coarse, coarse_boundary_mask = greedy_connected_partition(
            edge_index=current_edge_index,
            num_nodes=current_num_nodes,
            target_cluster_size=coarsening_factor,
            boundary_mask=current_boundary_mask,
        )
        num_coarse_nodes = int(fine_to_coarse.max().item()) + 1

        if num_coarse_nodes >= current_num_nodes:
            raise ValueError(
                f"Scale {level} did not reduce the graph ({current_num_nodes} nodes). "
                "Use fewer scales or a graph with more non-boundary nodes."
            )

        coarse_edge_index, crossing_ids, crossing_groups = build_coarse_topology(
            fine_edge_index=current_edge_index,
            fine_to_coarse=fine_to_coarse,
            num_coarse_nodes=num_coarse_nodes,
        )

        hierarchy.append(
            ScaleHierarchy(
                fine_to_coarse=fine_to_coarse,
                coarse_edge_index=coarse_edge_index,
                crossing_fine_edge_ids=crossing_ids,
                crossing_to_coarse_edge=crossing_groups,
                coarse_boundary_mask=coarse_boundary_mask,
                num_fine_nodes=current_num_nodes,
                num_coarse_nodes=num_coarse_nodes,
                num_fine_edges=current_edge_index.shape[1],
                num_coarse_edges=coarse_edge_index.shape[1],
            )
        )

        current_edge_index = coarse_edge_index
        current_num_nodes = num_coarse_nodes
        current_boundary_mask = coarse_boundary_mask

    return hierarchy
