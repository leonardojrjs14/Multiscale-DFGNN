import numpy as np
import torch

from utils.multiscale_utils import build_multiscale_hierarchy


CONSTANT_VALUES_PATH = (
    "/home/i/i0002673/projects/"
    "dual_flood_gnn-master/data/datasets/"
    "processed/constant_values.npz"
)

BOUNDARY_MASK_PATH = (
    "/home/i/i0002673/projects/"
    "dual_flood_gnn-master/data/datasets/"
    "processed/boundary_condition_masks.npz"
)

NUM_SCALES = 3
COARSENING_FACTOR = 4


print("=" * 70)
print("MULTISCALE TOPOLOGY DEBUG")
print("=" * 70)


# -------------------------------------------------
# 1. Load static graph only
# -------------------------------------------------

data = np.load(CONSTANT_VALUES_PATH)

edge_index = torch.tensor(
    data["edge_index"],
    dtype=torch.long,
)

static_nodes = data["static_nodes"]

num_nodes = static_nodes.shape[0]


print("Fine nodes :", num_nodes)
print("Fine edges :", edge_index.shape[1])


# -------------------------------------------------
# 2. Boundary mask
# -------------------------------------------------

boundary_data = np.load(BOUNDARY_MASK_PATH)

print("\nBoundary file keys:")
print(boundary_data.files)

boundary_mask = torch.tensor(
    boundary_data["boundary_nodes_mask"],
    dtype=torch.bool,
)

print("Boundary mask shape:", tuple(boundary_mask.shape))
print("Number of boundary nodes:", int(boundary_mask.sum()))

# -------------------------------------------------
# 3. Build hierarchy
# -------------------------------------------------

hierarchy = build_multiscale_hierarchy(
    fine_edge_index=edge_index,
    num_nodes=num_nodes,
    num_scales=NUM_SCALES,
    coarsening_factor=COARSENING_FACTOR,
    boundary_mask=boundary_mask,
    debug_hierarchy=True,
)


# -------------------------------------------------
# 4. Summary
# -------------------------------------------------

print("\n" + "=" * 70)
print("FINAL HIERARCHY SUMMARY")
print("=" * 70)

print(
    f"Scale 0: "
    f"{num_nodes} nodes, "
    f"{edge_index.shape[1]} edges"
)

for level, scale in enumerate(hierarchy):

    print(
        f"Scale {level + 1}: "
        f"{scale.num_coarse_nodes} nodes, "
        f"{scale.num_coarse_edges} edges"
    )

print("=" * 70)
print("DEBUG COMPLETE — NO TRAINING PERFORMED")
print("=" * 70)


