import torch

from data import dataset_factory
from models import model_factory
from utils import file_utils, model_utils, Logger


CONFIG_PATH = "configs/config.yaml"
MODEL_NAME = "MultiScaleDUALFloodGNN"
DEVICE = "cpu"


print("=" * 70)
print("MULTISCALE ONE-SAMPLE MESSAGE PASSING DEBUG")
print("=" * 70)


# ============================================================
# 1. Load config
# ============================================================

config = file_utils.read_yaml_file(CONFIG_PATH)

dataset_parameters = config["dataset_parameters"]
training_dataset_parameters = dataset_parameters["training"]
loss_parameters = config["loss_func_parameters"]

logger = Logger(log_path=None)


# ============================================================
# 2. Build dataset config
# ============================================================

dataset_config = {
    "mode": "train",

    "dataset_summary_file":
        training_dataset_parameters["dataset_summary_file"],

    "event_stats_file":
        training_dataset_parameters["event_stats_file"],

    "root_dir":
        dataset_parameters["root_dir"],

    "nodes_shp_file":
        dataset_parameters["nodes_shp_file"],

    "edges_shp_file":
        dataset_parameters["edges_shp_file"],

    "dem_file":
        dataset_parameters["dem_file"],

    "features_stats_file":
        dataset_parameters["features_stats_file"],

    "previous_timesteps":
        dataset_parameters["previous_timesteps"],

    "normalize":
        dataset_parameters["normalize"],

    "timestep_interval":
        dataset_parameters["timestep_interval"],

    "spin_up_time":
        dataset_parameters["spin_up_time"],

    "time_from_peak":
        dataset_parameters["time_from_peak"],

    "inflow_boundary_nodes":
        dataset_parameters["inflow_boundary_nodes"],

    "outflow_boundary_nodes":
        dataset_parameters["outflow_boundary_nodes"],

    "with_global_mass_loss":
        loss_parameters["use_global_mass_loss"],

    "with_local_mass_loss":
        loss_parameters["use_local_mass_loss"],

    "debug": False,
    "logger": logger,

    # IMPORTANT:
    # Do NOT preprocess every flood event again.
    "force_reload": False,
}


# ============================================================
# 3. Load existing processed dataset
# ============================================================

storage_mode = dataset_parameters["storage_mode"]

dataset = dataset_factory(
    storage_mode,
    autoregressive=False,
    **dataset_config,
)

print("\nDataset loaded.")
print("Number of samples:", len(dataset))


# ============================================================
# 4. Take ONE sample only
# ============================================================

sample = dataset[0]

print("\n" + "=" * 70)
print("ONE SAMPLE")
print("=" * 70)

print("Timestep   :", sample.timestep)
print("x          :", tuple(sample.x.shape))
print("edge_index :", tuple(sample.edge_index.shape))
print("edge_attr  :", tuple(sample.edge_attr.shape))
print("y          :", tuple(sample.y.shape))
print("y_edge     :", tuple(sample.y_edge.shape))


# ============================================================
# 5. Create model
# ============================================================

model_params = config["model_parameters"][MODEL_NAME]

base_model_params = {
    "static_node_features":
        dataset.num_static_node_features,

    "dynamic_node_features":
        dataset.num_dynamic_node_features,

    "static_edge_features":
        dataset.num_static_edge_features,

    "dynamic_edge_features":
        dataset.num_dynamic_edge_features,

    "previous_timesteps":
        dataset.previous_timesteps,

    "device":
        DEVICE,
}

model_config = {
    **model_params,
    **base_model_params,
}

model = model_factory(
    MODEL_NAME,
    **model_config,
)

model = model.to(DEVICE)

print("\nModel created:")
print(MODEL_NAME)

print("\nModel configuration:")
for key, value in model_config.items():
    print(f"{key}: {value}")


# ============================================================
# 6. Initialize multiscale hierarchy
# ============================================================

print("\n" + "=" * 70)
print("INITIALIZING MULTISCALE HIERARCHY")
print("=" * 70)

hierarchy_summary = model_utils.initialize_model_for_dataset(
    model,
    dataset,
)

print("\nHierarchy summary:")
print(hierarchy_summary)


# ============================================================
# 7. Move ONE sample to device
# ============================================================

x = sample.x.to(DEVICE)
edge_index = sample.edge_index.to(DEVICE)
edge_attr = sample.edge_attr.to(DEVICE)


# ============================================================
# 8. Run ONE forward pass
# ============================================================

print("\n" + "=" * 70)
print("STARTING ONE FORWARD PASS")
print("=" * 70)

model.eval()

with torch.no_grad():

    node_output, edge_output = model(
        x,
        edge_index,
        edge_attr,
        debug_message_passing=True,
    )


# ============================================================
# 9. Final outputs
# ============================================================

print("\n" + "=" * 70)
print("FINAL OUTPUT")
print("=" * 70)

print(
    "Node prediction shape:",
    tuple(node_output.shape)
)

print(
    "Edge prediction shape:",
    tuple(edge_output.shape)
)

print("\nFirst 5 node predictions:")

print(
    node_output[:5]
    .detach()
    .cpu()
)

print("\nFirst 5 edge predictions:")

print(
    edge_output[:5]
    .detach()
    .cpu()
)


print("\n" + "=" * 70)
print("DEBUG COMPLETE")
print("NO TRAINING")
print("NO LOSS")
print("NO BACKPROPAGATION")
print("=" * 70)
