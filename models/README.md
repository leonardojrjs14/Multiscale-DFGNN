# models

Contains different GNN model architectures.

### Overview

| Filename | Class Name | Description |
|---|---|---|
| \_\_init\_\_.py | N/A | Contains the model_factory function which loads the proper model class based on the given arguments. |
| base_model.py | BaseModel | Base class for all models. Defines important instance fields used by all model classes. |
| dual_flood_gnn.py | DUALFloodGNN | Node and edge prediction model. |
| node_edge_gnn_attn.py | NodeEdgeGNNAttn | Node and edge prediction model with attention mechanism (prototype). |
| node_edge_gnn_transformer.py | NodeEdgeGNNTransformer | Graph transformer with node and edge prediction model (prototype). |
| node_gnn.py | NodeGNN | Node only prediction model based on DUALFloodGNN. |
| edge_gnn.py | EdgeGNN | Edge only prediction model based on DUALFloodGNN. |
| base_node_model.py | BaseNodeModel | Base class for all models that only perform node prediction. |
| base_edge_model.py | BaseEdgeModel | Base class for all models that only perform edge prediction. |
| gcn.py | GCN, EdgeGCN | Graph Convolution Network (See [here](https://arxiv.org/abs/1609.02907)) |
| gat.py | GAT, EdgeGAT | Graph Attention Network (See [here](https://arxiv.org/abs/1710.10903v3)) |
| gin.py | GIN, EdgeGIN | Graph Isomorphism Network (See [here](https://arxiv.org/abs/1810.00826)) |
| gine.py | GINE, EdgeGINE | Graph Isomorphism Network with Edges (See [here](https://arxiv.org/abs/1905.12265)) |
| graphsage.py | GraphSAGE, EdgeGraphSAGE | Graph SAmple and aggreGatE (See [here](https://arxiv.org/abs/1706.02216)) |
