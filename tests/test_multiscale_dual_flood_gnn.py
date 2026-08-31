import unittest

import torch
from torch_geometric.data import Batch, Data

from constants import NODE_EDGE_MODELS
from models import MultiScaleDUALFloodGNN, model_factory
from utils.multiscale_utils import (
    build_coarse_topology,
    build_multiscale_hierarchy,
    greedy_connected_partition,
)


def bidirectional_line_graph(num_nodes: int) -> torch.Tensor:
    edges = []
    for node in range(num_nodes - 1):
        edges.extend([(node, node + 1), (node + 1, node)])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


class MultiscaleTopologyTest(unittest.TestCase):
    def test_partition_is_connected_deterministic_and_boundary_safe(self):
        edge_index = bidirectional_line_graph(6)
        boundary_mask = torch.tensor([False, False, False, False, False, True])

        mapping, coarse_boundary = greedy_connected_partition(
            edge_index=edge_index,
            num_nodes=6,
            target_cluster_size=2,
            boundary_mask=boundary_mask,
        )
        repeated_mapping, repeated_boundary = greedy_connected_partition(
            edge_index=edge_index,
            num_nodes=6,
            target_cluster_size=2,
            boundary_mask=boundary_mask,
        )

        torch.testing.assert_close(mapping, torch.tensor([0, 0, 1, 1, 2, 3]))
        torch.testing.assert_close(mapping, repeated_mapping)
        torch.testing.assert_close(coarse_boundary, repeated_boundary)

        boundary_parent = mapping[-1].item()
        self.assertTrue(coarse_boundary[boundary_parent])
        self.assertEqual((mapping == boundary_parent).sum().item(), 1)

    def test_contracted_topology_has_no_self_loops_or_duplicates(self):
        edge_index = bidirectional_line_graph(6)
        mapping = torch.tensor([0, 0, 1, 1, 2, 3])
        coarse_edges, crossing_ids, crossing_groups = build_coarse_topology(
            edge_index,
            mapping,
        )

        expected = {
            (0, 1),
            (1, 0),
            (1, 2),
            (2, 1),
            (2, 3),
            (3, 2),
        }
        actual = {tuple(edge) for edge in coarse_edges.t().tolist()}
        self.assertEqual(actual, expected)
        self.assertEqual(coarse_edges.shape[1], len(actual))
        self.assertTrue(torch.all(coarse_edges[0] != coarse_edges[1]))
        self.assertEqual(crossing_ids.numel(), crossing_groups.numel())
        self.assertTrue(torch.all(crossing_groups < coarse_edges.shape[1]))

    def test_multiple_scales_reduce_the_graph(self):
        edge_index = bidirectional_line_graph(16)
        hierarchy = build_multiscale_hierarchy(
            fine_edge_index=edge_index,
            num_nodes=16,
            num_scales=3,
            coarsening_factor=2,
        )

        self.assertEqual(len(hierarchy), 2)
        self.assertGreater(hierarchy[0].num_fine_nodes, hierarchy[0].num_coarse_nodes)
        self.assertGreater(hierarchy[1].num_fine_nodes, hierarchy[1].num_coarse_nodes)
        self.assertEqual(
            hierarchy[0].num_coarse_nodes,
            hierarchy[1].num_fine_nodes,
        )


class MultiScaleDUALFloodGNNTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.num_nodes = 6
        self.edge_index = bidirectional_line_graph(self.num_nodes)
        self.num_edges = self.edge_index.shape[1]
        self.boundary_mask = torch.tensor(
            [False, False, False, False, False, True]
        )

    def make_model(self, coarsening_factor: int = 2):
        model = MultiScaleDUALFloodGNN(
            static_node_features=8,
            dynamic_node_features=3,
            static_edge_features=5,
            dynamic_edge_features=1,
            previous_timesteps=1,
            hidden_features=8,
            num_scales=2,
            coarsening_factor=coarsening_factor,
            layers_per_scale=1,
            activation="relu",
            residual=True,
            mlp_layers=2,
            skip_connections=True,
            encoder_layers=2,
            encoder_activation="relu",
            decoder_layers=2,
            decoder_activation="relu",
            device="cpu",
        )
        model.initialize_hierarchy(
            fine_edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            boundary_mask=self.boundary_mask,
        )
        return model

    def make_data(self, seed: int):
        generator = torch.Generator().manual_seed(seed)
        return Data(
            x=torch.randn(self.num_nodes, 14, generator=generator),
            edge_index=self.edge_index,
            edge_attr=torch.randn(self.num_edges, 7, generator=generator),
        )

    def test_factory_and_forward_contract(self):
        self.assertIn("MultiScaleDUALFloodGNN", NODE_EDGE_MODELS)
        model = model_factory(
            "MultiScaleDUALFloodGNN",
            static_node_features=8,
            dynamic_node_features=3,
            static_edge_features=5,
            dynamic_edge_features=1,
            previous_timesteps=1,
            hidden_features=8,
            num_scales=2,
            coarsening_factor=2,
            layers_per_scale=1,
            device="cpu",
        )
        self.assertIsInstance(model, MultiScaleDUALFloodGNN)

        model.initialize_hierarchy(
            self.edge_index,
            self.num_nodes,
            self.boundary_mask,
        )
        data = self.make_data(seed=1)
        node_delta, edge_delta = model(data.x, data.edge_index, data.edge_attr)

        self.assertEqual(node_delta.shape, (self.num_nodes, 1))
        self.assertEqual(edge_delta.shape, (self.num_edges, 1))
        self.assertTrue(torch.isfinite(node_delta).all())
        self.assertTrue(torch.isfinite(edge_delta).all())

    def test_pooling_values_and_gradients(self):
        values = torch.tensor(
            [[0.0], [2.0], [4.0], [6.0], [8.0], [10.0]],
            requires_grad=True,
        )
        mapping = torch.tensor([0, 0, 1, 1, 2, 2])
        pooled = MultiScaleDUALFloodGNN._scatter_mean(values, mapping, dim_size=3)
        torch.testing.assert_close(pooled, torch.tensor([[1.0], [5.0], [9.0]]))
        pooled.sum().backward()
        self.assertTrue(torch.isfinite(values.grad).all())

    def test_batched_output_equals_individual_outputs(self):
        model = self.make_model().eval()
        first = self.make_data(seed=10)
        second = self.make_data(seed=20)

        with torch.no_grad():
            first_node, first_edge = model(
                first.x,
                first.edge_index,
                first.edge_attr,
            )
            second_node, second_edge = model(
                second.x,
                second.edge_index,
                second.edge_attr,
            )

            batch = Batch.from_data_list([first, second])
            batch_node, batch_edge = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
            )

        torch.testing.assert_close(
            batch_node,
            torch.cat([first_node, second_node]),
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            batch_edge,
            torch.cat([first_edge, second_edge]),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_coarse_processor_receives_gradients(self):
        model = self.make_model().train()
        data = self.make_data(seed=30)
        node_delta, edge_delta = model(data.x, data.edge_index, data.edge_attr)
        (node_delta.square().mean() + edge_delta.square().mean()).backward()

        gradients = [
            parameter.grad
            for parameter in model.bottleneck_processor.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertTrue(any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients))

    def test_state_dict_round_trip_rebuilds_static_hierarchy(self):
        first_model = self.make_model().eval()
        data = self.make_data(seed=35)
        with torch.no_grad():
            expected_node, expected_edge = first_model(
                data.x,
                data.edge_index,
                data.edge_attr,
            )

        state = first_model.state_dict()
        self.assertFalse(any("fine_to_coarse" in key for key in state))

        second_model = self.make_model().eval()
        second_model.load_state_dict(state)
        with torch.no_grad():
            actual_node, actual_edge = second_model(
                data.x,
                data.edge_index,
                data.edge_attr,
            )

        torch.testing.assert_close(actual_node, expected_node)
        torch.testing.assert_close(actual_edge, expected_edge)

    def test_checkpoint_rejects_a_different_hierarchy(self):
        first_model = self.make_model(coarsening_factor=2)
        second_model = self.make_model(coarsening_factor=3)
        with self.assertRaisesRegex(RuntimeError, "Checkpoint hierarchy"):
            second_model.load_state_dict(first_model.state_dict())

    def test_rejects_partial_fine_graph_batches(self):
        model = self.make_model()
        data = self.make_data(seed=40)
        with self.assertRaisesRegex(ValueError, "whole-graph"):
            model(data.x[:-1], data.edge_index, data.edge_attr)

    def test_rejects_reordered_fine_edges(self):
        model = self.make_model()
        data = self.make_data(seed=45)
        permutation = torch.arange(self.num_edges - 1, -1, -1)
        with self.assertRaisesRegex(ValueError, "edge ordering"):
            model(
                data.x,
                data.edge_index[:, permutation],
                data.edge_attr[permutation],
            )


if __name__ == "__main__":
    unittest.main()
