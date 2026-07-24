"""共有libcg＋試合横断NN batch backendのテスト。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from batched_tournament import (
    _CudaEnsembleEvaluator,
    _enumerate_actions,
    _load_runtime,
    _merge_combined_sparse,
    _pad_empty_sparse_rows,
    _pad_sparse_offsets,
    _pad_remote_decoder,
    run_batched_tournament,
)
from batched_training import (
    BatchedTrainingAgent,
    collect_batched_training_samples,
)


class ActionBatchTest(unittest.TestCase):
    def test_model_parallel_embedding_bag_matches_pytorch(self) -> None:
        torch.manual_seed(7)
        weights = torch.randn(2, 9, 4)
        indices = torch.tensor(
            [[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]],
            dtype=torch.int64,
        )
        per_sample_weights = torch.randn(2, 6)
        offsets = torch.tensor(
            [[0, 2, 2, 5], [0, 1, 4, 4]],
            dtype=torch.int64,
        )

        actual = _CudaEnsembleEvaluator._embedding_bag(
            weights,
            indices,
            per_sample_weights,
            offsets,
        )
        expected = torch.stack(
            [
                torch.nn.functional.embedding_bag(
                    indices[index],
                    weights[index],
                    offsets[index],
                    mode="sum",
                    per_sample_weights=per_sample_weights[index],
                )
                for index in range(2)
            ]
        )
        torch.testing.assert_close(actual, expected)

    def test_enumerate_actions_respects_limit(self) -> None:
        actions = _enumerate_actions(option_count=10, select_count=2, limit=7)
        self.assertEqual(len(actions), 7)
        self.assertEqual(actions[0], [0, 1])
        self.assertEqual(len({tuple(action) for action in actions}), 7)

    def test_sparse_offsets_are_padded_with_empty_bags(self) -> None:
        sparse = SimpleNamespace(index=[3, 5], offset=[0, 1])
        _pad_sparse_offsets(sparse, 4)
        self.assertEqual(sparse.offset, [0, 1, 2, 2])

    def test_worker_sparse_batches_are_rebased_when_merged(self) -> None:
        merged = _merge_combined_sparse(
            [
                ([3, 5], [0.5, 1.5], [0, 1]),
                ([7, 9, 11], [2.0, 3.0, 4.0], [0, 2]),
            ]
        )
        self.assertEqual(merged[0], [3, 5, 7, 9, 11])
        self.assertEqual(merged[1], [0.5, 1.5, 2.0, 3.0, 4.0])
        self.assertEqual(merged[2], [0, 1, 2, 4])

    def test_worker_decoder_rows_are_padded_independently(self) -> None:
        job = SimpleNamespace(
            decoder_words=2,
            batch_count=2,
            decoder=([1, 2, 3, 4, 5], [1.0] * 5, [0, 1, 2, 4]),
        )
        padded = _pad_remote_decoder(job, 4)
        self.assertEqual(padded[2], [0, 1, 2, 2, 2, 4, 5, 5])

    def test_model_axis_padding_adds_empty_rows(self) -> None:
        padded = _pad_empty_sparse_rows(
            ([3, 5], [0.5, 1.5], [0, 1]),
            current_rows=1,
            target_rows=3,
            words_per_row=2,
        )
        self.assertEqual(padded[0], [3, 5])
        self.assertEqual(padded[1], [0.5, 1.5])
        self.assertEqual(padded[2], [0, 1, 2, 2, 2, 2])


class BatchedTournamentIntegrationTest(unittest.TestCase):
    def test_cuda_stream_backend_rejects_cpu_device(self) -> None:
        with self.assertRaisesRegex(ValueError, "CUDA device"):
            run_batched_tournament(
                [],
                [],
                1,
                device_name="cpu",
                parallel_cuda_models=True,
            )

    def test_cuda_ensemble_backend_rejects_cpu_device(self) -> None:
        with self.assertRaisesRegex(ValueError, "CUDA device"):
            run_batched_tournament(
                [],
                [],
                1,
                device_name="cpu",
                cuda_ensemble_models=True,
            )

    def test_two_matches_share_one_model_batch(self) -> None:
        src = ROOT / "agents" / "rl_mcts_r_robin1" / "src"
        specs = [
            SimpleNamespace(
                name=name,
                agent_path=src / "main.py",
                deck_path=src / "deck.csv",
            )
            for name in ("a", "b")
        ]
        output = run_batched_tournament(
            specs,
            [("a", "b")],
            2,
            device_name="cpu",
            batch_size=8,
            lanes=2,
            search_count=1,
            max_selections=500,
            seed=123,
        )

        self.assertEqual(len(output.results), 2)
        self.assertTrue(all(result.error is None for result in output.results))
        self.assertTrue(all(result.result in (0, 1, 2) for result in output.results))
        self.assertGreaterEqual(output.profile.max_batch_size, 2)
        self.assertGreater(output.profile.search_steps, 0)

    def test_training_collector_returns_unpadded_samples(self) -> None:
        src = ROOT / "agents" / "rl_mcts_r_robin1" / "src"
        runtime = _load_runtime(src)
        model = runtime.create_model()
        output = collect_batched_training_samples(
            [
                BatchedTrainingAgent(
                    name="a",
                    model=model,
                    deck=[
                        int(line)
                        for line in (src / "deck.csv").read_text().splitlines()
                        if line.strip()
                    ],
                )
            ],
            [("a", "a")],
            2,
            canonical_src=src,
            device=next(model.parameters()).device,
            batch_size=8,
            lanes=2,
            search_count=1,
            lambda_value=0.9,
            seed=321,
        )

        self.assertEqual(len(output.results), 2)
        self.assertTrue(all(result.error is None for result in output.results))
        self.assertGreater(len(output.samples["a"]), 0)
        for sample in output.samples["a"]:
            self.assertEqual(len(sample.policy), len(sample.sv_dec.offset))
            self.assertLessEqual(len(sample.policy), 64)


if __name__ == "__main__":
    unittest.main()
