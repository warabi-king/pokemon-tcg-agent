"""共有libcg＋試合横断NN batch backendのテスト。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import os
import queue
import subprocess
import sys
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from batched_tournament import (
    _CudaEnsembleEvaluator,
    _collect_ready_messages,
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
    def test_ready_worker_messages_are_collected_without_waiting(self) -> None:
        first = object()
        second = object()
        third = object()
        requests: queue.Queue[object] = queue.Queue()
        requests.put(second)
        requests.put(third)

        messages = _collect_ready_messages(requests, first)

        self.assertEqual(messages, [first, second, third])
        with self.assertRaises(queue.Empty):
            requests.get_nowait()

    def test_ready_worker_messages_return_immediately_when_queue_is_empty(self) -> None:
        first = object()
        requests: queue.Queue[object] = queue.Queue()

        messages = _collect_ready_messages(requests, first)

        self.assertEqual(messages, [first])

    def test_lightweight_worker_import_does_not_load_torch(self) -> None:
        script = """
import sys
from pathlib import Path
sys.path.insert(0, r'{tools}')
import batched_tournament
assert batched_tournament.torch is None
runtime = batched_tournament._load_runtime(Path(r'{agent_src}'))
assert runtime.create_model is None
assert 'torch' not in sys.modules
print('lightweight-worker-ok')
""".format(
            tools=ROOT / "tools",
            agent_src=ROOT / "agents" / "rl_mcts_match_00" / "src",
        )
        env = os.environ.copy()
        env["PTCG_BATCHED_LIGHTWEIGHT_WORKER"] = "1"
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            check=False,
            env=env,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "lightweight-worker-ok")

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

    def test_enumerate_actions_reuses_cached_result(self) -> None:
        first = _enumerate_actions(option_count=12, select_count=3, limit=9)
        second = _enumerate_actions(option_count=12, select_count=3, limit=9)

        self.assertIs(first, second)

    def test_decoder_reused_options_keep_identical_sparse_rows(self) -> None:
        runtime = _load_runtime(ROOT / "agents" / "rl_mcts_match_00" / "src")
        from cg.api import OptionType

        observation = SimpleNamespace(
            current=SimpleNamespace(
                yourIndex=0,
                players=[SimpleNamespace(), SimpleNamespace()],
            ),
            select=SimpleNamespace(
                context=0,
                option=[
                    SimpleNamespace(type=OptionType.END),
                    SimpleNamespace(type=OptionType.YES),
                    SimpleNamespace(type=OptionType.NO),
                ],
            ),
        )

        sparse = runtime.get_decoder_input(
            observation,
            [[0, 1], [0, 2], [1, 2]],
        )

        self.assertEqual(sparse.index, [1, 2, 1, 3, 2, 3])
        self.assertEqual(sparse.value, [1.0] * 6)
        self.assertEqual(sparse.offset, [0, 2, 4])

    def test_sparse_offsets_are_padded_with_empty_bags(self) -> None:
        sparse = SimpleNamespace(index=[3, 5], offset=[0, 1])
        _pad_sparse_offsets(sparse, 4)
        self.assertEqual(sparse.offset, [0, 1, 2, 2])

    def test_worker_sparse_batches_are_rebased_when_merged(self) -> None:
        merged = _merge_combined_sparse(
            [
                (
                    np.asarray([3, 5], dtype=np.int32),
                    np.asarray([0.5, 1.5], dtype=np.float32),
                    np.asarray([0, 1], dtype=np.int32),
                ),
                (
                    np.asarray([7, 9, 11], dtype=np.int32),
                    np.asarray([2.0, 3.0, 4.0], dtype=np.float32),
                    np.asarray([0, 2], dtype=np.int32),
                ),
            ]
        )
        np.testing.assert_array_equal(merged[0], [3, 5, 7, 9, 11])
        np.testing.assert_allclose(merged[1], [0.5, 1.5, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(merged[2], [0, 1, 2, 4])

    def test_worker_decoder_rows_are_padded_independently(self) -> None:
        job = SimpleNamespace(
            decoder_words=2,
            batch_count=2,
            decoder=(
                np.asarray([1, 2, 3, 4, 5], dtype=np.int32),
                np.ones(5, dtype=np.float32),
                np.asarray([0, 1, 2, 4], dtype=np.int32),
            ),
        )
        padded = _pad_remote_decoder(job, 4)
        np.testing.assert_array_equal(padded[2], [0, 1, 2, 2, 2, 4, 5, 5])

    def test_model_axis_padding_adds_empty_rows(self) -> None:
        padded = _pad_empty_sparse_rows(
            (
                np.asarray([3, 5], dtype=np.int32),
                np.asarray([0.5, 1.5], dtype=np.float32),
                np.asarray([0, 1], dtype=np.int32),
            ),
            current_rows=1,
            target_rows=3,
            words_per_row=2,
        )
        np.testing.assert_array_equal(padded[0], [3, 5])
        np.testing.assert_allclose(padded[1], [0.5, 1.5])
        np.testing.assert_array_equal(padded[2], [0, 1, 2, 2, 2, 2])


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
