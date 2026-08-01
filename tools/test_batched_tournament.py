"""共有libcg＋試合横断NN batch backendのテスト。"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace
import os
import queue
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from batched_tournament import (
    _CudaEnsembleEvaluator,
    _collect_ready_messages,
    _enumerate_actions,
    _evaluation_model_key,
    _is_setup_context,
    _load_participants,
    _load_runtime,
    _mps_sparse_input_batches,
    _mps_sparse_inputs,
    _merge_combined_sparse,
    _pad_empty_sparse_rows,
    _pad_sparse_offsets,
    _pad_remote_decoder,
    _set_result_from_remaining_prizes,
    _to_result,
    _write_training_episode,
    run_batched_tournament,
    run_worker_batched_tournament,
)
from deck_belief import (
    _candidate_rank,
    _reconcile_candidate,
    predict_full_deck,
    public_cards_by_player,
    sample_hidden_zones,
)
from batched_training import (
    BatchedTrainingAgent,
    collect_batched_training_samples,
)


class ActionBatchTest(unittest.TestCase):
    @staticmethod
    def _unfinished_session(
        remaining_prizes: tuple[int, int],
        *,
        swap: bool = False,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            observation={
                "current": {
                    "result": -1,
                    "players": [
                        {"prize": [None] * remaining_prizes[0]},
                        {"prize": [None] * remaining_prizes[1]},
                    ],
                }
            },
            request=SimpleNamespace(
                name0="a",
                name1="b",
                game_index=1,
                swap=swap,
            ),
            players=(
                SimpleNamespace(deck=tuple(range(60))),
                SimpleNamespace(deck=tuple(range(60))),
            ),
            selections=100,
            training_decisions=[],
        )

    def test_limit_result_uses_fewer_remaining_prizes(self) -> None:
        session = self._unfinished_session((2, 4))

        result = _set_result_from_remaining_prizes(session)

        self.assertEqual(result, 0)
        self.assertEqual(_to_result(session).result, 0)

    def test_limit_result_respects_swapped_agent_order(self) -> None:
        session = self._unfinished_session((2, 4), swap=True)

        _set_result_from_remaining_prizes(session)

        self.assertEqual(_to_result(session).result, 1)

    def test_limit_result_is_draw_when_remaining_prizes_are_equal(self) -> None:
        session = self._unfinished_session((3, 3))

        result = _set_result_from_remaining_prizes(session)

        self.assertEqual(result, 2)
        self.assertEqual(_to_result(session).result, 2)

    def test_limit_result_is_written_as_training_reward(self) -> None:
        session = self._unfinished_session((5, 1))
        _set_result_from_remaining_prizes(session)

        with tempfile.TemporaryDirectory() as directory:
            episode_path = _write_training_episode(session, Path(directory))
            episode = json.loads(episode_path.read_text(encoding="utf-8"))

        self.assertEqual(episode["rewards"], [-1, 1])

    def test_database_tie_break_uses_wins_only(self) -> None:
        lower_win_rate_more_wins = {"wins": 10, "win_rate": 0.1, "games": 10}
        higher_win_rate_fewer_wins = {"wins": 9, "win_rate": 0.9, "games": 9999}

        self.assertGreater(
            _candidate_rank(lower_win_rate_more_wins, 1),
            _candidate_rank(higher_win_rate_fewer_wins, 0),
        )

    def test_distinct_opponent_checkpoint_gets_distinct_model_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory)
            (src / "model.pth").touch()
            (src / "opponent_model.pth").touch()
            spec = SimpleNamespace(
                name="a",
                agent_path=src / "main.py",
                deck_path=ROOT / "agents" / "rl_mcts_r_robin1" / "src" / "deck.csv",
            )

            participant = _load_participants(
                [spec],
                runtime=SimpleNamespace(),
                device=SimpleNamespace(type="cpu"),
                load_models=False,
            )["a"]

        self.assertNotEqual(participant.model_key, participant.opponent_model_key)
        self.assertTrue(participant.model_key.endswith("model.pth"))
        self.assertTrue(participant.opponent_model_key.endswith("opponent_model.pth"))

    def test_setup_with_facedown_active_skips_tree_search(self) -> None:
        participant = SimpleNamespace(model_key="self", opponent_model_key="opponent")
        context = SimpleNamespace(
            participant=participant,
            your_index=0,
            session=SimpleNamespace(
                observation={
                    "select": {"context": 0},
                    "current": {
                        "players": [
                            {"active": [{"id": 1}]},
                            {"active": [None]},
                        ]
                    },
                }
            ),
        )

        self.assertTrue(_is_setup_context(context))
        self.assertEqual(_evaluation_model_key(context, 0), "self")
        self.assertEqual(_evaluation_model_key(context, 1), "opponent")

    def test_public_cards_include_pokemon_stack_and_resolving_effect(self) -> None:
        state = {
            "players": [
                {
                    "active": [
                        {
                            "id": 10,
                            "serial": 1,
                            "preEvolution": [{"id": 11, "serial": 2}],
                            "energyCards": [{"id": 12, "serial": 3}],
                            "tools": [{"id": 13, "serial": 4}],
                        }
                    ],
                    "bench": [],
                    "discard": [{"id": 14, "serial": 5}],
                },
                {"active": [], "bench": [], "discard": []},
            ],
            "stadium": [{"id": 15, "serial": 6, "playerIndex": 0}],
            "looking": [{"id": 17, "serial": 8, "playerIndex": 0}],
        }
        select = {
            "effect": {"id": 16, "serial": 7, "playerIndex": 0},
            # 同じphysical cardはserialで重複計上しない。
            "contextCard": {"id": 10, "serial": 1, "playerIndex": 0},
        }

        public = public_cards_by_player(state, select)

        self.assertEqual(
            Counter(public[0]),
            Counter([10, 11, 12, 13, 14, 15, 16, 17]),
        )
        self.assertEqual(public[1], [])

    def test_resolving_card_still_in_hand_is_not_counted_twice(self) -> None:
        state = {
            "players": [
                {
                    "active": [],
                    "bench": [],
                    "discard": [],
                    "hand": [{"id": 2, "serial": 46, "playerIndex": 0}],
                    "prize": [],
                },
                {"active": [], "bench": [], "discard": []},
            ]
        }
        select = {
            "contextCard": {"id": 2, "serial": 46, "playerIndex": 0},
            "effect": None,
        }

        self.assertEqual(public_cards_by_player(state, select)[0], [])

    def test_hidden_zone_sampling_removes_transient_hand_duplicate(self) -> None:
        full_deck = tuple(range(1, 61))
        own_hand = [
            {"id": card_id, "serial": 100 + card_id, "playerIndex": 0}
            for card_id in range(7, 12)
        ]
        observation = {
            "current": {
                "yourIndex": 0,
                "players": [
                    {
                        "active": [{"id": 1, "serial": 1}],
                        "bench": [],
                        "deckCount": 43,
                        "discard": [
                            {"id": card_id, "serial": card_id}
                            for card_id in range(2, 7)
                        ],
                        "prize": [None] * 6,
                        "handCount": 5,
                        "hand": own_hand,
                    },
                    {
                        "active": [],
                        "bench": [],
                        "deckCount": 47,
                        "discard": [],
                        "prize": [None] * 6,
                        "handCount": 7,
                        "hand": None,
                    },
                ],
                "stadium": [],
                "looking": [],
            },
            # 同じカードが別serialのcontextCardとして一時的に重複するケース。
            "select": {
                "contextCard": {"id": 7, "serial": 999, "playerIndex": 0},
                "effect": None,
            },
        }

        hidden = sample_hidden_zones(
            observation,
            your_index=0,
            your_full_deck=full_deck,
            seen_opponent_cards=[],
        )

        self.assertEqual(len(hidden.your_deck), 43)
        self.assertEqual(len(hidden.your_prize), 6)

    def test_own_hand_is_not_sampled(self) -> None:
        full_deck = predict_full_deck([])
        own_hand = [
            {"id": card_id, "serial": index + 1, "playerIndex": 0}
            for index, card_id in enumerate(full_deck[:7])
        ]
        observation = {
            "current": {
                "yourIndex": 0,
                "players": [
                    {
                        "active": [],
                        "bench": [],
                        "deckCount": 47,
                        "discard": [],
                        "prize": [None] * 6,
                        "handCount": 7,
                        "hand": own_hand,
                    },
                    {
                        "active": [],
                        "bench": [],
                        "deckCount": 47,
                        "discard": [],
                        "prize": [None] * 6,
                        "handCount": 7,
                        "hand": None,
                    },
                ],
                "stadium": [],
            },
            "select": {"effect": None, "contextCard": None},
        }

        hidden = sample_hidden_zones(
            observation,
            your_index=0,
            your_full_deck=full_deck,
            seen_opponent_cards=[],
        )

        self.assertEqual(observation["current"]["players"][0]["hand"], own_hand)
        self.assertFalse(hasattr(hidden, "your_hand"))
        self.assertEqual(len(hidden.your_deck), 47)
        self.assertEqual(len(hidden.your_prize), 6)
        self.assertEqual(len(hidden.opponent_deck), 47)
        self.assertEqual(len(hidden.opponent_prize), 6)
        self.assertEqual(len(hidden.opponent_hand), 7)

    def test_database_candidate_is_repaired_to_include_observed_cards(self) -> None:
        candidate = list(range(60))
        repaired = _reconcile_candidate(candidate, Counter({9999: 2, 0: 1}))

        self.assertEqual(len(repaired), 60)
        self.assertGreaterEqual(Counter(repaired)[9999], 2)
        self.assertGreaterEqual(Counter(repaired)[0], 1)

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
            agent_src=ROOT / "agents" / "16model_result" / "cluster_00" / "src",
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

    def test_mps_sparse_inputs_preserve_six_arrays(self) -> None:
        encoder = (
            np.arange(12, dtype=np.int32).reshape(2, 6),
            np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(2, 6),
            np.arange(8, dtype=np.int32).reshape(2, 4),
        )
        decoder = (
            np.arange(10, dtype=np.int32).reshape(2, 5),
            np.linspace(1.0, 2.0, 10, dtype=np.float32).reshape(2, 5),
            np.arange(6, dtype=np.int32).reshape(2, 3),
        )

        actual = _mps_sparse_inputs(encoder, decoder, torch.device("cpu"))

        for tensor, expected in zip(
            actual,
            (*encoder, *decoder),
            strict=True,
        ):
            np.testing.assert_array_equal(tensor.numpy(), expected)

    def test_mps_sparse_input_batches_preserve_each_batch(self) -> None:
        sparse_batches = []
        for base in (0, 100):
            encoder = (
                np.arange(base, base + 12, dtype=np.int32).reshape(2, 6),
                np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(2, 6),
                np.arange(base + 20, base + 28, dtype=np.int32).reshape(2, 4),
            )
            decoder = (
                np.arange(base + 30, base + 40, dtype=np.int32).reshape(2, 5),
                np.linspace(1.0, 2.0, 10, dtype=np.float32).reshape(2, 5),
                np.arange(base + 40, base + 46, dtype=np.int32).reshape(2, 3),
            )
            sparse_batches.append((encoder, decoder))

        actual_batches = _mps_sparse_input_batches(
            sparse_batches,
            torch.device("cpu"),
        )

        self.assertEqual(len(actual_batches), len(sparse_batches))
        for actual, (encoder, decoder) in zip(
            actual_batches,
            sparse_batches,
            strict=True,
        ):
            for tensor, expected in zip(
                actual,
                (*encoder, *decoder),
                strict=True,
            ):
                np.testing.assert_array_equal(tensor.numpy(), expected)

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
        runtime = _load_runtime(
            ROOT / "agents" / "16model_result" / "cluster_00" / "src"
        )
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
    def test_worker_batched_hidden_sampling_survives_spawn(self) -> None:
        src = ROOT / "agents" / "rl_mcts_r_robin1" / "src"
        specs = [
            SimpleNamespace(
                name=name,
                agent_path=src / "main.py",
                deck_path=src / "deck.csv",
            )
            for name in ("a", "b")
        ]

        with tempfile.TemporaryDirectory() as directory:
            training_json_dir = Path(directory)
            output = run_worker_batched_tournament(
                specs,
                [("a", "b")],
                2,
                device_name="cpu",
                batch_size=8,
                lanes=2,
                search_count=1,
                max_selections=500,
                seed=456,
                cpu_workers=2,
                training_json_dir=training_json_dir,
            )
            episode_paths = sorted(training_json_dir.glob("episode_*.json"))
            episodes = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in episode_paths
            ]

        self.assertEqual(len(output.results), 2)
        self.assertTrue(all(result.error is None for result in output.results))
        self.assertGreater(output.profile.nn_evaluations, 0)
        self.assertEqual(len(episodes), 2)
        for episode in episodes:
            self.assertEqual(
                set(episode),
                {"format", "rewards", "decks", "decisions"},
            )
            self.assertEqual([len(deck) for deck in episode["decks"]], [60, 60])
            self.assertGreater(len(episode["decisions"]), 0)
            for decision in episode["decisions"]:
                self.assertEqual(
                    set(decision),
                    {"player", "observation", "chosenIndex"},
                )
                self.assertNotIn("logs", decision["observation"])
                player = decision["player"]
                self.assertIsNotNone(
                    decision["observation"]["players"][player]["hand"]
                )

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
