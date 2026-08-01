"""従来JSONと高速形式から学習した重みを直接比較する。"""

from __future__ import annotations

import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from batched_tournament import run_worker_batched_tournament  # noqa: E402


class PreencodedTrainingEquivalenceTest(unittest.TestCase):
    def test_one_epoch_produces_identical_self_and_opponent_weights(self) -> None:
        agent_root = ROOT / "agents" / "16model_pretrained"
        specs = [
            SimpleNamespace(
                name=name,
                agent_path=agent_root / name / "src" / "main.py",
                deck_path=agent_root / name / "src" / "deck.csv",
            )
            for name in ("cluster_00", "cluster_01")
        ]
        train_root = ROOT.parent / "pokemon-tcg-agent-nomura"
        preprocessor = train_root / "tools" / "train" / "preprocess_match_agents.py"
        trainer = train_root / "tools" / "train" / "train_imitation.py"

        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            episodes = temporary / "episodes"
            output = run_worker_batched_tournament(
                specs,
                [("cluster_00", "cluster_01")],
                1,
                device_name="cpu",
                batch_size=4,
                lanes=1,
                search_count=1,
                max_turns=5,
                max_selections=30,
                seed=321,
                cpu_workers=1,
                training_json_dir=episodes,
                training_format="comparison",
            )
            self.assertEqual(len(output.results), 1)
            self.assertIsNone(output.results[0].error)

            episode_by_format = {
                "json": next(episodes.glob("episode_*.json")),
                "preencoded": next(episodes.glob("episode_*.pkl")),
            }
            shards: dict[str, Path] = {}
            for training_format, episode_path in episode_by_format.items():
                shards[training_format] = temporary / f"{training_format}_shards"
                subprocess.run(
                    [
                        sys.executable,
                        str(preprocessor),
                        "--episodes",
                        str(episode_path),
                        "--output-dir",
                        str(shards[training_format]),
                        "--shard-size",
                        "1000",
                        "--agent",
                        f"cluster_00={specs[0].deck_path}",
                        "--agent",
                        f"cluster_01={specs[1].deck_path}",
                    ],
                    cwd=ROOT,
                    check=True,
                    stdout=subprocess.DEVNULL,
                )

            roles = {
                "own": agent_root / "cluster_00" / "src" / "model.pth",
                "opponent": (
                    agent_root
                    / "cluster_00"
                    / "src"
                    / "opponent_model.pth"
                ),
            }
            training_env = os.environ.copy()
            training_env["OMP_NUM_THREADS"] = "1"
            training_env["MKL_NUM_THREADS"] = "1"
            for role, initial_model in roles.items():
                trained_states = {}
                dataset = f"cluster_00_{role}"
                json_shards = sorted(
                    (shards["json"] / dataset).glob("shard_*.pkl")
                )
                preencoded_shards = sorted(
                    (shards["preencoded"] / dataset).glob("shard_*.pkl")
                )
                self.assertEqual(len(json_shards), len(preencoded_shards))
                for json_shard, preencoded_shard in zip(
                    json_shards,
                    preencoded_shards,
                    strict=True,
                ):
                    json_samples = pickle.loads(json_shard.read_bytes())
                    preencoded_samples = pickle.loads(
                        preencoded_shard.read_bytes()
                    )
                    self.assertEqual(len(json_samples), len(preencoded_samples))
                    for json_sample, preencoded_sample in zip(
                        json_samples,
                        preencoded_samples,
                        strict=True,
                    ):
                        for field_index in (0, 2, 3, 5):
                            self.assertEqual(
                                json_sample[field_index],
                                preencoded_sample[field_index],
                            )
                        for field_index in (1, 4):
                            np.testing.assert_array_equal(
                                np.asarray(
                                    json_sample[field_index],
                                    dtype=np.float32,
                                ),
                                np.asarray(
                                    preencoded_sample[field_index],
                                    dtype=np.float32,
                                ),
                            )
                        self.assertEqual(json_sample[6:], preencoded_sample[6:])
                for training_format in episode_by_format:
                    output_model = temporary / f"{training_format}_{role}.pth"
                    subprocess.run(
                        [
                            sys.executable,
                            str(trainer),
                            "--shards",
                            str(shards[training_format] / dataset),
                            "--epochs",
                            "1",
                            "--batch-size",
                            "4",
                            "--lr",
                            "0.0003",
                            "--initial-model",
                            str(initial_model),
                            "--output-model",
                            str(output_model),
                            "--metrics-file",
                            str(temporary / f"{training_format}_{role}.csv"),
                            "--seed",
                            "999",
                            "--device",
                            "cpu",
                            "--val-shards",
                            "0",
                        ],
                        cwd=ROOT,
                        check=True,
                        stdout=subprocess.DEVNULL,
                        env=training_env,
                    )
                    trained_states[training_format] = torch.load(
                        output_model,
                        map_location="cpu",
                        weights_only=True,
                    )

                self.assertEqual(
                    trained_states["json"].keys(),
                    trained_states["preencoded"].keys(),
                )
                for name in trained_states["json"]:
                    self.assertTrue(
                        torch.equal(
                            trained_states["json"][name],
                            trained_states["preencoded"][name],
                        ),
                        f"{role} weight mismatch: {name}",
                    )


if __name__ == "__main__":
    unittest.main()
