from __future__ import annotations

import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np

from preprocess_match_agents import (
    PREENCODED_TRAINING_EPISODE_FORMAT,
    preprocess_match_agents,
)


def sample(value: float) -> tuple:
    return ([1], [0.5], [0], [2], [1.0], [0], 0, value)


class PreencodedMatchAgentPreprocessTest(unittest.TestCase):
    def test_routes_preencoded_samples_without_reencoding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes = root / "episodes"
            output = root / "shards"
            episodes.mkdir()
            deck_a = [1] * 60
            deck_b = [2] * 60
            deck_a_path = root / "deck_a.csv"
            deck_b_path = root / "deck_b.csv"
            deck_a_path.write_text("\n".join(map(str, deck_a)), encoding="utf-8")
            deck_b_path.write_text("\n".join(map(str, deck_b)), encoding="utf-8")
            episode = {
                "format": PREENCODED_TRAINING_EPISODE_FORMAT,
                "decks": [deck_a, deck_b],
                "playerSamples": [[sample(1.0)], [sample(-1.0)]],
            }
            with (episodes / "episode_test.pkl").open("wb") as file:
                pickle.dump(episode, file, protocol=pickle.HIGHEST_PROTOCOL)

            summary = preprocess_match_agents(
                [episodes],
                [("a", deck_a_path), ("b", deck_b_path)],
                output,
                shard_size=10,
            )

            def read_shard(name: str) -> list[tuple]:
                with (output / name / "shard_00000.pkl").open("rb") as file:
                    return pickle.load(file)

            self.assertEqual(read_shard("a_own"), [sample(1.0)])
            self.assertEqual(read_shard("a_opponent"), [sample(-1.0)])
            self.assertEqual(read_shard("b_own"), [sample(-1.0)])
            self.assertEqual(read_shard("b_opponent"), [sample(1.0)])
            self.assertEqual(summary["a_own"]["samples"], 1)
            self.assertEqual(summary["b_opponent"]["samples"], 1)

    def test_routes_packed_preencoded_samples(self) -> None:
        def ragged(values, dtype):
            return {
                "values": np.asarray(values, dtype=dtype),
                "boundaries": np.asarray([0, len(values)], dtype=np.int64),
            }

        def packed_player(value: float) -> dict:
            return {
                "count": 1,
                "chosenIndex": np.asarray([0], dtype=np.int64),
                "value": np.asarray([value], dtype=np.float32),
                "encoderIndex": ragged([1], np.int32),
                "encoderValue": ragged([0.5], np.float32),
                "encoderOffset": ragged([0], np.int32),
                "decoderIndex": ragged([2], np.int32),
                "decoderValue": ragged([1.0], np.float32),
                "decoderOffset": ragged([0], np.int32),
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes = root / "episodes"
            output = root / "shards"
            episodes.mkdir()
            deck_a = [1] * 60
            deck_b = [2] * 60
            deck_a_path = root / "deck_a.csv"
            deck_b_path = root / "deck_b.csv"
            deck_a_path.write_text("\n".join(map(str, deck_a)), encoding="utf-8")
            deck_b_path.write_text("\n".join(map(str, deck_b)), encoding="utf-8")
            episode = {
                "format": PREENCODED_TRAINING_EPISODE_FORMAT,
                "decks": [deck_a, deck_b],
                "packedPlayerSamples": [packed_player(1.0), packed_player(-1.0)],
            }
            with (episodes / "episode_test.pkl").open("wb") as file:
                pickle.dump(episode, file, protocol=pickle.HIGHEST_PROTOCOL)

            summary = preprocess_match_agents(
                [episodes],
                [("a", deck_a_path), ("b", deck_b_path)],
                output,
                shard_size=10,
            )

            with (output / "a_own" / "shard_00000.pkl").open("rb") as file:
                self.assertEqual(pickle.load(file), [sample(1.0)])
            with (output / "a_opponent" / "shard_00000.pkl").open("rb") as file:
                self.assertEqual(pickle.load(file), [sample(-1.0)])
            self.assertEqual(summary["a_own"]["samples"], 1)


if __name__ == "__main__":
    unittest.main()
