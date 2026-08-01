"""並列自己対戦・学習ループの割当テスト。"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from parallel_selfplay_training import (  # noqa: E402
    _balanced_games_per_pairing,
    _completed_pairing_games,
    _safe_training_name,
    read_parallel_training_metrics,
    run_parallel_games,
    train_models,
)


class ParallelSelfplayTrainingTest(unittest.TestCase):
    @patch("parallel_selfplay_training.subprocess.run")
    @patch("parallel_selfplay_training._completed_pairing_games")
    def test_game_generation_uses_benchmarked_worker_batched_backend(
        self,
        completed_pairing_games,
        subprocess_run,
    ) -> None:
        completed_pairing_games.side_effect = (
            ([0, 0, 0], [0, 0, 0]),
            ([2, 2, 2], [2, 2, 2]),
        )

        count = run_parallel_games(
            python=Path("/python"),
            runner=Path("/repo/tools/run_matches_round_robin.py"),
            match_root=Path("/repo"),
            agents={"a": "agents/a/src/main.py", "b": "agents/b/src/main.py"},
            episodes_dir=Path("/episodes"),
            workers=2,
            lanes_per_worker=3,
            include_self=True,
            match_batch_size=256,
            search_count=10,
            device="mps",
            seed=7,
        )

        self.assertEqual(count, 6)
        command = subprocess_run.call_args.args[0]
        self.assertEqual(command[command.index("--backend") + 1], "worker-batched")
        self.assertEqual(command[command.index("--workers") + 1], "2")
        self.assertEqual(command[command.index("--lanes") + 1], "6")
        self.assertEqual(command[command.index("--batch-size") + 1], "256")
        self.assertEqual(command[command.index("--search-count") + 1], "10")
        self.assertEqual(
            json.loads(command[command.index("--games-per-pairing-json") + 1]),
            [2, 2, 2],
        )

    def test_3600_games_are_balanced_across_16_agents(self) -> None:
        names = [f"cluster_{index:02d}" for index in range(16)]
        pairings = list(itertools.combinations_with_replacement(names, 2))

        counts = _balanced_games_per_pairing(pairings, 3600)

        self.assertEqual(len(pairings), 136)
        self.assertEqual(sum(counts), 3600)
        self.assertEqual(min(counts), 26)
        self.assertEqual(max(counts), 27)

        standings = {name: 0 for name in names}
        player_sides = {name: 0 for name in names}
        for (name0, name1), count in zip(pairings, counts, strict=True):
            standings[name0] += count
            player_sides[name0] += count
            if name0 == name1:
                player_sides[name0] += count
            else:
                standings[name1] += count
                player_sides[name1] += count

        self.assertEqual(set(standings.values()), {424})
        self.assertEqual(set(player_sides.values()), {450})

    def test_completed_games_are_counted_per_pairing(self) -> None:
        pairings = [("cluster_00", "cluster_00"), ("cluster_00", "cluster_01")]
        with tempfile.TemporaryDirectory() as directory:
            episodes_dir = Path(directory)
            for name0, name1, indices in (
                ("cluster_00", "cluster_00", (1, 3)),
                ("cluster_00", "cluster_01", (2,)),
            ):
                matchup = hashlib.sha1(
                    f"{name0}\0{name1}".encode("utf-8")
                ).hexdigest()[:8]
                prefix = (
                    f"episode_{_safe_training_name(name0)}_vs_"
                    f"{_safe_training_name(name1)}_{matchup}_"
                )
                for game_index in indices:
                    (episodes_dir / f"{prefix}g{game_index:06d}_s0.json").touch()

            completed, max_indices = _completed_pairing_games(
                episodes_dir,
                pairings,
            )

        self.assertEqual(completed, [2, 1])
        self.assertEqual(max_indices, [3, 2])

    def test_reads_job_metrics_for_live_loss_display(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            metrics = output / "jobs" / "cluster_00_self" / "metrics.csv"
            metrics.parent.mkdir(parents=True)
            metrics.write_text(
                "epoch,batches,loss,loss_value,loss_policy,"
                "train_accuracy,val_accuracy,elapsed_seconds\n"
                "0,4,1.2,0.2,1.0,0.75,0.0,2.5\n",
                encoding="utf-8",
            )

            rows = read_parallel_training_metrics(output)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["job"], "cluster_00_self")
        self.assertEqual(rows[0]["epoch"], 0)
        self.assertEqual(rows[0]["loss"], 1.2)

    def test_parallel_training_publishes_only_completed_model_set(self) -> None:
        fake_preprocessor = """\
import argparse
import json

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--episodes', nargs='+')
parser.add_argument('--agent', action='append', default=[])
parser.add_argument('--output-dir', type=__import__('pathlib').Path)
parser.add_argument('--shard-size')
args = parser.parse_args()
for raw in args.agent:
    name = raw.split('=', 1)[0]
    for role in ('own', 'opponent'):
        target = args.output_dir / f'{name}_{role}'
        target.mkdir(parents=True)
        (target / 'shard_00000.pkl').write_bytes(b'shard')
        (target / 'manifest.json').write_text(
            json.dumps({'samples': 8, 'episodes': 1}), encoding='utf-8'
        )
"""
        fake_parallel_trainer = """\
import argparse
import csv
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--agents-root', type=Path)
parser.add_argument('--output-dir', type=Path)
parser.add_argument('--workers', type=int)
parser.add_argument('--device-wave-size', type=int)
args, unknown = parser.parse_known_args()
args.output_dir.mkdir(parents=True)
jobs = []
for agent_dir in sorted(args.agents_root.iterdir()):
    if not (agent_dir / 'src').is_dir():
        continue
    for role in ('self', 'opponent'):
        name = f'{agent_dir.name}_{role}'
        job = args.output_dir / 'jobs' / name
        job.mkdir(parents=True)
        (job / 'model.pth').write_bytes(name.encode())
        with (job / 'metrics.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(('epoch', 'batches', 'loss', 'loss_value',
                             'loss_policy', 'train_accuracy', 'val_accuracy',
                             'elapsed_seconds'))
            writer.writerow((0, 1, 1.0, 0.1, 0.9, 0.5, 0.0, 0.1))
        jobs.append(name)
(args.output_dir / 'summary.json').write_text(json.dumps({
    'jobs': len(jobs),
    'device': 'cpu',
    'executionMode': 'central-device',
    'workers': args.workers,
    'deviceWaveSize': args.device_wave_size,
    'argv': sys.argv[1:],
}), encoding='utf-8')
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            match_root = root / "match"
            train_root = root / "train"
            agents_root = match_root / "agents" / "models"
            episodes = root / "episodes"
            episodes.mkdir()
            train_root.mkdir()
            agents: dict[str, Path] = {}
            for name in ("cluster_00", "cluster_01"):
                source = agents_root / name / "src"
                source.mkdir(parents=True)
                (source / "main.py").touch()
                (source / "deck.csv").write_text("1\n", encoding="utf-8")
                (source / "model.pth").write_bytes(b"initial")
                agents[name] = source / "main.py"

            preprocessor = train_root / "fake_preprocessor.py"
            preprocessor.write_text(fake_preprocessor, encoding="utf-8")
            train_script = train_root / "fake_train.py"
            train_script.touch()
            parallel_trainer = match_root / "fake_parallel.py"
            parallel_trainer.write_text(
                fake_parallel_trainer,
                encoding="utf-8",
            )
            checkpoint = agents_root / "model_episode1"
            callbacks: list[list[dict[str, object]]] = []

            summary = train_models(
                python=Path(sys.executable),
                preprocessor=preprocessor,
                train_script=train_script,
                parallel_trainer=parallel_trainer,
                train_root=train_root,
                match_root=match_root,
                agents=agents,
                episodes_dir=episodes,
                work_dir=root / "learning",
                checkpoint_dir=checkpoint,
                epochs=1,
                train_minibatch_size=8,
                learning_rate=3e-4,
                shard_size=100,
                device="cpu",
                seed=0,
                training_workers=2,
                device_wave_size=2,
                progress_callback=lambda rows: callbacks.append(rows),
                poll_interval_seconds=0.01,
            )

            parallel_summary = json.loads(
                (
                    root
                    / "learning"
                    / "parallel_training"
                    / "summary.json"
                ).read_text(encoding="utf-8")
            )
            for name, main_path in agents.items():
                source = main_path.parent
                self_bytes = f"{name}_self".encode()
                opponent_bytes = f"{name}_opponent".encode()
                self.assertEqual((source / "model.pth").read_bytes(), self_bytes)
                self.assertEqual(
                    (source / "opponent_model.pth").read_bytes(),
                    opponent_bytes,
                )
                self.assertEqual(
                    (checkpoint / name / "model.pth").read_bytes(),
                    self_bytes,
                )

        self.assertEqual(summary["executionMode"], "central-device")
        self.assertEqual(parallel_summary["workers"], 2)
        self.assertEqual(parallel_summary["deviceWaveSize"], 2)
        self.assertIn("--central-device-owner", parallel_summary["argv"])
        self.assertIn("--keep-models", parallel_summary["argv"])
        self.assertEqual(len(callbacks[-1]), 4)


if __name__ == "__main__":
    unittest.main()
