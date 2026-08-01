"""学習並列化ベンチマークの軽量テスト。"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest

from .benchmark_parallel_training import (
    _absolute_path_without_resolving_symlinks,
    TrainingJob,
    assign_training_jobs,
    benchmark_parallel_training,
    discover_training_jobs,
    schedule_training_jobs,
)


class DiscoverTrainingJobsTest(unittest.TestCase):
    def test_discovers_self_and_opponent_jobs_in_stable_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shards_root = root / "shards"
            agents_root = root / "agents"
            for agent in ("cluster_01", "cluster_00"):
                for suffix in ("own", "opponent"):
                    shard_dir = shards_root / f"{agent}_{suffix}"
                    shard_dir.mkdir(parents=True)
                    (shard_dir / "manifest.json").touch()
                    (shard_dir / "shard_00000.pkl").touch()
                src = agents_root / agent / "src"
                src.mkdir(parents=True)
                (src / "model.pth").touch()
            (agents_root / "cluster_00" / "src" / "opponent_model.pth").touch()

            jobs = discover_training_jobs(shards_root, agents_root, seed=10)

        self.assertEqual(
            [job.name for job in jobs],
            [
                "cluster_00_self",
                "cluster_00_opponent",
                "cluster_01_self",
                "cluster_01_opponent",
            ],
        )
        self.assertEqual([job.seed for job in jobs], [10, 11, 12, 13])
        self.assertEqual(jobs[1].initial_model.name, "opponent_model.pth")
        self.assertEqual(jobs[3].initial_model.name, "model.pth")


class BenchmarkParallelTrainingTest(unittest.TestCase):
    def test_largest_shards_are_scheduled_first(self) -> None:
        scheduled = schedule_training_jobs(
            [
                TrainingJob("small", Path("s"), Path("m"), 0, 10),
                TrainingJob("large_b", Path("s"), Path("m"), 0, 30),
                TrainingJob("large_a", Path("s"), Path("m"), 0, 30),
            ],
            "largest-first",
        )

        self.assertEqual(
            [job.name for job in scheduled],
            ["large_a", "large_b", "small"],
        )

    def test_jobs_are_balanced_across_cpu_workers(self) -> None:
        jobs = [
            TrainingJob(
                f"job_{index}",
                Path("s"),
                Path("m"),
                index,
                samples,
            )
            for index, samples in enumerate((9, 8, 7, 6, 5, 4))
        ]

        assignments = assign_training_jobs(jobs, workers=3)

        loads = [sum(job.samples for job in worker) for worker in assignments]
        self.assertEqual(loads, [13, 13, 13])

    def test_python_symlink_is_not_resolved_out_of_virtualenv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            python_link = Path(directory) / "python"
            try:
                python_link.symlink_to(sys.executable)
            except OSError as error:
                self.skipTest(f"symlinkを作成できません: {error}")

            absolute = _absolute_path_without_resolving_symlinks(python_link)

        self.assertEqual(absolute, python_link)

    def test_persistent_worker_runs_multiple_training_jobs(self) -> None:
        fake_training_source = """\
import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--output-model', type=Path, required=True)
parser.add_argument('--metrics-file', type=Path, required=True)
args, _unknown = parser.parse_known_args()
args.output_model.write_bytes(b'model')
args.metrics_file.write_text('epoch\\n', encoding='utf-8')
print(f'worker={os.getpid()}')
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shards_root = root / "shards"
            agents_root = root / "agents"
            for agent in ("cluster_00", "cluster_01"):
                for suffix in ("own", "opponent"):
                    shard_dir = shards_root / f"{agent}_{suffix}"
                    shard_dir.mkdir(parents=True)
                    (shard_dir / "manifest.json").touch()
                    (shard_dir / "shard_00000.pkl").touch()
                src = agents_root / agent / "src"
                src.mkdir(parents=True)
                (src / "model.pth").write_bytes(b"initial")
            train_script = root / "fake_train.py"
            train_script.write_text(fake_training_source, encoding="utf-8")

            summary = benchmark_parallel_training(
                python=Path(sys.executable),
                train_script=train_script,
                shards_root=shards_root,
                agents_root=agents_root,
                output_dir=root / "output",
                workers=2,
                job_limit=0,
                epochs=1,
                batch_size=8,
                learning_rate=3e-4,
                device="cpu",
                seed=0,
                persistent_workers=True,
                threads_per_worker=1,
            )

        self.assertEqual(summary["executionMode"], "persistent")
        self.assertEqual(summary["threadsPerWorker"], 1)
        self.assertEqual(summary["jobs"], 4)
        worker_pids = {
            result["workerPid"] for result in summary["jobResults"]
        }
        self.assertLessEqual(len(worker_pids), 2)
        self.assertNotIn(os.getpid(), worker_pids)


if __name__ == "__main__":
    unittest.main()
