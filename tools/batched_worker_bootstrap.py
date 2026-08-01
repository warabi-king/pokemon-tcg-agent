"""Torchを読み込まずにworker-batchedのCPU workerを起動する。"""

from __future__ import annotations

import os
from typing import Any


def worker_main(
    worker_id: int,
    spec_records: list[tuple[str, str, str]],
    game_request_records: list[tuple[str, str, int, bool]],
    lanes: int,
    search_count: int,
    max_selections: int,
    max_turns: int | None,
    seed: int,
    remote_batch_size: int,
    training_json_dir: str | None,
    request_queue: Any,
    response_queue: Any,
    result_queue: Any,
) -> None:
    """プリミティブな引数を復元して、軽量worker本体へ渡す。"""
    os.environ["PTCG_BATCHED_LIGHTWEIGHT_WORKER"] = "1"

    from batched_tournament import BatchedGameRequest, _parallel_worker_main

    game_requests = [
        BatchedGameRequest(
            name0=name0,
            name1=name1,
            game_index=game_index,
            swap=swap,
        )
        for name0, name1, game_index, swap in game_request_records
    ]
    _parallel_worker_main(
        worker_id,
        spec_records,
        game_requests,
        lanes,
        search_count,
        max_selections,
        max_turns,
        seed,
        remote_batch_size,
        training_json_dir,
        request_queue,
        response_queue,
        result_queue,
    )
