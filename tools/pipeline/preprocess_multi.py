"""単一パス・多クラスタ前処理（Phase0 の前処理を 29回スキャン → 1回に）。

現状 phase0.py は (agent × role) ごとに preprocess() を呼び、そのたびに全公式リプレイを
json.loads ＋特徴抽出し直していた。しかし各プレイヤー p のサンプル特徴量
（get_encoder_input(obs, deck_p) ＋ value=reward_p）は、どのクラスタ/role へ割り当てるかと
無関係に一意に決まる。割当を決めるのは deck_p / deck_opp の最近傍クラスタだけ。

したがって全エピソードを 1 回だけ走査し、各プレイヤー p のサンプルを
  own シャード[ cluster(deck_p) ]         … p が自分側
  opp シャード[ cluster(deck_opp) ]        … p が「そのクラスタと対戦した相手」
の両方へ同じタプルを振り分ければ、16 own + 16 opp シャードを同時に生成できる。

出力レイアウト・manifest は preprocess.py と互換（<out_root>/<name>_own, <name>_opp）。
"""

from __future__ import annotations

import functools
import gc
import json
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import config  # noqa: F401  (sys.path 設定の副作用)
from deck_utils import nearest_deck
from episode_io import iter_multi_source

# imitation_data と同一の特徴抽出コードを流用（重複実装で挙動がズレるのを避ける）
from imitation_data import (  # noqa: E402
    enumerate_actions,
    get_decoder_input,
    get_encoder_input,
    to_observation_class,
)

# (deck_p, deck_opp, value, samples) の並び
PlayerBlock = tuple[list[int], list[int], float, list[tuple]]


def extract_player_blocks(data: bytes, value_decay: float = 1.0) -> list[PlayerBlock]:
    """1エピソードから、各プレイヤーの (自デッキ, 相手デッキ, value, サンプル列) を返す。

    extract_samples_from_episode と同じ特徴量を作るが、deck_filter で捨てず、
    後段のクラスタ振り分けのために自/相手デッキを保持したまま返す。

    value_decay(既定1.0=無効)を1未満にすると、imitation_data.extract_samples_from_episode
    と同様に、終局から遠い(序盤の)局面ほどvalue教師を0へ指数減衰させる
    (全局面へ一律で最終結果を貼るとvalueヘッドが序盤局面でも飽和しやすいため)。
    戻り値のPlayerBlock.valueは互換性のため引き続き最終結果(減衰前)を返す。
    """
    try:
        j = json.loads(data)
    except Exception:
        return []
    rewards = j.get("rewards")
    if not rewards or len(rewards) != 2 or any(r is None for r in rewards):
        return []
    steps = j.get("steps")
    if not steps or len(steps) < 3:
        return []
    decks = [steps[1][0]["action"], steps[1][1]["action"]]
    if len(decks[0]) != 60 or len(decks[1]) != 60:
        return []

    blocks: list[PlayerBlock] = []
    for player in range(2):
        your_deck = decks[player]
        opponent_deck = decks[1 - player]
        final_value = float(rewards[player])
        raw_samples: list[tuple] = []
        for i in range(1, len(steps) - 1):
            sel = steps[i][player]["observation"].get("select")
            if sel is None:
                continue
            actual_action = steps[i + 1][player]["action"]
            if actual_action is None:
                continue
            obs = to_observation_class(steps[i][player]["observation"])
            actions = enumerate_actions(len(obs.select.option), obs.select.maxCount)
            if not actions:
                continue
            target = tuple(sorted(actual_action))
            chosen_index = next(
                (idx for idx, candidate in enumerate(actions) if tuple(candidate) == target),
                None,
            )
            if chosen_index is None:
                continue
            sv_enc = get_encoder_input(obs, your_deck)
            sv_dec = get_decoder_input(obs, actions)
            raw_samples.append(
                (
                    np.asarray(sv_enc.index, dtype=np.int32),
                    np.asarray(sv_enc.value, dtype=np.float32),
                    np.asarray(sv_enc.offset, dtype=np.int32),
                    np.asarray(sv_dec.index, dtype=np.int32),
                    np.asarray(sv_dec.value, dtype=np.float32),
                    np.asarray(sv_dec.offset, dtype=np.int32),
                    chosen_index,
                )
            )

        # 終局に一番近い局面(末尾)にfinal_valueをそのまま付与し、そこから遡るほど
        # value_decay倍ずつ0へ近づける。value_decay=1.0なら全局面が従来通りfinal_value。
        # Python list のままだと int 1個あたり数十バイト消費するため、
        # numpy int32/float32 へ圧縮してから返す（IPC・メインのバッファ・
        # シャードファイルすべてが小さくなり、学習側の読み込みも速くなる）。
        n = len(raw_samples)
        samples: list[tuple] = []
        for offset, sample in enumerate(raw_samples):
            distance_from_end = n - 1 - offset
            decayed_value = final_value * (value_decay**distance_from_end)
            samples.append((*sample, np.float32(decayed_value)))
        if samples:
            blocks.append((your_deck, opponent_deck, final_value, samples))
    return blocks


def _worker(data: bytes, value_decay: float = 1.0) -> list[PlayerBlock]:
    return extract_player_blocks(data, value_decay=value_decay)


class _ShardWriter:
    """(role, cluster) 単位の pkl シャード書き出し器。"""

    def __init__(self, out_dir: Path, shard_size: int, role: str, label: str):
        self.out_dir = out_dir
        self.shard_size = shard_size
        self.role = role
        self.label = label
        self.buffer: list = []
        self.shard_index = 0
        self.total = 0
        self.episodes = 0  # 参照用（振り分けなので厳密なepisode数ではない）
        out_dir.mkdir(parents=True, exist_ok=True)

    def extend(self, samples: list) -> None:
        self.buffer.extend(samples)
        self.total += len(samples)
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        with open(self.out_dir / f"shard_{self.shard_index:05d}.pkl", "wb") as f:
            pickle.dump(self.buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.shard_index += 1
        self.buffer = []

    def close(self, episodes: int) -> None:
        self.flush()
        (self.out_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "label": self.label, "role": self.role,
                    "episodes": episodes, "samples": self.total,
                    "shards": self.shard_index, "shard_size": self.shard_size,
                },
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )


def preprocess_all(
    episodes: list[Path],
    out_root: Path,
    reps: list[dict],
    threshold: float,
    shard_size: int,
    workers: int = 1,
    roles: tuple[str, ...] = ("own", "opp"),
    verbose: bool = True,
    value_decay: float = 1.0,
) -> dict[str, dict[str, int]]:
    """全エピソードを1回走査し、reps 各クラスタの own/opp シャードを同時生成する。

    value_decayはextract_player_blocks参照(既定1.0=無効)。

    戻り値: {cluster_name: {"own": shards, "opp": shards}}
    """
    out_root.mkdir(parents=True, exist_ok=True)
    writers: dict[tuple[str, str], _ShardWriter] = {}
    for rep in reps:
        name = rep["name"]
        for role in roles:
            suffix = "own" if role == "own" else "opp"
            writers[(name, role)] = _ShardWriter(
                out_root / f"{name}_{suffix}", shard_size, role, f"{name}/{suffix}"
            )

    # デッキ → クラスタ名 のメモ化（同一デッキが多数回出るため）
    memo: dict[tuple, str | None] = {}

    def cluster_of(deck: list[int]) -> str | None:
        key = tuple(sorted(deck))
        if key in memo:
            return memo[key]
        rep, sim = nearest_deck(deck, reps)
        name = rep["name"] if (rep is not None and sim >= threshold) else None
        memo[key] = name
        return name

    t0 = time.time()
    episode_count = 0

    def route(blocks: list[PlayerBlock]) -> None:
        for deck_p, deck_opp, _value, samples in blocks:
            if "own" in roles:
                c = cluster_of(deck_p)
                if c is not None:
                    writers[(c, "own")].extend(samples)
            if "opp" in roles:
                c = cluster_of(deck_opp)
                if c is not None:
                    writers[(c, "opp")].extend(samples)

    stream = (
        data for _source, name, data in iter_multi_source(episodes) if name != "manifest.csv"
    )

    def _rss_mb() -> str:
        """メイン＋ワーカー全プロセスの合計RSSを返す（真のメモリ使用量を可視化）。"""
        try:
            import psutil
            p = psutil.Process()
            rss = p.memory_info().rss
            for child in p.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except Exception:
                    pass
            return f" rss_total={rss / (1024 * 1024):.0f}MB"
        except Exception:
            return ""

    min_avail_mb = config.MIN_AVAIL_MB

    def _memory_guard() -> None:
        """空きメモリが閾値を割ったら全バッファを即フラッシュしてOOMを回避する。"""
        try:
            import psutil
            avail_mb = psutil.virtual_memory().available / (1024 * 1024)
        except Exception:
            return
        if avail_mb < min_avail_mb:
            for w in writers.values():
                w.flush()
            gc.collect()
            print(f"  [preprocess_all] 空きメモリ{avail_mb:.0f}MB < {min_avail_mb}MB: "
                  f"全バッファをフラッシュしました", flush=True)

    if workers and workers > 1:
        # fork 前に親プロセス側で語彙サイズ等の lru_cache を温めておく。
        # こうすると全ワーカーが populated なキャッシュを Copy-on-Write で共有し、
        # ワーカー起動/世代交代のたびに all_card_data()（数百MBの一時確保）が
        # 同時多発してOOMを誘発するのを防げる。
        try:
            from rl_mcts.model import attack_count, card_count
            card_count()
            attack_count()
        except Exception:
            pass
        # 既存オブジェクトを恒久世代へ移し、GC由来のCoWページ汚染を抑える。
        gc.freeze()
        # maxtasksperchild でワーカーを定期的に再起動し、長時間実行時の
        # プロセス単位メモリ増加（CoWドリフト等）を上限内に抑える。
        worker = functools.partial(_worker, value_decay=value_decay)
        with Pool(workers, maxtasksperchild=2000) as pool:
            for blocks in pool.imap_unordered(worker, stream, chunksize=8):
                route(blocks)
                episode_count += 1
                if episode_count % 2000 == 0:
                    _memory_guard()
                    if verbose:
                        print(f"  [preprocess_all] episodes={episode_count} "
                              f"elapsed={time.time()-t0:.1f}s workers={workers}{_rss_mb()}", flush=True)
    else:
        for data in stream:
            route(extract_player_blocks(data, value_decay=value_decay))
            episode_count += 1
            if episode_count % 2000 == 0:
                _memory_guard()
                if verbose:
                    print(f"  [preprocess_all] episodes={episode_count} "
                          f"elapsed={time.time()-t0:.1f}s{_rss_mb()}", flush=True)

    result: dict[str, dict[str, int]] = {}
    for (name, role), w in writers.items():
        w.close(episode_count)
        result.setdefault(name, {})[role] = w.shard_index
    if verbose:
        print(f"  [preprocess_all] done: episodes={episode_count} "
              f"elapsed={time.time()-t0:.1f}s clusters={len(reps)} roles={roles}")
    return result
