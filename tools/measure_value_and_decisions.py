"""あるagentツリーについて「valueヘッドの校正」と「着手の決まり方」を実測する。

2026-08-06計測で分かったこと(本番設定 = visit tie-break q / prior exponent 4 /
search_count 10、実測33,331決定・57,200局面):

  - visit最大が同数になる決定が83.9%、そのうちQ値まで完全同値でpriorに落ちる決定が
    全体の28.9%。visitの一意最大で決まるのは16.1%しかない。
  - valueヘッドは |v|>0.99 が48.9%。符号の的中率は67.7%あるが、予測+0.999の局面の
    実際の平均結果は+0.583しかない(2〜5倍の過信)。outcomeに対するMSEは1.043で、
    「常に0」を出す定数予測(≈0.96)より悪い。

原因は value の損失が ``HuberLoss(delta=0.2)`` であること。誤差0.2超で勾配が一定
(L1領域)になるため条件付き「中央値」に寄り、教師が±1しかない以上ネットは±1へ
振り切る。tanhの飽和で兄弟ノードのQ値が完全同値になり、採用済みのq tie-breakが
効かなくなる。

このスクリプトは学習条件を変えたあと、その歪みが実際に直ったかを同じ物差しで
確認するためのもの。使い方:

    .venv/bin/python tools/measure_value_and_decisions.py \\
        --weights agents/_convergence_study --games 4 --label baseline
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

MATCH_ROOT = Path(__file__).resolve().parents[1]
RUNNER = MATCH_ROOT / 'tools' / 'run_matches_round_robin.py'
VENV_PYTHON = MATCH_ROOT / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
PYTHON = VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--clusters', type=int, default=16)
    parser.add_argument('--games', type=int, default=4,
                        help='ペアあたりの試合数(16クラスタなら136ペア)')
    parser.add_argument('--label', default='measure')
    parser.add_argument('--seed', type=int, default=555)
    parser.add_argument('--search-count', type=int, default=10)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument('--device', default='mps' if sys.platform == 'darwin' else 'cuda')
    parser.add_argument('--keep', action='store_true', help='中間データを消さない')
    return parser.parse_args()


def selfplay(args: argparse.Namespace, weights: Path, workspace: Path) -> None:
    episodes = workspace / 'episodes'
    dump = workspace / 'decisions'
    for path in (episodes, dump):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)

    command = [str(PYTHON), str(RUNNER)]
    for index in range(args.clusters):
        cluster = f'cluster_{index:02d}'
        command.extend(['--agent', f'{cluster}={weights / cluster / "src" / "main.py"}'])
    command.extend([
        '--backend', 'worker-batched',
        '--device', args.device,
        '--workers', str(args.workers),
        '--lanes', '800',
        '--batch-size', '256',
        '--search-count', str(args.search_count),
        '--max-turns', '100',
        '--max-selections', '500',
        '--quiet',
        '--training-json-dir', str(episodes),
        '--training-format', 'preencoded',
        '--games', str(args.games),
        '--seed', str(args.seed),
    ])
    environment = dict(os.environ)
    # 本番と同じ探索設定に固定する(batched_tournament.pyの素の既定は旧値)。
    environment['SELFPLAY_VISIT_TIE_BREAK'] = 'q'
    environment['SELFPLAY_PRIOR_EXPONENT'] = '4.0'
    environment['SELFPLAY_DECISION_DUMP'] = str(dump)
    subprocess.run(command, cwd=MATCH_ROOT, check=True, env=environment,
                   stdout=subprocess.DEVNULL)


def report_value_calibration(episodes: Path) -> None:
    values = []
    outcomes = []
    files = sorted(episodes.glob('*.pkl'))
    for path in files:
        with path.open('rb') as handle:
            episode = pickle.load(handle)
        for packed in episode['packedPlayerSamples']:
            search_value = np.asarray(packed['searchValue'], dtype=np.float64)
            outcome = np.asarray(packed['value'], dtype=np.float64)
            finite = np.isfinite(search_value)
            values.append(search_value[finite])
            outcomes.append(outcome[finite])
    value = np.concatenate(values)
    outcome = np.concatenate(outcomes)
    decided = outcome != 0

    print(f'== valueヘッドの校正 (games={len(files):,} / 探索あり局面={len(value):,}) ==')
    print(f'  分布      : mean {value.mean():+.4f} sd {value.std():.4f} '
          f'|v|>0.99 {100 * np.mean(np.abs(value) > 0.99):.1f}% '
          f'|v|<0.2 {100 * np.mean(np.abs(value) < 0.2):.1f}%')
    print(f'  符号的中率: {100 * np.mean(np.sign(value[decided]) == np.sign(outcome[decided])):.1f}% '
          f'(決着局 n={int(decided.sum()):,})')
    constant = float(np.mean(outcome ** 2))
    print(f'  MSE(予測,結果)= {np.mean((value - outcome) ** 2):.4f} '
          f'(常に0を出す定数予測 = {constant:.4f})')
    print('  校正:')
    edges = [-1.01, -0.99, -0.9, -0.5, 0.0, 0.5, 0.9, 0.99, 1.01]
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (value >= low) & (value < high)
        if mask.sum() < 50:
            continue
        print(f'    [{low:+.2f},{high:+.2f}) n={int(mask.sum()):7,d} ({100 * mask.mean():5.1f}%)'
              f'  予測平均{value[mask].mean():+.3f}  実結果平均{outcome[mask].mean():+.3f}')


def report_decisions(dump: Path) -> None:
    records = []
    for path in sorted(dump.glob('decisions.*.jsonl')):
        import json
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    multi = [row for row in records if len(row['visits']) >= 2]
    if not multi:
        print('== 着手の決まり方: 対象決定なし ==')
        return

    decided_by = Counter()
    tie_sizes = []
    for row in multi:
        visits = row['visits']
        qualities = row['q']
        top = max(visits)
        tied = [index for index, visit in enumerate(visits) if visit == top]
        if len(tied) < 2:
            decided_by['visitの一意最大'] += 1
            continue
        tie_sizes.append(len(tied))
        best = max(qualities[index] for index in tied)
        if sum(1 for index in tied if qualities[index] == best) == 1:
            decided_by['Q値のtie-break'] += 1
        else:
            decided_by['prior (Q値も完全同値)'] += 1

    total = len(multi)
    print(f'\n== 着手の決まり方 (決定 {total:,}) ==')
    for name in ('visitの一意最大', 'Q値のtie-break', 'prior (Q値も完全同値)'):
        print(f'  {name:<22}: {100 * decided_by[name] / total:5.1f}%  ({decided_by[name]:,})')
    if tie_sizes:
        print(f'  visit同数の決定 {100 * len(tie_sizes) / total:.1f}% '
              f'(同数の子 平均 {sum(tie_sizes) / len(tie_sizes):.2f})')


def main() -> None:
    args = parse_args()
    weights = args.weights if args.weights.is_absolute() else MATCH_ROOT / args.weights
    workspace = MATCH_ROOT / 'results' / 'value_measure' / args.label
    workspace.mkdir(parents=True, exist_ok=True)

    print(f'weights={weights} label={args.label}')
    selfplay(args, weights, workspace)
    report_value_calibration(workspace / 'episodes')
    report_decisions(workspace / 'decisions')

    if not args.keep:
        shutil.rmtree(workspace / 'episodes', ignore_errors=True)
        shutil.rmtree(workspace / 'decisions', ignore_errors=True)


if __name__ == '__main__':
    main()
