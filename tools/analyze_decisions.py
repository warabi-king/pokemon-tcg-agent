"""SELFPLAY_DECISION_DUMP が残したroot決定の内訳を集計する。

「今の規則で着手が実際に何によって決まっているか」を実測する。
2026-08-05の計測(305局面)はtie-break導入前のもので、当時とは
prior exponent(10→4)もtie-break(enum→q)も変わっているため取り直す。
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def load(directory: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(directory.glob('decisions.*.jsonl')):
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--margin', type=int, default=1,
                        help='near-tie marginをこの値にしたとき何%%の決定が変わるか')
    args = parser.parse_args()

    records = load(args.directory)
    multi = [row for row in records if len(row['visits']) >= 2]
    print(f'decisions          : {len(records)} (展開済みの子が2つ以上: {len(multi)})')
    if not multi:
        return

    legal = [row['legal'] for row in multi]
    expanded = [len(row['visits']) for row in multi]
    print(f'legal actions      : mean {statistics.mean(legal):.2f} median {statistics.median(legal):.0f}')
    print(f'expanded children  : mean {statistics.mean(expanded):.2f} '
          f'({100 * statistics.mean([e / l for e, l in zip(expanded, legal)]):.1f}% of legal)')

    exact_tie = 0
    tie_sizes: list[int] = []
    decided_by = Counter()
    q_spread_in_tie: list[float] = []
    margin_changes = 0
    margin_tie_sizes: list[int] = []

    for row in multi:
        visits = row['visits']
        qualities = row['q']
        priors = row['prior']
        top = max(visits)
        tied = [index for index, visit in enumerate(visits) if visit == top]
        if len(tied) >= 2:
            exact_tie += 1
            tie_sizes.append(len(tied))
            tie_q = [qualities[index] for index in tied]
            q_spread_in_tie.append(max(tie_q) - min(tie_q))
            best_q = max(tie_q)
            q_best = [index for index in tied if qualities[index] == best_q]
            decided_by['Q (tie-break)' if len(q_best) == 1 else 'prior (Q also tied)'] += 1
        else:
            decided_by['visit (unique max)'] += 1

        # near-tie margin を入れたら着手が変わる決定の割合
        current = _pick(visits, qualities, priors, margin=0)
        widened = _pick(visits, qualities, priors, margin=args.margin)
        if current != widened:
            margin_changes += 1
        threshold = top - args.margin
        margin_tie_sizes.append(sum(1 for visit in visits if visit >= threshold))

    total = len(multi)
    print()
    print(f'exact visit tie    : {100 * exact_tie / total:.1f}% '
          f'(tied children mean {statistics.mean(tie_sizes):.2f})' if tie_sizes else 'exact visit tie    : 0%')
    print('final move decided by:')
    for name, count in decided_by.most_common():
        print(f'  {name:<22}: {100 * count / total:5.1f}%  ({count})')
    if q_spread_in_tie:
        print(f'Q spread inside tie: mean {statistics.mean(q_spread_in_tie):.4f} '
              f'median {statistics.median(q_spread_in_tie):.4f} '
              f'p90 {sorted(q_spread_in_tie)[int(0.9 * len(q_spread_in_tie))]:.4f}')
    print()
    print(f'margin={args.margin}: tie集合 mean {statistics.mean(margin_tie_sizes):.2f}, '
          f'着手が変わる決定 {100 * margin_changes / total:.1f}% ({margin_changes}/{total})')

    root_nn = [row['root_nn'] for row in multi if row.get('root_nn') is not None]
    if root_nn:
        print()
        print(f'root NN value      : mean {statistics.mean(root_nn):+.4f} '
              f'sd {statistics.pstdev(root_nn):.4f} '
              f'|v|>0.9 {100 * sum(1 for v in root_nn if abs(v) > 0.9) / len(root_nn):.1f}% '
              f'|v|<0.2 {100 * sum(1 for v in root_nn if abs(v) < 0.2) / len(root_nn):.1f}%')


def _pick(visits: list[int], qualities: list[float], priors: list[float], margin: int) -> int:
    threshold = max(visits) - margin
    candidates = [index for index, visit in enumerate(visits) if visit >= threshold]
    if margin == 0:
        return max(candidates, key=lambda index: (visits[index], qualities[index], priors[index]))
    return max(candidates, key=lambda index: (qualities[index], priors[index], visits[index]))


if __name__ == '__main__':
    main()
