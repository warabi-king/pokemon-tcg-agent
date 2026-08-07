"""同じ重みのまま「探索・着手規則」だけを変えてA/B対戦させる。

学習を伴わずに規則の良し悪しを測るための道具。groupA_<cluster> と groupB_<cluster>
という名前で同じ重みを2回登録し、同じクラスタ(=同じデッキ)同士だけを対戦させる。
``SELFPLAY_*_AGENT_PREFIX`` 系の環境変数はagent名の接頭辞で効く側を切り替えられる
ので、``groupA_`` を指定すれば片側だけが新しい規則で戦う。

デッキ強度は勝率のばらつきの主成分(SD 12〜15pt)なので、集計はクラスタ対応づけの
t検定で行う。両群に同じデッキが出るミラー方式なのでデッキ差は差分で消える。

使い方:
    .venv/bin/python tools/run_rule_ab.py \\
        --weights agents/_convergence_study \\
        --games 40 --seed 101 --label margin1 \\
        --env SELFPLAY_VISIT_TIE_BREAK=q \\
        --group-a-env SELFPLAY_VISIT_TIE_MARGIN=1

``--env`` は両群に、``--group-a-env`` はgroupAだけに効かせたい値(実体は
``<NAME>`` と ``<NAME>_AGENT_PREFIX=groupA_`` の組)。
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

MATCH_ROOT = Path(__file__).resolve().parents[1]
RUNNER = MATCH_ROOT / 'tools' / 'run_matches_round_robin.py'
VENV_PYTHON = MATCH_ROOT / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
PYTHON = VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, required=True,
                        help='cluster_XX/src/model.pth を持つagentツリー(groupA側 兼 テンプレート)')
    parser.add_argument('--weights-b', type=Path, default=None,
                        help='groupB側の重み。省略すると--weightsと同じ(=規則だけのA/B)。'
                             'agentツリーでも、cluster_XX/model.pth だけを持つ'
                             'snapshotディレクトリでもよい')
    parser.add_argument('--clusters', type=int, default=16)
    parser.add_argument('--games', type=int, default=40,
                        help='1ミラーあたりの試合数(総試合数 = games * clusters)')
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--label', default='ab')
    parser.add_argument('--search-count', type=int, default=10)
    parser.add_argument('--max-turns', type=int, default=100)
    parser.add_argument('--max-selections', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument('--device', default='mps' if sys.platform == 'darwin' else 'cuda')
    parser.add_argument('--env', action='append', default=[],
                        help='両群に効かせる KEY=VALUE')
    parser.add_argument('--group-a-env', action='append', default=[],
                        help='groupAだけに効かせる KEY=VALUE (KEY_AGENT_PREFIX も自動設定)')
    parser.add_argument('--output-dir', type=Path,
                        default=MATCH_ROOT / 'results' / 'rule_ab')
    parser.add_argument('--keep-trees', action='store_true',
                        help='複製したagentツリー(1組 約3.2GB)を実行後も残す')
    return parser.parse_args()


def build_trees(
    weights: Path,
    weights_b: Path | None,
    clusters: list[str],
    workspace: Path,
) -> tuple[Path, Path]:
    """groupA用/groupB用のagentツリーを用意する。

    ``weights`` は必ず完全なagentツリー(src/main.py入り)で、テンプレートも兼ねる。
    ``weights_b`` は完全なagentツリーでも、convergence_studyのsnapshotのように
    ``cluster_XX/model.pth`` だけを持つディレクトリでもよい(その場合テンプレートを
    複製した上で重みだけ差し替える)。
    """
    roots = []
    for side, source in (('a', weights), ('b', weights_b or weights)):
        root = workspace / f'side_{side}'
        if root.exists():
            shutil.rmtree(root)
        if (source / clusters[0] / 'src' / 'main.py').is_file():
            shutil.copytree(source, root)
        else:
            # snapshotディレクトリ: テンプレートを複製して重みだけ上書きする。
            shutil.copytree(weights, root)
            overlaid = 0
            for cluster in clusters:
                for weight in ('model.pth', 'opponent_model.pth'):
                    origin = source / cluster / weight
                    if origin.is_file():
                        shutil.copy2(origin, root / cluster / 'src' / weight)
                        overlaid += 1
            if overlaid == 0:
                raise SystemExit(f'重みが1つも見つかりません: {source}')
            print(f'side_{side}: {source} の重み{overlaid}件を上書きしました')
        for cluster in clusters:
            main_path = root / cluster / 'src' / 'main.py'
            if not main_path.is_file():
                raise SystemExit(f'{cluster}: {main_path} がありません')
        roots.append(root)
    return roots[0], roots[1]


def run_tournament(
    args: argparse.Namespace,
    clusters: list[str],
    root_a: Path,
    root_b: Path,
    environment: dict[str, str],
) -> dict:
    agents: dict[str, Path] = {}
    for cluster in clusters:
        agents[f'groupA_{cluster}'] = root_a / cluster / 'src' / 'main.py'
    for cluster in clusters:
        agents[f'groupB_{cluster}'] = root_b / cluster / 'src' / 'main.py'

    names = list(agents)
    pairings = list(itertools.combinations(names, 2))
    counts = [
        args.games
        if (
            name0.startswith('groupA_')
            and name1.startswith('groupB_')
            and name0[len('groupA_'):] == name1[len('groupB_'):]
        )
        else 0
        for name0, name1 in pairings
    ]
    if sum(1 for count in counts if count > 0) != len(clusters):
        raise SystemExit(f'ミラー対戦カード数が{len(clusters)}になりません')

    results_dir = MATCH_ROOT / 'results'
    before = {path.name for path in results_dir.glob('tournament_*.json')}

    command = [str(PYTHON), str(RUNNER)]
    for name, main_path in agents.items():
        command.extend(['--agent', f'{name}={main_path}'])
    command.extend([
        '--backend', 'worker-batched',
        '--device', args.device,
        '--workers', str(args.workers),
        '--lanes', str(args.games * len(clusters)),
        '--batch-size', str(args.batch_size),
        '--search-count', str(args.search_count),
        '--max-turns', str(args.max_turns),
        '--max-selections', str(args.max_selections),
        '--seed', str(args.seed),
        '--no-self',
        '--quiet',
        '--save-json',
        '--games-per-pairing-json', json.dumps(counts),
    ])
    subprocess.run(command, cwd=MATCH_ROOT, check=True, env=environment)

    new_results = [
        path for path in results_dir.glob('tournament_*.json')
        if path.name not in before
    ]
    if not new_results:
        raise SystemExit('tournament jsonが生成されませんでした')
    latest = max(new_results, key=lambda path: path.stat().st_mtime_ns)
    return json.loads(latest.read_text(encoding='utf-8'))


def collect_per_cluster(payload: dict) -> dict[str, dict[str, int]]:
    per_cluster: dict[str, dict[str, int]] = {}
    for matchup, result in payload['head_to_head'].items():
        name0, name1 = matchup.split('_vs_', 1)
        if name0.startswith('groupA_'):
            a_wins, b_wins = result['name0_wins'], result['name1_wins']
            cluster = name0[len('groupA_'):]
        elif name1.startswith('groupA_'):
            a_wins, b_wins = result['name1_wins'], result['name0_wins']
            cluster = name1[len('groupA_'):]
        else:
            continue
        if a_wins + b_wins + result['draws'] + result['unresolved'] == 0:
            continue
        per_cluster[cluster] = {
            'a_wins': a_wins,
            'b_wins': b_wins,
            'draws': result['draws'],
            'unresolved': result['unresolved'],
        }
    return per_cluster


def summarize(per_cluster: dict[str, dict[str, int]]) -> dict:
    """二項検定とデッキ対応づけt検定の両方を返す。

    このプロジェクトが報告する「+Xpt」は勝率ではなく2群の差 gap = 2*(p-0.5) なので、
    SEも勝率のSEの2倍になる。試合は独立ではない(同じ16デッキが全ペアに出る)ため、
    デッキごとの勝率を1標本としたt検定を主指標にする。
    """
    a_total = sum(row['a_wins'] for row in per_cluster.values())
    b_total = sum(row['b_wins'] for row in per_cluster.values())
    decided = a_total + b_total
    win_rate = a_total / decided if decided else 0.0
    gap = 2.0 * (win_rate - 0.5)
    binomial_se_gap = 2.0 * math.sqrt(0.25 / decided) if decided else float('nan')
    z = (gap / binomial_se_gap) if decided else float('nan')

    rates = []
    for row in per_cluster.values():
        cluster_decided = row['a_wins'] + row['b_wins']
        if cluster_decided:
            rates.append(2.0 * (row['a_wins'] / cluster_decided - 0.5))
    n = len(rates)
    mean = sum(rates) / n if n else float('nan')
    if n > 1:
        variance = sum((value - mean) ** 2 for value in rates) / (n - 1)
        standard_error = math.sqrt(variance / n)
        t_stat = mean / standard_error if standard_error else float('nan')
        p_value = _t_two_sided(t_stat, n - 1)
    else:
        variance = standard_error = t_stat = p_value = float('nan')

    return {
        'decided_games': decided,
        'a_wins': a_total,
        'b_wins': b_total,
        'win_rate_a': win_rate,
        'gap_pt': gap * 100.0,
        'binomial_se_gap_pt': binomial_se_gap * 100.0,
        'binomial_z': z,
        'paired_n_decks': n,
        'paired_mean_gap_pt': mean * 100.0,
        'paired_se_gap_pt': standard_error * 100.0,
        'paired_sd_deck_pt': math.sqrt(variance) * 100.0 if n > 1 else float('nan'),
        'paired_t': t_stat,
        'paired_p': p_value,
    }


def _t_two_sided(t_stat: float, degrees: int) -> float:
    """自由度degreesのt分布の両側p値。"""
    if degrees <= 0 or not math.isfinite(t_stat):
        return float('nan')
    from scipy import stats

    return float(2.0 * stats.t.sf(abs(t_stat), degrees))


def main() -> None:
    args = parse_args()
    clusters = [f'cluster_{index:02d}' for index in range(args.clusters)]
    weights = args.weights if args.weights.is_absolute() else MATCH_ROOT / args.weights
    weights_b = args.weights_b
    if weights_b is not None and not weights_b.is_absolute():
        weights_b = MATCH_ROOT / weights_b

    workspace = MATCH_ROOT / 'agents' / f'_rule_ab_{args.label}'
    workspace.mkdir(parents=True, exist_ok=True)
    root_a, root_b = build_trees(weights, weights_b, clusters, workspace)

    environment = dict(os.environ)
    # 前のA/Bの環境変数が残っていると条件が汚れるので、規則系は一度全部落とす。
    for name in list(environment):
        if name.startswith('SELFPLAY_'):
            environment.pop(name)
    for entry in args.env:
        key, _, value = entry.partition('=')
        environment[key] = value
    for entry in args.group_a_env:
        key, _, value = entry.partition('=')
        environment[key] = value
        environment[f'{key}_AGENT_PREFIX'] = 'groupA_'

    print(f'label={args.label} games/mirror={args.games} total={args.games * len(clusters)}')
    print('shared env :', {k: v for k, v in sorted(environment.items()) if k.startswith('SELFPLAY_') and not k.endswith('_AGENT_PREFIX')})
    print('groupA only:', [entry.split('=')[0] for entry in args.group_a_env])

    started = time.time()
    payload = run_tournament(args, clusters, root_a, root_b, environment)
    elapsed = time.time() - started

    per_cluster = collect_per_cluster(payload)
    stats = summarize(per_cluster)
    stats.update({
        'label': args.label,
        'seed': args.seed,
        'games_per_mirror': args.games,
        'weights': str(weights),
        'weights_b': str(weights_b) if weights_b else str(weights),
        'shared_env': args.env,
        'group_a_env': args.group_a_env,
        'elapsed_seconds': elapsed,
        'per_cluster': per_cluster,
    })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f'{args.label}_seed{args.seed}.json'
    output.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')

    print(
        f"\n[{args.label}] groupA {stats['a_wins']}-{stats['b_wins']} "
        f"(decided={stats['decided_games']}, {elapsed:.0f}s)\n"
        f"  gap = {stats['gap_pt']:+.2f}pt  "
        f"binomial SE={stats['binomial_se_gap_pt']:.2f}pt z={stats['binomial_z']:+.2f}\n"
        f"  paired-by-deck: {stats['paired_mean_gap_pt']:+.2f}pt "
        f"SE={stats['paired_se_gap_pt']:.2f} deckSD={stats['paired_sd_deck_pt']:.2f} "
        f"t({stats['paired_n_decks'] - 1})={stats['paired_t']:+.2f} p={stats['paired_p']:.4f}\n"
        f"  -> {output}"
    )

    if not args.keep_trees:
        # ツリー1組で約3.2GB。条件を続けて回すとすぐ埋まるので既定で片付ける。
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == '__main__':
    main()
