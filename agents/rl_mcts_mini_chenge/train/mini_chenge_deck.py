"""簡易的な進化戦略でデッキを変更し、最良個体を別ファイルに保存する。"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from pathlib import Path
import random
import re
import sys
import tempfile

AGENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = AGENT_ROOT / "src"
TRAIN_ROOT = Path(__file__).resolve().parent
CARD_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "EN_Card_Data.csv"
sys.path.insert(0, str(SRC_ROOT))

DECK_SIZE = 60
MAX_GENERATION_ATTEMPTS = 10_000
DECK_FILE_PATTERN = re.compile(r"deck_(\d+)\.csv")


@dataclass(frozen=True)
class CardInfo:
    """デッキ構築ルールの検証に必要なカード情報。"""

    card_id: int
    name: str
    card_type: str
    basic: bool
    ace_spec: bool


@dataclass
class Individual:
    """進化戦略で扱うデッキ個体。"""

    deck: list[int]
    score: float = 0.0
    wins: int = 0
    losses: int = 0
    draws: int = 0


def read_card_data(path: Path) -> list[CardInfo]:
    """カードデータCSVを読み込み、カードIDごとに1件へまとめる。"""
    cards: dict[int, CardInfo] = {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            card_id = int(row["Card ID"])
            if card_id not in cards:
                card_type = row["Stage (Pokémon)/Type (Energy and Trainer)"]
                cards[card_id] = CardInfo(
                    card_id=card_id,
                    name=row["Card Name"],
                    card_type=card_type,
                    basic=card_type == "Basic Pokémon",
                    ace_spec=row["Rule"] == "ACE SPEC",
                )
    if not cards:
        raise ValueError(f"カードデータが空です: {path}")
    return list(cards.values())


def read_deck(path: Path) -> list[int]:
    """1行1カードIDのデッキCSVを読み込む。"""
    try:
        deck = [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except ValueError as error:
        raise ValueError(f"カードIDではない値が含まれています: {path}") from error

    if len(deck) != DECK_SIZE:
        raise ValueError(f"デッキは{DECK_SIZE}枚である必要があります: {path} ({len(deck)}枚)")
    return deck


def validate_deck(deck: list[int], card_map: dict[int, CardInfo]) -> bool:
    """ゲームの主要なデッキ構築ルールを満たすか確認する。"""
    if len(deck) != DECK_SIZE or any(card_id not in card_map for card_id in deck):
        return False

    cards = [card_map[card_id] for card_id in deck]
    name_counts = Counter(
        card.name for card in cards if card.card_type != "Basic Energy"
    )
    if any(count > 4 for count in name_counts.values()):
        return False
    if sum(card.ace_spec for card in cards) > 1:
        return False
    return any(card.basic for card in cards)


def change_deck(
    deck: list[int],
    m: int,
    cards: list[CardInfo],
    rng: random.Random,
) -> list[int]:
    """m個の異なる位置をランダムな別カードへ変更する。"""
    if not 1 <= m <= len(deck):
        raise ValueError(f"mは1から{len(deck)}の範囲で指定してください: {m}")

    card_ids = [card.card_id for card in cards]
    card_map = {card.card_id: card for card in cards}
    if any(card_id not in card_map for card_id in deck):
        raise ValueError("元デッキに無効なカードIDが含まれています。")

    for _ in range(MAX_GENERATION_ATTEMPTS):
        changed = deck.copy()
        positions = rng.sample(range(len(deck)), m)
        for position in positions:
            original_card_id = deck[position]
            replacement = rng.choice(card_ids)
            while replacement == original_card_id:
                replacement = rng.choice(card_ids)
            changed[position] = replacement

        if validate_deck(changed, card_map):
            return changed

    raise RuntimeError(
        f"{MAX_GENERATION_ATTEMPTS}回試行しましたが、有効なデッキを生成できませんでした。"
        "mを小さくして再実行してください。"
    )


def raise_for_deck_error(start_data) -> None:
    """battle_startが返すデッキエラーを例外にする。"""
    if start_data.errorPlayer < 0:
        return

    errors = {
        1: "無効なカードIDが含まれています。",
        2: "同名カードが上限を超えています。",
        3: "たねポケモンが含まれていません。",
        4: "ACE SPECが上限を超えています。",
    }
    raise ValueError(errors.get(start_data.errorType, "不正なデッキです。"))


def evaluate_individual(
    individual: Individual,
    opponent_deck: list[int],
    model,
    games: int,
    search_count: int,
) -> None:
    """変更元デッキとの対戦結果から個体のfitnessを計算する。"""
    import torch

    from cg.game import battle_finish, battle_select, battle_start
    from rl_mcts.mcts import mcts_agent

    wins = losses = draws = 0
    with torch.inference_mode():
        for game_index in range(games):
            candidate_index = game_index % 2
            decks = (
                [individual.deck, opponent_deck]
                if candidate_index == 0
                else [opponent_deck, individual.deck]
            )
            obs, start_data = battle_start(decks[0], decks[1])
            try:
                raise_for_deck_error(start_data)
                while obs["current"]["result"] < 0:
                    player_index = obs["current"]["yourIndex"]
                    selected, _ = mcts_agent(
                        obs,
                        decks[player_index],
                        model,
                        search_count=search_count,
                    )
                    obs = battle_select(selected)
            finally:
                battle_finish()

            result = obs["current"]["result"]
            if result == 2:
                draws += 1
            elif result == candidate_index:
                wins += 1
            else:
                losses += 1

    individual.wins = wins
    individual.losses = losses
    individual.draws = draws
    individual.score = (wins + 0.5 * draws) / games


def load_model(model_path: Path):
    """進化戦略の対戦評価に使う学習済みモデルを読み込む。"""
    import torch

    from rl_mcts.model import create_model

    if not model_path.exists():
        raise FileNotFoundError(f"学習済みモデルが見つかりません: {model_path}")
    model = create_model()
    state = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state)
    model.eval()
    return model


def evolve_deck(
    initial_deck: list[int],
    cards: list[CardInfo],
    model,
    generations: int,
    population_size: int,
    elite_count: int,
    games: int,
    search_count: int,
    mutation_count: int,
    rng: random.Random,
) -> Individual:
    """エリート選択とランダム変異による簡易的な(μ+λ)進化戦略。"""
    if generations < 1:
        raise ValueError("generationsは1以上で指定してください。")
    if population_size < 2:
        raise ValueError("population-sizeは2以上で指定してください。")
    if not 1 <= elite_count < population_size:
        raise ValueError("elite-countは1以上かつpopulation-size未満で指定してください。")
    if games < 1:
        raise ValueError("gamesは1以上で指定してください。")
    if search_count < 1:
        raise ValueError("search-countは1以上で指定してください。")

    population = [Individual(initial_deck.copy())]
    while len(population) < population_size:
        population.append(
            Individual(change_deck(initial_deck, mutation_count, cards, rng))
        )

    for generation in range(1, generations + 1):
        print(f"Generation {generation}/{generations}")
        for index, individual in enumerate(population, start=1):
            evaluate_individual(
                individual,
                initial_deck,
                model,
                games,
                search_count,
            )
            print(
                f"  Individual {index}/{len(population)}: "
                f"score={individual.score:.3f} "
                f"(W/L/D={individual.wins}/{individual.losses}/{individual.draws})"
            )

        population.sort(key=lambda individual: individual.score, reverse=True)
        elites = population[:elite_count]
        if generation == generations:
            return elites[0]

        next_population = [
            Individual(
                elite.deck.copy(),
                elite.score,
                elite.wins,
                elite.losses,
                elite.draws,
            )
            for elite in elites
        ]
        while len(next_population) < population_size:
            parent = rng.choice(elites)
            child_deck = change_deck(parent.deck, mutation_count, cards, rng)
            next_population.append(Individual(child_deck))
        population = next_population

    raise AssertionError("到達不能な処理です。")


def next_output_path(output_dir: Path) -> Path:
    """deck_N.csvのうち、未使用の最小番号を返す。"""
    used_numbers = {
        int(match.group(1))
        for path in output_dir.glob("deck_*.csv")
        if (match := DECK_FILE_PATTERN.fullmatch(path.name))
    }
    number = 1
    while number in used_numbers:
        number += 1
    return output_dir / f"deck_{number}.csv"


def write_deck(output_dir: Path, deck: list[int]) -> Path:
    """番号が重複しないファイルへデッキを書き込む。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    while True:
        output_path = next_output_path(output_dir)
        try:
            with output_path.open("x", encoding="utf-8", newline="") as file:
                file.writelines(f"{card_id}\n" for card_id in deck)
            return output_path
        except FileExistsError:
            # 並行実行で同じ番号が先に作られた場合は、次の番号を探す。
            continue


def apply_deck(
    selected_deck_path: Path,
    destination: Path = SRC_ROOT / "deck.csv",
) -> Path:
    """指定された採用デッキを提出用のsrc/deck.csvへ反映する。"""
    deck = read_deck(selected_deck_path)
    cards = read_card_data(CARD_DATA_PATH)
    card_map = {card.card_id: card for card in cards}
    if not validate_deck(deck, card_map):
        raise ValueError(
            f"採用デッキが構築ルールを満たしていません: {selected_deck_path}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            file.writelines(f"{card_id}\n" for card_id in deck)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("m", type=int, help="1回の変異で変更するカード枚数")
    parser.add_argument(
        "--strategy",
        choices=("random", "genetic"),
        default="genetic",
        help="デッキ変更戦略。randomは1回だけ変更し、geneticは対戦評価を繰り返す",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=SRC_ROOT / "deck.csv",
        help="変更元のデッキCSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TRAIN_ROOT,
        help="deck_N.csvの保存先",
    )
    parser.add_argument("--generations", type=int, default=3, help="世代数")
    parser.add_argument(
        "--population-size",
        type=int,
        default=8,
        help="各世代の個体数",
    )
    parser.add_argument(
        "--elite-count",
        type=int,
        default=2,
        help="次世代へ残す上位個体数",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=10,
        help="各個体を評価する対戦数",
    )
    parser.add_argument(
        "--search-count",
        type=int,
        default=5,
        help="各手のMCTS探索回数",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=SRC_ROOT / "model.pth",
        help="対戦評価に使う学習済みモデル",
    )
    parser.add_argument("--seed", type=int, default=None, help="乱数seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deck = read_deck(args.input)
    cards = read_card_data(CARD_DATA_PATH)
    card_map = {card.card_id: card for card in cards}
    if not validate_deck(deck, card_map):
        raise ValueError(f"変更元デッキが構築ルールを満たしていません: {args.input}")

    rng = random.Random(args.seed)
    if args.strategy == "random":
        changed_deck = change_deck(deck, args.m, cards, rng)
        output_path = write_deck(args.output_dir, changed_deck)
        print(f"Strategy: {args.strategy}")
    else:
        model = load_model(args.model)
        best = evolve_deck(
            initial_deck=deck,
            cards=cards,
            model=model,
            generations=args.generations,
            population_size=args.population_size,
            elite_count=args.elite_count,
            games=args.games,
            search_count=args.search_count,
            mutation_count=args.m,
            rng=rng,
        )
        output_path = write_deck(args.output_dir, best.deck)
        print(
            f"Best deck: score={best.score:.3f} "
            f"(W/L/D={best.wins}/{best.losses}/{best.draws})"
        )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
