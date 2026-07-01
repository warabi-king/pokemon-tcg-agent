"""cabtで利用可能なカードメタデータを表示する。

SDKにカードメタデータAPIが含まれる場合はそれを使う。
含まれない場合は、Kaggle Dataから取得したカードCSVを読む。
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

try:
    from cg.api import all_card_data
except ImportError:
    all_card_data = None

CSV_CANDIDATES = (
    ROOT / "data" / "EN_Card_Data.csv",
    ROOT / "data" / "JP_Card_Data.csv",
    ROOT / "data" / "EN Card Data.csv",
    ROOT / "data" / "JP Card Data.csv",
)


def main() -> None:
    if all_card_data is not None:
        cards = all_card_data()
        for card in cards:
            print(card)
        return

    for csv_path in CSV_CANDIDATES:
        if csv_path.exists():
            with csv_path.open(newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    print(row)
            return

    raise RuntimeError(
        "このSDKにはall_card_data()が含まれていません。"
        "Kaggle Dataから 'EN Card Data.csv' または 'JP Card Data.csv' を取得し、"
        "feat-simulator/data/ に配置してください。"
    )


if __name__ == "__main__":
    main()
