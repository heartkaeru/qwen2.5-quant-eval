"""Проверить задания до загрузки в Colab: python scripts/validate_dataset.py."""

import argparse
from collections import Counter
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.dataset import load_tests


def main():
    parser = argparse.ArgumentParser(description="Проверка формата JSONL-заданий")
    parser.add_argument("path", nargs="?", type=Path, default=REPO_ROOT / "data/prompts.jsonl")
    args = parser.parse_args()
    try:
        tests = load_tests(args.path)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Ошибка: {error}\n")
    print(f"OK: {len(tests)} заданий в {args.path}")
    for category, count in sorted(Counter(test["category"] for test in tests).items()):
        print(f"  {category}: {count}")
    manual = sum(test["check"] == "manual" for test in tests)
    print(f"Автоматических: {len(tests) - manual}; ручных: {manual}")
    print("Формат проверен. Правильность эталонных ответов нужно проверить самостоятельно.")


if __name__ == "__main__":
    main()
