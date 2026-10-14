"""Формат заданий для экспериментов. Только стандартная библиотека Python."""

import json
from pathlib import Path


def normalize(text):
    """Пробелы по краям и регистр не влияют на exact/one_of."""
    return text.strip().casefold()


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def check_answer(test, answer):
    kind = test["check"]
    if kind == "manual":
        return None
    if kind == "exact":
        return normalize(answer) == normalize(test["expected"])
    if kind == "one_of":
        return normalize(answer) in {normalize(item) for item in test["expected"]}
    if kind == "json":
        try:
            def reject_constant(value):
                raise ValueError(f"Недопустимое значение JSON: {value}")

            def unique_keys(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError(f"Повторяющийся ключ JSON: {key}")
                    result[key] = value
                return result

            actual = json.loads(answer, parse_constant=reject_constant,
                                object_pairs_hook=unique_keys)
            return canonical_json(actual) == canonical_json(test["expected"])
        except (ValueError, TypeError):
            return False
    raise ValueError(f"Неизвестная проверка: {kind}")


def validate_tests(tests):
    if not tests:
        raise ValueError("Набор заданий пуст.")
    seen = set()
    for index, test in enumerate(tests, 1):
        prefix = f"Задание {index}"
        if not isinstance(test, dict):
            raise ValueError(f"{prefix}: ожидается JSON-объект.")
        for key in ["id", "category", "prompt", "check"]:
            if not isinstance(test.get(key), str) or not test[key].strip():
                raise ValueError(f"{prefix}: требуется непустая строка {key!r}.")
        identifier = test["id"]
        if not all(c.isalnum() or c in "_-" for c in identifier):
            raise ValueError(f"{prefix}: в id разрешены буквы, цифры, _ и -.")
        if identifier in seen:
            raise ValueError(f"{prefix}: повторяющийся id {identifier!r}.")
        seen.add(identifier)
        kind = test["check"]
        if kind not in {"exact", "one_of", "json", "manual"}:
            raise ValueError(f"{prefix}: check должен быть exact, one_of, json или manual.")
        if kind == "manual":
            if not isinstance(test.get("rubric"), str) or not test["rubric"].strip():
                raise ValueError(f"{prefix}: для ручной проверки нужен rubric с критериями.")
            continue
        if "expected" not in test:
            raise ValueError(f"{prefix}: отсутствует expected.")
        expected = test["expected"]
        if kind == "exact" and not isinstance(expected, str):
            raise ValueError(f"{prefix}: expected для exact должен быть строкой.")
        if kind == "one_of" and not (
            isinstance(expected, list) and expected
            and all(isinstance(item, str) and item.strip() for item in expected)
        ):
            raise ValueError(f"{prefix}: expected для one_of — непустой список строк.")
        if kind == "json":
            try:
                canonical_json(expected)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{prefix}: expected не является допустимым JSON.") from error
    return tests


def load_tests(path):
    tests = []
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                tests.append(json.loads(line))
            except ValueError as error:
                raise ValueError(f"{path}, строка {line_number}: неверный JSON: {error}") from error
    return validate_tests(tests)
