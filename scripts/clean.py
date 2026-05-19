#!/usr/bin/env python3
import argparse
from pathlib import Path
import re

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "1) Удаляет строки с пустыми значениями в первых двух столбцах.\n"
            "2) Сначала удаляет почти-дубликаты по первому столбцу (по Jaccard),\n"
            "   затем по второму столбцу."
        )
    )
    parser.add_argument(
        "input_file",
        help="Путь к исходному Excel-файлу (.xlsx)",
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.95,
        help="Порог Jaccard-похожести для «почти одинаковых» значений (0..1). "
             "По умолчанию 0.8.",
    )
    parser.add_argument(
        "-o", "--output",
        help="Имя выходного файла (по умолчанию: <имя>_cleaned.xlsx)",
        default=None,
    )
    return parser.parse_args()


_token_re = re.compile(r"\w+", re.UNICODE)


def value_to_tokens(value) -> set[str]:
    """
    Преобразуем значение ячейки в множество токенов.
    Используем \w+ (слова/цифры/подчёркивания), приводим к нижнему регистру.
    """
    if pd.isna(value):
        return set()
    text = str(value).strip().lower()
    if not text:
        return set()
    return set(_token_re.findall(text))


def jaccard(tokens1: set[str], tokens2: set[str]) -> float:
    if not tokens1 and not tokens2:
        return 0.0
    inter = tokens1 & tokens2
    union = tokens1 | tokens2
    return len(inter) / len(union) if union else 0.0


def deduplicate_similar_in_column(df: pd.DataFrame, col: str, threshold: float):
    """
    Удаляет почти одинаковые строки по одному столбцу:
    - сравниваем только значения в этом столбце;
    - если Jaccard >= threshold, считаем значения «почти одинаковыми»;
    - оставляем первую строку, остальные похожие удаляем.

    Возвращает:
      df_dedup: DataFrame после удаления почти-дубликатов по этому столбцу
      groups: список групп похожих строк (список списков исходных индексов)
    """
    if df.empty:
        return df, []

    # токены для каждого значения
    tokens_list = [value_to_tokens(df.iloc[i][col]) for i in range(len(df))]

    n = len(df)
    to_drop = set()
    groups = []

    for i in range(n):
        if i in to_drop:
            continue
        group = [df.index[i]]
        tokens_i = tokens_list[i]
        for j in range(i + 1, n):
            if j in to_drop:
                continue
            sim = jaccard(tokens_i, tokens_list[j])
            if sim >= threshold:
                to_drop.add(j)
                group.append(df.index[j])
        if len(group) > 1:
            groups.append(group)

    df_dedup = df.drop(index=[df.index[j] for j in to_drop])
    return df_dedup, groups


def main():
    args = parse_args()
    input_path = Path(args.input_file)

    if not input_path.is_file():
        print(f"Файл '{input_path}' не найден.")
        return

    df = pd.read_excel(input_path)

    if df.empty:
        print("Файл пуст или не содержит данных.")
        return

    # Проверяем, что есть как минимум два столбца
    if len(df.columns) < 2:
        print("Ожидается минимум два столбца в файле.")
        return

    col1 = df.columns[0]
    col2 = df.columns[1]
    print(f"Первый столбец: '{col1}', второй столбец: '{col2}'")

    # 1. Удаляем строки, где хотя бы в одном из двух столбцов пусто
    before_rows = len(df)
    not_empty_col1 = df[col1].notna() & (df[col1].astype(str).str.strip() != "")
    not_empty_col2 = df[col2].notna() & (df[col2].astype(str).str.strip() != "")
    mask_not_empty = not_empty_col1 & not_empty_col2
    df_clean = df[mask_not_empty].copy()
    removed_empty = before_rows - len(df_clean)
    print(f"Удалено строк с пустыми значениями в '{col1}' или '{col2}': {removed_empty}")

    # 2. Сначала удаляем похожие по первому столбцу
    print(f"\nШаг 1: поиск почти одинаковых строк по столбцу '{col1}' "
          f"(Jaccard ≥ {args.threshold})...")
    df_step1, groups1 = deduplicate_similar_in_column(df_clean, col1, args.threshold)
    removed_step1 = len(df_clean) - len(df_step1)
    print(f"Удалено строк как почти-дубликатов по '{col1}': {removed_step1}")

    if groups1:
        print("\nГруппы похожих строк по первому столбцу (оставлена первая строка):")
        for idx, group in enumerate(groups1, start=1):
            print(f"  Группа {idx}: индексы строк: {group}")

    # 3. Затем удаляем похожие по второму столбцу из уже очищенного набора
    print(f"\nШаг 2: поиск почти одинаковых строк по столбцу '{col2}' "
          f"(Jaccard ≥ {args.threshold})...")
    df_step2, groups2 = deduplicate_similar_in_column(df_step1, col2, args.threshold)
    removed_step2 = len(df_step1) - len(df_step2)
    print(f"Удалено строк как почти-дубликатов по '{col2}': {removed_step2}")

    if groups2:
        print("\nГруппы похожих строк по второму столбцу (оставлена первая строка):")
        for idx, group in enumerate(groups2, start=1):
            print(f"  Группа {idx}: индексы строк: {group}")

    # Итоговый результат
    df_result = df_step2

    # 4. Сохраняем результат
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(input_path.stem + "cleaned.xlsx")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_result.to_excel(writer, index=False, sheet_name="cleaned")

    print(f"\nГотово. Итоговое количество строк: {len(df_result)}")
    print(f"Очищенные данные сохранены в: {output_path}")


if __name__ == "__main__":
    main()