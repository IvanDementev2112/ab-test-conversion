"""Статистические функции для учебного A/B-кейса.

Единица наблюдения: id_view (отдельный просмотр/событие).
Независимость просмотров не доказана: без user_id стандартные p-value/CI
корректны только при условии независимости строк и правильной рандомизации.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chisquare
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest

COLUMNS = {"id_view", "time_view", "id_group", "id_city", "nflag_order"}
PEAK_HOURS = (7, 8, 9, 10, 17, 18, 19, 20)


def prepare_events(events: pd.DataFrame, cities: pd.DataFrame) -> pd.DataFrame:
    """Валидация данных, объединение и приведение типов; fail-fast."""
    miss = COLUMNS - set(events.columns)
    if miss:
        raise ValueError(f"Нет обязательных столбцов: {sorted(miss)}")
    if not {"id_city", "name_city"} <= set(cities.columns):
        raise ValueError("Справочник городов требует id_city и name_city")
    if events[list(COLUMNS)].isna().any().any():
        raise ValueError("Есть пропуски в ключевых столбцах")
    if events["id_view"].duplicated().any():
        raise ValueError("Найдены повторяющиеся id_view")
    if not set(events["id_group"].unique()).issubset({0, 1}):
        raise ValueError("id_group должен содержать 0/1")
    if not set(events["nflag_order"].unique()).issubset({0, 1}):
        raise ValueError("nflag_order должен содержать 0/1")
    if cities["id_city"].duplicated().any():
        raise ValueError("Дублирующиеся id_city в справочнике городов")
    df = events.merge(cities[["id_city", "name_city"]], on="id_city", how="left", validate="many_to_one")
    if df["name_city"].isna().any():
        raise ValueError("Не найден город для части событий")
    df["time_view"] = pd.to_datetime(df["time_view"], errors="coerce")
    if df["time_view"].isna().any():
        raise ValueError("Не удалось преобразовать time_view в дату/время")
    if not {0, 1} <= set(df["id_group"].unique()):
        raise ValueError("В данных должна быть хотя бы одна строка каждой группы")
    return df


def group_table(df: pd.DataFrame) -> pd.DataFrame:
    """Число просмотров, заказов и конверсия в каждой группе."""
    out = (df.groupby("id_group")["nflag_order"]
           .agg(n="size", orders="sum", conversion="mean")
           .reindex([0, 1]))
    return out


def experiment_result(df: pd.DataFrame) -> dict:
    """Двусторонний z-тест двух долей и Newcombe 95% CI (B-A)."""
    gt = group_table(df)
    n_a, n_b = int(gt.loc[0, "n"]), int(gt.loc[1, "n"])
    y_a, y_b = int(gt.loc[0, "orders"]), int(gt.loc[1, "orders"])
    rate_a, rate_b = y_a / n_a, y_b / n_b
    z_stat, pval = proportions_ztest([y_b, y_a], [n_b, n_a], alternative="two-sided")
    low, high = confint_proportions_2indep(y_b, n_b, y_a, n_a, method="newcomb", compare="diff", alpha=0.05)
    return {
        "a_n": n_a, "b_n": n_b, "a_orders": y_a, "b_orders": y_b,
        "a_conversion": rate_a, "b_conversion": rate_b,
        "effect_pp": (rate_b - rate_a) * 100,
        "relative_effect_pct": (rate_b / rate_a - 1) * 100 if rate_a else float("nan"),
        "ci_low_pp": low * 100, "ci_high_pp": high * 100,
        "z": float(z_stat), "p_value": float(pval),
    }


def sample_ratio_mismatch(df: pd.DataFrame, expected_control_share: float = 0.5) -> dict:
    """SRM-проверка количества наблюдений. Не доказывает корректность рандомизации."""
    counts = df["id_group"].value_counts().reindex([0, 1], fill_value=0)
    n = counts.sum()
    expected = [n * expected_control_share, n * (1 - expected_control_share)]
    stat, pval = chisquare(counts.to_numpy(), f_exp=expected)
    return {"control_n": int(counts.loc[0]), "test_n": int(counts.loc[1]),
            "chi2": float(stat), "p_value": float(pval)}


def by_city(df: pd.DataFrame) -> pd.DataFrame:
    """Описательные сегментные сравнения; Holm для 11 гипотез."""
    results = []
    for city, part in df.groupby("name_city", sort=True):
        if part["id_group"].nunique() != 2:
            continue
        results.append({"city": city, **experiment_result(part)})
    out = pd.DataFrame(results)
    if out.empty:
        return out
    out["p_holm"] = multipletests(out["p_value"].to_numpy(), alpha=.05, method="holm")[1]
    out["significant_holm"] = out["p_holm"] < .05
    return out.sort_values(["a_n", "b_n"], ascending=False).reset_index(drop=True)


def by_peak_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Разрез по пиковым/непиковым часам (заданный набор 7-10, 17-20)."""
    work = df.copy()
    work["peak_period"] = np.where(work["time_view"].dt.hour.isin(PEAK_HOURS), "Пиковые часы", "Непиковые часы")
    result = [{"period": period, **experiment_result(part)} for period, part in work.groupby("peak_period")]
    return pd.DataFrame(result)
