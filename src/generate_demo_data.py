"""Создаёт полностью синтетический и воспроизводимый A/B-датасет.

НЕТ связи с исходными учебными строками. Нельзя трактовать демо-результаты
как повторный расчёт исходного эксперимента.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo"


def main():
    rng = np.random.default_rng(20261103)
    city_names = ["Москва", "Санкт-Петербург", "Новосибирск", "Казань", "Краснодар", "Челябинск", "Сочи", "Нижний Новгород", "Воронеж", "Архангельск", "Хабаровск"]
    city_p = np.array([.40, .23, .14, .06, .035, .035, .03, .02, .02, .02, .01]); city_p/=city_p.sum()
    n = 8000
    groups = rng.integers(0, 2, n)
    city = rng.choice(np.arange(1, 12), size=n, p=city_p)
    timestamps = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.uniform(0, 31 * 24 * 3600, n), unit="s")
    offsets = np.array([0, -.015, .012, -.020, .002, -.008, .008, .005, .015, -.025, .000])
    p = np.clip(.60 + .06 * groups + offsets[city - 1], .05, .95)
    orders = rng.binomial(1, p, n)
    events = pd.DataFrame({"id_view": np.arange(20000001, 20000001 + n), "time_view": timestamps,
                           "id_group": groups, "id_city": city, "nflag_order": orders})
    cities = pd.DataFrame({"id_city": range(1, 12), "name_city": city_names})
    DEMO.mkdir(parents=True, exist_ok=True)
    events.to_csv(DEMO / "events.csv", index=False)
    cities.to_csv(DEMO / "cities.csv", index=False)
    print(f"Created SYNTHETIC dataset with {n} observations in {DEMO}")


if __name__ == "__main__":
    main()
