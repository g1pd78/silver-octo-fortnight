"""
Прогноз сборов (факт против плана) на любой месяц — по каждому дню, с выполнением плана и recovery rate.

ГЛАВНАЯ МЕТРИКА
  Эффективность = факт / план и прогноз / план. Цель — разница между ними не больше 1 п.п.
  Бэктест, подбор параметров и отчёт считаются по этой разнице: на конец месяца (в зависимости от того,
  сколько дней месяца уже известно) и на каждую дату месяца (накопленный прогноз / накопленный план против
  накопленного факта / накопленного плана). Если план ещё не загружен, эффективность прогноза считается
  к спрогнозированному плану.

ЛОГИКА (как в первой версии)
  1. Календарные признаки по производственному календарю РФ 2024–2027.
  2. Градиентный бустинг предсказывает, какую долю МЕСЯЧНОГО плана соберут в каждый день.
  3. Сравнение с «референсными месяцами» (ваш текущий подход) и со смесью методов.
  4. Внутри месяца остаток корректируется по тому, как месяц идёт к текущей дате.
  5. Бэктест: ошибка итога месяца в зависимости от того, сколько дней месяца уже известно.

ЧТО ДОБАВЛЕНО
  6. Если план месяца ещё не загружен — он прогнозируется: итог месяца по тренду прошлых планов,
     раскладка по дням — по последним месяцам. Бэктест считается в двух режимах:
     «план загружен» и «план ещё не загружен» (план месяца спрогнозирован).
  7. Recovery rate (если задан DEBT_COL): на каждый день факт / долг, накопительно Σ факт / Σ долг
     с начала месяца и с начала данных. Долг на будущие дни прогнозируется так же, как план.
     Признаки долга (recovery прошлых месяцев, долг к плану) проверяются бэктестом: остаются, только если помогают.
  8. Ещё один кандидат в бэктесте — «бустинг × уровень»: бустинг без признаков уровня (только календарь и план)
     × уровень выполнения, очищенный от календаря и сглаженный по прошлым месяцам. Нужен потому, что признак
     «выполнение прошлого месяца» путает календарь с уровнем: предоплата в апреле перед майскими праздниками
     поднимает прогноз мая. Берётся (сам или в смеси), только если в бэктесте точнее.
  9. Полный набор на любой месяц (TARGET_MONTHS): прошлый, текущий или будущий — таблица по каждому дню.
     forecast            — прогноз на КАЖДЫЙ день месяца; факт этого месяца в него не попадает никогда
                           (для текущего месяца — прогноз, сделанный 1-го числа). По нему — проверка по факту.
     fact_plus_forecast  — факт, где он есть, + прогноз остатка с поправкой по факту на сегодня: куда придёт месяц.
     Для обоих — накопленные суммы, эффективность на дату (eff_*_pct), разница с фактом (eff_diff_pp), recovery rate.

Вход: DataFrame с колонками дата / факт / план [/ долг] — значения ЗА ДЕНЬ.
Будущие даты не нужны. Если план месяца не загружен, оставьте его пустым или не выгружайте эти даты.

Зависимости: pandas, numpy, scikit-learn, matplotlib. lightgbm — по желанию, используется, если установлен.
Запуск: заполните load_data() и выполните файл, или в Jupyter:  from collections_forecast import run;  res = run(df)
Настройки — в блоке НАСТРОЙКИ ниже. Прогон 2–4 минуты (с lightgbm быстрее; N_SEEDS = 1 — около минуты, для черновых прогонов).
"""

import warnings

import numpy as np
import pandas as pd

# ============================================================================
# НАСТРОЙКИ
# ============================================================================
DATE_COL, FACT_COL, PLAN_COL = "date", "fact", "plan"
DEBT_COL = None                # колонка «сумма долгов на дату» для recovery rate, например "debt"; None — без RR
CUMULATIVE_INPUT = False       # True, если факт и план в выгрузке накопленные с начала месяца
LAST_FACT_DATE = None          # последняя дата с ПОЛНЫМ фактом. None = последняя дата, где факт не пустой.
                               # Если в выгрузке есть неполный «сегодня», укажите вчерашнюю дату.
TARGET_MONTHS = None           # за какие месяцы выдать полный прогноз по дням, например ["2026-10", "2026-11"];
                               # None = текущий месяц и HORIZON_MONTHS следующих
HORIZON_MONTHS = 1             # сколько месяцев после текущего прогнозировать, если TARGET_MONTHS = None
RR_DENOM = "sum"               # recovery rate на дату: "sum" — Σ факт / Σ долг за те же дни;
                               # "last" — Σ факт / долг на эту дату (если долг — остаток портфеля на дату)
BACKTEST_FROM = "2025-01"      # с какого месяца считать бэктест (более ранние месяцы — только обучение)
PAYDAYS = (5, 10, 15, 20, 25)  # типичные дни зарплаты/аванса клиентов; выходной -> предыдущий рабочий день
N_REF_MONTHS = 3               # сколько предыдущих месяцев брать в методе референсных месяцев
RECENCY_HALFLIFE = 12          # вес наблюдения падает вдвое каждые N месяцев давности (None — без весов)
EXCLUDE_DATES = []             # аномальные даты, которые не брать в обучение, например ["2025-03-14"]
REPORT_KNOWN_DAYS = (0, 1, 3, 5, 10, 15, 20, 25)  # «сколько дней месяца уже известно» в отчёте
ALPHAS = np.round(np.arange(0, 1.01, 0.1), 2)          # сила коррекции остатка по ходу месяца
LEVEL_HALFLIVES = (1, 2, 3, 6, 12, np.inf)   # «бустинг × уровень»: вес прошлого месяца падает вдвое каждые N мес.
LEVEL_PRIOR_WEIGHTS = (0, 1, 3, 10, np.inf)  # «бустинг × уровень»: тяга уровня к среднему (inf = не подстраивать)
PROFILE_MONTHS = 6             # по скольким последним месяцам раскладывать прогноз плана/долга по дням
TREND_HALFLIVES = (0, 1, 3, 6, 12, np.inf)  # темп роста плана/долга: 0 = без роста, иначе полураспад весов, мес.
OUT_DIR = "."
RANDOM_STATE = 42
N_SEEDS = 5                    # сколько бустингов с разными seed усреднять (1 — быстрее, но прогноз «дрожит»)

# ============================================================================
# ПРОИЗВОДСТВЕННЫЙ КАЛЕНДАРЬ РФ
# Нерабочие дни (праздники и перенесённые выходные). Обычные сб/вс считаются автоматически.
# 2024–2026: КонсультантПлюс / Главкнига; 2027: постановление № 1187 от 17.09.2026 (по данным БУХ.1С).
# При появлении календаря на следующий год просто допишите его сюда.
# ============================================================================
RU_NONWORKING = {
    2024: ["01-01", "01-02", "01-03", "01-04", "01-05", "01-06", "01-07", "01-08",
           "02-23", "03-08", "04-29", "04-30", "05-01", "05-09", "05-10",
           "06-12", "11-04", "12-30", "12-31"],
    2025: ["01-01", "01-02", "01-03", "01-04", "01-05", "01-06", "01-07", "01-08",
           "05-01", "05-02", "05-08", "05-09", "06-12", "06-13",
           "11-03", "11-04", "12-31"],
    2026: ["01-01", "01-02", "01-03", "01-04", "01-05", "01-06", "01-07", "01-08", "01-09",
           "02-23", "03-09", "05-01", "05-11", "06-12", "11-04", "12-31"],
    2027: ["01-01", "01-02", "01-03", "01-04", "01-05", "01-06", "01-07", "01-08",
           "02-22", "02-23", "03-08", "05-03", "05-10", "06-14", "11-04", "11-05", "12-31"],
}
# Рабочие субботы (перенесённые рабочие дни)
RU_WORKING_WEEKENDS = ["2024-04-27", "2024-11-02", "2024-12-28", "2025-11-01", "2027-02-20"]


def build_calendar(start, end, paydays=PAYDAYS) -> pd.DataFrame:
    """Календарные признаки на каждый день диапазона (с запасом по краям)."""
    pad = pd.Timedelta(days=40)
    dates = pd.date_range(pd.Timestamp(start) - pad, pd.Timestamp(end) + pad, freq="D")
    missing = sorted(set(dates.year) - set(RU_NONWORKING))
    inside = [y for y in missing if pd.Timestamp(start).year <= y <= pd.Timestamp(end).year]
    if inside:
        warnings.warn(f"Нет производственного календаря на {inside}: праздники там не учтены, допишите RU_NONWORKING")

    nonwork = pd.to_datetime([f"{y}-{md}" for y, mds in RU_NONWORKING.items() for md in mds])
    working_weekends = pd.to_datetime(RU_WORKING_WEEKENDS)

    cal = pd.DataFrame({"date": dates})
    dow = cal["date"].dt.dayofweek
    is_nonwork = cal["date"].isin(nonwork)
    is_work = ((dow < 5) & ~is_nonwork) | cal["date"].isin(working_weekends)

    cal["dow"] = dow
    cal["day"] = cal["date"].dt.day
    cal["month"] = cal["date"].dt.month
    cal["dim"] = cal["date"].dt.days_in_month
    cal["days_to_eom"] = cal["dim"] - cal["day"]
    cal["is_workday"] = is_work.astype(int)
    cal["is_holiday"] = (is_nonwork & (dow < 5)).astype(int)  # будний день, но нерабочий

    ym = cal["date"].dt.to_period("M")
    cal["wd_num"] = cal.groupby(ym)["is_workday"].cumsum()
    cal["wd_total"] = cal.groupby(ym)["is_workday"].transform("sum")
    cal["wd_left"] = cal["wd_total"] - cal["wd_num"]
    cal["wd_frac"] = cal["wd_num"] / cal["wd_total"]

    # сколько нерабочих дней подряд было прямо перед днём / будет сразу после
    w = cal["is_workday"].to_numpy()
    n = len(w)
    before, after = np.zeros(n, int), np.zeros(n, int)
    run = 0
    for i in range(n):
        before[i] = run
        run = 0 if w[i] else run + 1
    run = 0
    for i in range(n - 1, -1, -1):
        after[i] = run
        run = 0 if w[i] else run + 1
    cal["nonwork_before"] = before
    cal["nonwork_after"] = after
    cal["first_wd_month"] = ((cal["is_workday"] == 1) & (cal["wd_num"] == 1)).astype(int)
    cal["last_wd_month"] = ((cal["is_workday"] == 1) & (cal["wd_left"] == 0)).astype(int)

    # новогодние каникулы: после последнего рабочего дня декабря и до первого рабочего дня января
    wd = cal.loc[cal["is_workday"] == 1, "date"]
    first_jan = wd[wd.dt.month == 1].groupby(wd.dt.year).min()
    last_dec = wd[wd.dt.month == 12].groupby(wd.dt.year).max()
    y = cal["date"].dt.year
    in_jan = (cal["month"] == 1) & (cal["date"] < y.map(first_jan))
    in_dec = (cal["month"] == 12) & (cal["date"] > y.map(last_dec))
    cal["ny_break"] = (in_jan | in_dec).astype(int)

    # дни зарплаты: если выпадают на нерабочий день — переносятся на предыдущий рабочий (ст. 136 ТК)
    wd_arr = wd.to_numpy()
    months = pd.period_range(dates.min(), dates.max(), freq="M")
    all_paydays = []
    for p in paydays:
        eff = []
        for m in months:
            target = np.datetime64(pd.Timestamp(m.year, m.month, min(p, m.days_in_month)))
            idx = np.searchsorted(wd_arr, target, side="right") - 1
            if idx >= 0:
                eff.append(wd_arr[idx])
        eff = pd.to_datetime(eff)
        cal[f"pay_{p}"] = cal["date"].isin(eff).astype(int)
        all_paydays.extend(eff)
    pays = np.sort(pd.to_datetime(all_paydays).unique().to_numpy())
    idx = np.searchsorted(pays, cal["date"].to_numpy(), side="right") - 1
    since = np.where(idx >= 0, (cal["date"].to_numpy() - pays[np.clip(idx, 0, None)]) / np.timedelta64(1, "D"), np.nan)
    cal["days_since_pay"] = since
    return cal



# ============================================================================
# ПОДГОТОВКА ДАННЫХ
# ============================================================================
def _cum_to_daily(s: pd.Series) -> pd.Series:
    last = s.last_valid_index()
    if last is None:
        return s
    f = s.loc[:last].ffill()
    return f.diff().fillna(f).reindex(s.index)


def target_months(cur):
    if TARGET_MONTHS:
        return sorted(pd.Period(m, "M") for m in TARGET_MONTHS)
    return [cur + i for i in range(HORIZON_MONTHS + 1)]


def prepare(df: pd.DataFrame, last_fact_date=None):
    """Дневная сетка до конца последнего прогнозного месяца + календарь.
    plan_actual / debt_actual — только известные значения (неизвестные дни пустые)."""
    cols = {DATE_COL: "date", FACT_COL: "fact", PLAN_COL: "plan"}
    if DEBT_COL:
        cols[DEBT_COL] = "debt"
    d = df[list(cols)].rename(columns=cols)
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    value_cols = [c for c in d.columns if c != "date"]
    d = d.groupby("date", as_index=False)[value_cols].sum(min_count=1)

    if last_fact_date is None:
        last_fact_date = d.loc[d["fact"].notna(), "date"].max()
    last_fact_date = pd.Timestamp(last_fact_date)
    cur = (last_fact_date + pd.Timedelta(days=1)).to_period("M")      # первый месяц без полного факта
    targets = target_months(cur)
    start = d["date"].min()
    end = max(d["date"].max(), max(targets).end_time.normalize(), (cur + 1).end_time.normalize())

    d = pd.DataFrame({"date": pd.date_range(start, end, freq="D")}).merge(d, on="date", how="left")
    d["ym"] = d["date"].dt.to_period("M")
    if CUMULATIVE_INPUT:  # накопленные с начала месяца -> дневные
        for c in ["fact", "plan"]:
            d[c] = d.groupby("ym")[c].transform(_cum_to_daily)
    kf = d["date"] <= last_fact_date
    d["fact"] = d["fact"].where(kf).fillna(0.0).where(kf).round(2)

    for c in value_cols[1:]:  # plan, debt
        last = d.loc[d[c].notna(), "date"].max()
        kn = d["date"] <= last
        filled = d[c].fillna(0.0) if c == "plan" else d[c].ffill()  # пропуск внутри известного: план 0, долг как вчера
        d[f"{c}_actual"] = filled.where(kn).round(2)
    # месяц с нулевым планом считаем незагруженным
    zero = d.groupby("ym")["plan_actual"].transform("sum") == 0
    d.loc[zero, "plan_actual"] = np.nan

    d = d.merge(build_calendar(start, end), on="date", how="left")
    return d, last_fact_date, cur, targets


# ============================================================================
# ПРОГНОЗ ПЛАНА И ДОЛГА (для дней и месяцев, где их ещё нет)
# ============================================================================
def full_months(frame: pd.DataFrame, col: str) -> pd.Series:
    """Итоги месяцев, где ряд известен на все дни месяца."""
    g = frame.groupby("ym")[f"{col}_actual"]
    n_known, n_rows = g.count(), g.size()
    dim = pd.Series([m.days_in_month for m in n_rows.index], index=n_rows.index)
    ok = (n_known == n_rows) & (n_rows == dim)
    return g.sum()[ok]


def trend_forecast(s: pd.Series, T, h: float) -> float:
    """Последнее значение × сглаженный темп роста (h = 0 — без роста; иначе вес темпа падает вдвое каждые h мес.)."""
    s = s.dropna()
    s = s[s > 0]
    if s.empty:
        return np.nan
    steps = (T - s.index[-1]).n
    g = 0.0
    if h != 0 and len(s) >= 2:
        gaps = np.array([(b - a).n for a, b in zip(s.index[:-1], s.index[1:])], float)
        lg = np.diff(np.log(s.to_numpy())) / gaps
        ages = np.arange(len(lg))[::-1]
        w = 0.5 ** (ages / h) if np.isfinite(h) else np.ones(len(lg))
        g = (w * lg).sum() / w.sum()
    return float(s.iloc[-1] * np.exp(g * steps))


def series_total_forecast(totals: pd.Series, T, cfg) -> float:
    """Итог месяца T по полным месяцам до T: последний × темп роста (за месяц или в пересчёте на день)."""
    hist = totals[totals.index < T]
    if hist.empty:
        return np.nan
    if cfg["unit"] == "день":
        per_day = hist / np.array([m.days_in_month for m in hist.index])
        return trend_forecast(per_day, T, cfg["h"]) * T.days_in_month
    return trend_forecast(hist, T, cfg["h"])


def series_profile(frame: pd.DataFrame, col: str, T, how: str, totals: pd.Series) -> np.ndarray:
    """Раскладка итога месяца T по дням (доли, сумма = 1) по последним PROFILE_MONTHS полным месяцам до T:
    по числу месяца или по номеру рабочего дня (все нерабочие дни — одна общая доля)."""
    rows = frame[frame["ym"] == T]
    hist = [m for m in totals.index if m < T][-PROFILE_MONTHS:]
    if not hist:
        return np.full(len(rows), 1.0 / len(rows))
    h = frame[frame["ym"].isin(hist)]
    share = (h[f"{col}_actual"] / h["ym"].map(totals).replace(0, np.nan)).to_numpy()
    if how == "по рабочим дням":
        key_h = np.where(h["is_workday"] == 1, h["wd_num"], 0)
        key_t = np.where(rows["is_workday"] == 1, rows["wd_num"], 0)
    else:
        key_h, key_t = h["day"].to_numpy(), rows["day"].to_numpy()
    mean_by_key = pd.Series(share).groupby(key_h).mean()
    prof = pd.Series(key_t).map(mean_by_key).to_numpy(dtype=float)
    fallback = np.nanmean(share) if np.isfinite(share).any() else 1.0
    prof = np.where(np.isnan(prof), fallback, prof)
    s = prof.sum()
    return prof / s if s > 0 else np.full(len(rows), 1.0 / len(rows))


def fill_series(frame: pd.DataFrame, col: str, cfg: dict, months=None) -> pd.Series:
    """Заполняет неизвестные дни ряда col (plan / debt). Известная часть месяца остаётся как есть;
    остальное = итог месяца × раскладка по дням. Итог месяца — по известной части (если она ≥ 25% месяца
    по раскладке), иначе по тренду прошлых месяцев."""
    s = frame[f"{col}_actual"].copy()
    totals = full_months(frame, col)
    todo = sorted(m for m in frame.loc[s.isna(), "ym"].unique() if months is None or m in months)
    for T in todo:
        idx = frame.index[frame["ym"] == T]
        known = s.loc[idx].notna().to_numpy()
        prof = series_profile(frame, col, T, cfg["profile"], totals)
        k_share = prof[known].sum()
        if known.any() and k_share >= 0.25:
            total = s.loc[idx][known].sum() / k_share
        else:
            total = series_total_forecast(totals, T, cfg)
        s.loc[idx[~known]] = total * prof[~known]
    return s


def tune_series(frame: pd.DataFrame, col: str) -> dict:
    """Подбор прогноза итога месяца (за месяц / в день, темп роста) и раскладки по дням — по прошлым месяцам."""
    totals = full_months(frame, col)
    test = [m for m in totals.index if m >= pd.Period(BACKTEST_FROM, "M") and (totals.index < m).any()]
    if not test:
        test = list(totals.index[1:])
    best = None
    for unit in ("месяц", "день"):
        for h in TREND_HALFLIVES:
            cfg = dict(unit=unit, h=h)
            err = np.array([series_total_forecast(totals, m, cfg) / totals[m] - 1 for m in test])
            score = np.nanmean(np.abs(err))
            if best is None or score < best["total_err"] - 1e-12:
                best = dict(cfg, total_err=score)
    prof_err = {}
    for how in ("по дням месяца", "по рабочим дням"):
        errs = []
        for m in test:
            a = frame.loc[frame["ym"] == m, f"{col}_actual"].to_numpy()
            p = series_profile(frame, col, m, how, totals)
            errs.append(np.abs(p * a.sum() - a).sum() / a.sum())
        prof_err[how] = np.mean(errs)
    best["profile"] = min(prof_err, key=prof_err.get)
    best["profile_err"] = prof_err[best["profile"]]
    return best


# ============================================================================
# ПРИЗНАКИ
# ============================================================================
def add_plan_features(f: pd.DataFrame) -> pd.DataFrame:
    """Признаки плана (доли месячного плана). План — известный или спрогнозированный."""
    f["plan_m"] = f.groupby("ym")["plan"].transform("sum")
    pm = f["plan_m"].replace(0, np.nan)
    f["plan_share"] = f["plan"] / pm
    f["plan_cum_share"] = f.groupby("ym")["plan"].cumsum() / pm
    f["plan_next3"] = sum(f["plan"].shift(-k) for k in (1, 2, 3)) / pm
    f["plan_prev3"] = sum(f["plan"].shift(k) for k in (1, 2, 3)) / pm
    # план, пришедшийся на нерабочие дни прямо перед этим днём (его часто платят в первый рабочий)
    acc, prev_nw = 0.0, np.zeros(len(f))
    for i, (p, wk) in enumerate(zip(np.nan_to_num(f["plan"].to_numpy()), f["is_workday"].to_numpy())):
        prev_nw[i] = acc
        acc = 0.0 if wk else acc + p
    f["plan_prev_nw"] = prev_nw / pm
    return f


def month_table(d: pd.DataFrame, last_fact_date) -> pd.DataFrame:
    g = d.groupby("ym")
    mt = pd.DataFrame(index=pd.period_range(d["ym"].min(), d["ym"].max(), freq="M"))
    mt["plan_m"] = g["plan"].sum()
    mt["plan_known"] = g["plan_actual"].count() == g.size()
    mt["fact_m"] = g["fact"].sum()
    whole = mt.index.start_time >= d["date"].min()
    mt["complete"] = whole & (mt.index.end_time.normalize() <= last_fact_date)
    ok = mt["complete"] & mt["plan_known"]
    mt["rate"] = np.where(ok, mt["fact_m"] / mt["plan_m"].replace(0, np.nan), np.nan)
    if "debt" in d:
        mt["debt_m"] = g["debt"].sum()
        mt["debt_last"] = g["debt"].last()
        mt["rr"] = np.where(mt["complete"], mt["fact_m"] / mt["debt_m"].replace(0, np.nan), np.nan)
    return mt


BASE_LEVEL = ["rate_prev1", "rate_prev3", "rate_ly", "plan_growth"]
DEBT_LEVEL = ["rr_prev1", "rr_prev3", "debt_cover"]
CAL_FEATURES = (["dow", "day", "month", "dim", "days_to_eom", "is_workday", "is_holiday",
                 "wd_num", "wd_total", "wd_left", "wd_frac", "nonwork_before", "nonwork_after",
                 "first_wd_month", "last_wd_month", "ny_break", "days_since_pay"]
                + [f"pay_{p}" for p in PAYDAYS])
PLAN_FEATURES = ["plan_share", "plan_cum_share", "plan_next3", "plan_prev3", "plan_prev_nw"]


def feature_list(use_debt: bool) -> list:
    return CAL_FEATURES + PLAN_FEATURES + BASE_LEVEL + (DEBT_LEVEL if use_debt else [])


def month_level_features(mt: pd.DataFrame, rate: pd.Series, rr: pd.Series = None, use_debt=False) -> pd.DataFrame:
    """Признаки уровня месяца, известные на его начало (по предыдущим месяцам)."""
    f = pd.DataFrame(index=mt.index)
    f["rate_prev1"] = rate.shift(1)
    f["rate_prev3"] = rate.shift(1).rolling(3, min_periods=1).mean()
    f["rate_ly"] = rate.shift(12)
    f["plan_growth"] = mt["plan_m"] / mt["plan_m"].shift(1).rolling(3, min_periods=1).mean() - 1
    if use_debt:
        rr = mt["rr"] if rr is None else rr
        dim = np.array([m.days_in_month for m in mt.index])
        f["rr_prev1"] = rr.shift(1)                                           # recovery прошлого месяца
        f["rr_prev3"] = rr.shift(1).rolling(3, min_periods=1).mean()
        f["debt_cover"] = (mt["debt_m"] / dim).shift(1) * dim / mt["plan_m"]  # долг прошлого месяца к плану этого
    return f


def attach_level(frame: pd.DataFrame, lvl: pd.DataFrame) -> pd.DataFrame:
    frame = frame.drop(columns=[c for c in BASE_LEVEL + DEBT_LEVEL if c in frame.columns])
    return frame.merge(lvl, left_on="ym", right_index=True, how="left")


# ============================================================================
# МОДЕЛИ
# ============================================================================
def make_model(seed=RANDOM_STATE):
    try:
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            n_estimators=700, learning_rate=0.02, num_leaves=15, min_child_samples=15,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=seed, verbose=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_iter=700, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=15,
            l2_regularization=1.0, max_features=0.8, early_stopping=False, random_state=seed)


class SeedBag:
    """Среднее нескольких бустингов с разными seed: на коротком ряду это заметно снижает разброс прогноза."""

    def __init__(self, n=None):
        self.models = [make_model(RANDOM_STATE + i) for i in range(n or N_SEEDS)]

    def fit(self, X, y, sample_weight=None):
        for m in self.models:
            m.fit(X, y, sample_weight=sample_weight)
        if hasattr(self.models[0], "feature_importances_"):
            self.feature_importances_ = np.mean([m.feature_importances_ for m in self.models], axis=0)
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)


def fit_model(train: pd.DataFrame, target_month, feats: list):
    tr = train[~train["date"].isin(pd.to_datetime(EXCLUDE_DATES))]
    w = None
    if RECENCY_HALFLIFE:
        age = (target_month - tr["ym"]).apply(lambda x: x.n).to_numpy()
        w = 0.5 ** (age / RECENCY_HALFLIFE)
    feats = [f for f in feats if tr[f].notna().any()]  # признак без единого значения (напр. rate_ly в первый год) — пропускаем
    model = SeedBag().fit(tr[feats], tr["y"], sample_weight=w)
    model.feats_ = feats
    return model


def predict_share(model, rows: pd.DataFrame) -> np.ndarray:
    """Прогноз доли месячного плана на каждый день."""
    return np.clip(model.predict(rows[model.feats_]), 0, None)


def reference_curve(d: pd.DataFrame, month, refs) -> np.ndarray:
    """Текущий подход: средняя кривая выполнения (накопл. факт / план месяца) по референсным месяцам,
    по календарному дню. Возвращает дневные доли плана (умножить на план месяца)."""
    n = month.days_in_month
    curves = []
    for r in refs:
        g = d[d["ym"] == r]
        if g.empty or g["fact"].isna().any() or not g["plan_m"].iloc[0] > 0:
            continue
        c = (g["fact"].cumsum() / g["plan_m"]).to_numpy()
        curves.append(np.concatenate([c, np.repeat(c[-1], max(0, n - len(c)))])[:n])
    if not curves:
        return np.full(n, np.nan)
    return np.diff(np.r_[0.0, np.mean(curves, axis=0)])


def plan_unknown_rows(d: pd.DataFrame, mt: pd.DataFrame, m, plan_cfg: dict, use_debt: bool) -> pd.DataFrame:
    """Строки месяца m такими, какими они видны, если план m (и следующих месяцев) ещё не загружен:
    план спрогнозирован только по месяцам до m."""
    f = d[["date", "ym", "day", "is_workday", "wd_num", "plan_actual"]].copy()
    f.loc[f["ym"] >= m, "plan_actual"] = np.nan
    f["plan"] = fill_series(f, "plan", plan_cfg, months={m, m + 1})
    f = add_plan_features(f)
    rows = d[d["ym"] == m].copy()
    fm = f[f["ym"] == m]
    for c in ["plan", "plan_m"] + PLAN_FEATURES:
        rows[c] = fm[c].to_numpy()
    mt2 = mt.copy()
    mt2.loc[m, "plan_m"] = rows["plan_m"].iloc[0]
    return attach_level(rows, month_level_features(mt2, mt2["rate"], use_debt=use_debt))


# ============================================================================
# NOWCAST: КОРРЕКЦИЯ ОСТАТКА ПО ХОДУ МЕСЯЦА
# ============================================================================
def nowcast_path(actual: np.ndarray, pred: np.ndarray, known: int, alpha: float) -> np.ndarray:
    """Накопленный прогноз на дни known..конец, если известны первые `known` дней факта.
    Остаток домножается на 1 + alpha*(факт/прогноз за известные дни - 1)."""
    ka = actual[:known].sum()
    kp = pred[:known].sum()
    ratio = ka / kp if known and kp > 0 else 1.0
    return ka + np.cumsum(pred[known:] * (1 + alpha * (ratio - 1)))


def eom_forecasts(actual: np.ndarray, pred: np.ndarray, alphas) -> np.ndarray:
    """Прогноз итога месяца [alpha, known_days] (векторно)."""
    ca = np.r_[0.0, np.cumsum(actual)][:-1]   # факт за первые k дней, k = 0..n-1
    cp = np.r_[0.0, np.cumsum(pred)][:-1]
    rem = pred.sum() - cp
    ratio = np.where(cp > 0, ca / np.where(cp > 0, cp, 1), 1.0)
    ratio[0] = 1.0
    a = np.asarray(alphas)[:, None]
    return ca + rem * (1 + a * (ratio - 1))


def eom_errors(actual: np.ndarray, pred: np.ndarray, alphas) -> np.ndarray:
    """Ошибка итога месяца [alpha, known_days] (векторно)."""
    return eom_forecasts(actual, pred, alphas) / actual.sum() - 1


def eom_eff_errors(actual: np.ndarray, pred: np.ndarray, plan_used: float, plan_true: float, alphas) -> np.ndarray:
    """Разница эффективности на конец месяца [alpha, known_days], п.п.:
    прогноз / план (загруженный или спрогнозированный, если план ещё не загружен) − факт / план."""
    return 100 * (eom_forecasts(actual, pred, alphas) / plan_used - actual.sum() / plan_true)


# ============================================================================
# УРОВЕНЬ ДЛЯ «БУСТИНГ × УРОВЕНЬ»
# ============================================================================
def level_estimate(L: pd.Series, T, h: float, w0: float) -> float:
    """Уровень выполнения на месяц T = взвешенное среднее уровней прошлых месяцев (вес падает вдвое каждые h мес.)
    и «нейтрального» уровня 1 с весом w0. Уровень месяца = факт / (план × ожидание модели без признаков уровня),
    то есть выполнение, очищенное от календаря: апрельская предоплата перед майскими его не сдвигает."""
    if np.isinf(w0):
        return 1.0
    hist = L[L.index < T].dropna()
    ages = np.array([(T - m).n for m in hist.index], float)
    w = 0.5 ** (ages / h) if np.isfinite(h) else np.ones(len(ages))
    den = w0 + w.sum()
    return float((w0 + (w * hist.to_numpy()).sum()) / den) if den > 0 else 1.0


def month_levels(bt: pd.DataFrame) -> pd.Series:
    g = bt.groupby("ym").agg(F=("fact", "sum"), S=("share_cal", "sum"), P=("plan_m", "first"))
    return g["F"] / (g["S"] * g["P"])


def tune_level(bt: pd.DataFrame):
    """Подбор сглаживания уровня по ошибке итога месяца без факта месяца (известно 0 дней)."""
    L = month_levels(bt)
    g = bt.groupby("ym").agg(F=("fact", "sum"), S=("share_cal", "sum"), P=("plan_m", "first"))
    best = None
    for h in LEVEL_HALFLIVES:
        for w0 in LEVEL_PRIOR_WEIGHTS:
            lh = np.array([level_estimate(L, m, h, w0) for m in g.index])
            score = np.mean(np.abs(lh * g["S"] * g["P"] / g["F"] - 1))
            if best is None or score < best["score"] - 1e-12:
                best = dict(h=h, w0=w0, score=score)
    return best, L


# ============================================================================
# БЭКТЕСТ
# ============================================================================
SCENARIOS = {"": "план загружен", "_pu": "план ещё не загружен"}
M_REF, M_BOOST, M_CAL, M_MIX = "Референсные месяцы", "Бустинг", "Бустинг × уровень", "Смесь"
MIX_GRID = [(a, b, round(1 - a - b, 2)) for a in np.arange(0, 1.01, 0.25) for b in np.arange(0, 1.01 - a, 0.25)]


def backtest(d: pd.DataFrame, mt: pd.DataFrame, plan_cfg: dict, debt_cfg: dict = None):
    """Для каждого месяца с BACKTEST_FROM: модели обучены только на месяцах до него.
    Бустинг (как в первой версии, с признаками уровня), он же с признаками долга (если есть долг)
    и бустинг без признаков уровня (для «бустинг × уровень»). Каждый — с загруженным и со спрогнозированным планом."""
    has_debt = "debt" in d
    feats = {"": feature_list(False), "_debt": feature_list(True)} if has_debt else {"": feature_list(False)}
    dd = attach_level(d, month_level_features(mt, mt["rate"], use_debt=has_debt))
    ok = mt.index[mt["complete"] & mt["plan_known"]]
    test_months = [m for m in ok if m >= pd.Period(BACKTEST_FROM, "M")]
    if not test_months:
        raise ValueError("Нет полных месяцев для бэктеста — сдвиньте BACKTEST_FROM")
    if has_debt:
        debt_totals = full_months(d, "debt")

    parts = []
    for m in test_months:
        train = dd[(dd["ym"] < m) & dd["ym"].isin(ok) & dd["y"].notna()]
        te = dd[dd["ym"] == m].copy()
        pu = plan_unknown_rows(d, mt, m, plan_cfg, has_debt)    # тот же месяц, если его план ещё не загружен
        plan_m, plan_hat = te["plan_m"].iloc[0], pu["plan_m"].iloc[0]
        te["plan_m_hat"] = plan_hat
        te["plan_hat"] = pu["plan"].to_numpy()
        for tag, fl in feats.items():
            model = fit_model(train, m, fl)
            te[f"pred_model{tag}"] = predict_share(model, te) * plan_m
            te[f"pred_model{tag}_pu"] = predict_share(model, pu) * plan_hat
        cal = fit_model(train, m, CAL_FEATURES + PLAN_FEATURES)
        te["share_cal"], te["share_cal_pu"] = predict_share(cal, te), predict_share(cal, pu)
        ref = reference_curve(dd, m, [m - i for i in range(1, N_REF_MONTHS + 1)])
        te["pred_ref"], te["pred_ref_pu"] = ref * plan_m, ref * plan_hat
        if has_debt:  # для recovery rate: раскладка и тренд долга по месяцам до m
            te["debt_prof"] = series_profile(d, "debt", m, debt_cfg["profile"], debt_totals)
            te["debt_trend"] = series_total_forecast(debt_totals, m, debt_cfg)
        parts.append(te)
        print(f"  бэктест {m}: обучено на {train['ym'].nunique()} мес.")
    return pd.concat(parts, ignore_index=True)


def add_level_method(bt: pd.DataFrame, lvl_cfg: dict, L: pd.Series) -> pd.DataFrame:
    lh = {m: level_estimate(L, m, lvl_cfg["h"], lvl_cfg["w0"]) for m in bt["ym"].unique()}
    bt["level_hat"] = bt["ym"].map(lh)
    bt["pred_cal"] = bt["share_cal"] * bt["plan_m"] * bt["level_hat"]
    bt["pred_cal_pu"] = bt["share_cal_pu"] * bt["plan_m_hat"] * bt["level_hat"]
    return bt


def _plan_used(g, sfx):
    """План по дням, к которому считается эффективность прогноза: загруженный или спрогнозированный."""
    return g["plan_hat"].to_numpy() if sfx == "_pu" else g["plan"].to_numpy()


def _blend(g, w, sfx):
    """w = (вес бустинга, вес «бустинг × уровень», вес референсных месяцев)."""
    return (w[0] * g[f"pred_model{sfx}"].to_numpy() + w[1] * g[f"pred_cal{sfx}"].to_numpy()
            + w[2] * np.nan_to_num(g[f"pred_ref{sfx}"].to_numpy()))


def tune(bt: pd.DataFrame, sfx: str = ""):
    """Подбор весов и силы коррекции по средней |разнице эффективности на конец месяца| (п.п.) по всем дням месяца."""
    results = {}
    candidates = {M_REF: [(0, 0, 1)], M_BOOST: [(1, 0, 0)], M_CAL: [(0, 1, 0)],
                  M_MIX: [w for w in MIX_GRID if max(w) < 1]}
    for name, weights in candidates.items():
        best = None
        for w in weights:
            errs = [np.abs(eom_eff_errors(g["fact"].to_numpy(), _blend(g, w, sfx), _plan_used(g, sfx).sum(),
                                          g["plan"].sum(), ALPHAS)).mean(axis=1)
                    for _, g in bt.groupby("ym")]
            score = np.mean(errs, axis=0)  # по alpha
            j = int(np.argmin(score))
            if best is None or score[j] < best["score"]:
                best = dict(w=w, alpha=float(ALPHAS[j]), score=float(score[j]))
        results[name] = best
    return results


def evaluate(bt: pd.DataFrame, w, alpha: float, sfx: str = "") -> pd.DataFrame:
    """Для каждого месяца и числа известных дней k: разница эффективности прогноза и факта, п.п. —
    на конец месяца и в среднем по датам после k (накопленный прогноз / накопленный план против
    накопленного факта / накопленного плана)."""
    rows = []
    for m, g in bt.groupby("ym"):
        a = g["fact"].to_numpy()
        p = _blend(g, w, sfx)
        cp_true, cp_used = np.cumsum(g["plan"].to_numpy()), np.cumsum(_plan_used(g, sfx))
        ca = np.cumsum(a)
        with np.errstate(divide="ignore", invalid="ignore"):
            eff_fact = ca / cp_true
        for k in range(len(a)):
            path = nowcast_path(a, p, k, alpha)
            with np.errstate(divide="ignore", invalid="ignore"):
                diff = 100 * (path / cp_used[k:] - eff_fact[k:])
            diff = diff[np.isfinite(diff)]
            mask = ca[k:] > 0
            rows.append(dict(
                ym=str(m), known_days=k,
                eff_fact=100 * ca[-1] / cp_true[-1],                              # эффективность факта на конец месяца, %
                err_eff_pp=100 * path[-1] / cp_used[-1] - 100 * ca[-1] / cp_true[-1],   # разница на конец месяца, п.п.
                eff_dates_pp=np.mean(np.abs(diff)) if len(diff) else np.nan,      # ср. |разница| на датах после k, п.п.
                eff_dates_within1=np.mean(np.abs(diff) <= 1) if len(diff) else np.nan,
                err_eom=path[-1] / ca[-1] - 1,                                    # относительная ошибка суммы
                err_path=np.mean(np.abs(path[mask] / ca[k:][mask] - 1)),
                err_daily_wape=np.abs(p[k:] - a[k:]).sum() / max(a[k:].sum(), 1e-9)))
    return pd.DataFrame(rows)


def debt_forecast_total(debt, prof, trend, k):
    """Долг месяца (Σ за месяц или на последний день) при известных первых k днях — как в fill_series."""
    known_sum, k_share = debt[:k].sum(), prof[:k].sum()
    total = known_sum / k_share if k and k_share >= 0.25 else trend
    rest = total * prof[k:]
    if RR_DENOM == "last":
        return rest[-1] if len(rest) else debt[-1]
    return known_sum + rest.sum()


def rr_evaluate(bt: pd.DataFrame, w, alpha: float, sfx: str = "") -> pd.DataFrame:
    """Ошибка recovery rate месяца: прогноз факта / прогноз долга против факта / долга."""
    rows = []
    for m, g in bt.groupby("ym"):
        a, debt = g["fact"].to_numpy(), g["debt"].to_numpy()
        prof, trend = g["debt_prof"].to_numpy(), g["debt_trend"].iloc[0]
        f_hat = eom_forecasts(a, _blend(g, w, sfx), [alpha])[0]
        d_act = debt[-1] if RR_DENOM == "last" else debt.sum()
        rr_act = a.sum() / d_act
        for k in range(len(a)):
            rr_hat = f_hat[k] / debt_forecast_total(debt, prof, trend, k)
            rows.append(dict(ym=str(m), known_days=k, rr_fact=rr_act, rr_forecast=rr_hat, err_rr=rr_hat / rr_act - 1))
    return pd.DataFrame(rows)


# ============================================================================
# ПОЛНЫЙ ПРОГНОЗ НА МЕСЯЦЫ (по дням)
# ============================================================================
def production(d, mt, cur, targets, use_debt, best, evals, rr_evals, winner, L, lvl_cfg):
    feats = feature_list(use_debt)
    ok = mt.index[mt["complete"] & mt["plan_known"]]
    rate = mt["rate"].copy()
    rr = mt["rr"].copy() if "rr" in mt else None
    L = L.copy()
    series = d["fact"].copy()  # факт; по мере прогноза дописываем прогноз (для recovery с начала данных)
    months = sorted(set(t for t in targets if t < cur) |
                    (set(pd.period_range(cur, max(targets), freq="M")) if max(targets) >= cur else set()))
    models, out, summary = {}, [], []

    for T in months:
        train_until = T if T < cur else cur   # прошедший месяц — модели, обученные только на месяцах до него
        dd = attach_level(d, month_level_features(mt, rate, rr, use_debt))
        if train_until not in models:
            train = dd[(dd["ym"] < train_until) & dd["ym"].isin(ok) & dd["y"].notna()]
            models[train_until] = (fit_model(train, train_until, feats),
                                   fit_model(train, train_until, CAL_FEATURES + PLAN_FEATURES))
        model, cal = models[train_until]
        rows = dd[dd["ym"] == T].copy()
        sc = "" if mt.loc[T, "plan_known"] else "_pu"
        cfg = best[sc][winner]
        w = cfg["w"]
        refs = [m for m in ok if m < min(T, cur)][-N_REF_MONTHS:]
        plan_m = rows["plan_m"].iloc[0]
        share_cal = predict_share(cal, rows)
        level_hat = level_estimate(L, T, lvl_cfg["h"], lvl_cfg["w0"])
        p = plan_m * (w[0] * predict_share(model, rows) + w[1] * share_cal * level_hat
                      + w[2] * np.nan_to_num(reference_curve(d, T, refs)))   # прогноз без факта месяца
        a = rows["fact"].to_numpy()
        k = int(np.sum(~np.isnan(a)))
        cum = np.r_[np.cumsum(a[:k]), nowcast_path(np.nan_to_num(a), p, k, cfg["alpha"])]
        fc = np.diff(np.r_[0.0, cum])
        series.loc[(d["ym"] == T).to_numpy()] = fc
        if T >= cur:   # уровень следующих месяцев — с учётом прогноза этого
            rate.loc[T] = cum[-1] / plan_m
            L.loc[T] = cum[-1] / (share_cal.sum() * plan_m)
            if rr is not None:
                rr.loc[T] = cum[-1] / mt.loc[T, "debt_m"]

        has_fact = np.arange(len(a)) < k
        cum_plan, cum_p = np.cumsum(rows["plan"].to_numpy()), np.cumsum(p)
        out_rows = pd.DataFrame({
            "month": str(T), "date": rows["date"].to_numpy(), "is_workday": rows["is_workday"].to_numpy(),
            "has_fact": has_fact,                                   # True — по этому дню факт уже есть
            "plan": rows["plan"].to_numpy(), "plan_is_forecast": rows["plan_actual"].isna().to_numpy(),
            "fact": a,
            "forecast": p,                  # прогноз на КАЖДЫЙ день месяца, без единого факта этого месяца
            "fact_plus_forecast": fc,       # факт, где он есть, + прогноз остатка с поправкой по факту на сегодня
            "cum_plan": cum_plan,
            "cum_fact": np.where(has_fact, np.cumsum(np.nan_to_num(a)), np.nan),
            "cum_forecast": cum_p,
            "cum_fact_plus_forecast": cum})
        cp = out_rows["cum_plan"].replace(0, np.nan)
        out_rows["eff_fact_pct"] = 100 * out_rows["cum_fact"] / cp                      # эффективность факта на дату
        out_rows["eff_forecast_pct"] = 100 * out_rows["cum_forecast"] / cp              # эффективность прогноза на дату
        out_rows["eff_fact_plus_forecast_pct"] = 100 * out_rows["cum_fact_plus_forecast"] / cp  # куда придёт месяц
        out_rows["eff_diff_pp"] = out_rows["eff_forecast_pct"] - out_rows["eff_fact_pct"]       # проверка по факту, п.п.
        out_rows["forecast_err_pct"] = 100 * (out_rows["cum_forecast"] / out_rows["cum_fact"] - 1)

        eff, eff_pure = 100 * cum[-1] / plan_m, 100 * cum_p[-1] / plan_m
        e = evals[sc][winner]
        e = e.loc[e["known_days"] == min(k, e["known_days"].max()), "err_eff_pp"] if k < len(a) else pd.Series(dtype=float)
        lo, hi = (eff - e.quantile(0.9), eff - e.quantile(0.1)) if len(e) >= 5 else (np.nan, np.nan)
        s = {"месяц": str(T), "дней факта": f"{k} из {len(a)}",
             "план": f"{plan_m:,.0f}".replace(",", " ") + (" (прогноз)" if sc else ""),
             "эфф.: прогноз без факта": f"{eff_pure:.1f}%",
             "эфф.: факт + прогноз остатка": f"{eff:.1f}%",
             "80% интервал": (f"{lo:.1f}% – {hi:.1f}%" if np.isfinite(lo) else "—"),
             "сумма: факт + прогноз остатка": f"{cum[-1]:,.0f}".replace(",", " "),
             "разница прогноза с фактом на сегодня": (f"{out_rows['eff_diff_pp'].iloc[k - 1]:+.2f} п.п." if k else "—")}

        if "debt" in d:
            debt = rows["debt"].to_numpy()
            hist = (d["date"] < rows["date"].iloc[0]).to_numpy()
            f_before, d_before = series[hist].sum(), d.loc[hist, "debt"].sum()
            cum_debt = np.cumsum(debt) if RR_DENOM == "sum" else debt
            tot_debt = d_before + np.cumsum(debt) if RR_DENOM == "sum" else debt
            out_rows["debt"] = debt
            out_rows["debt_is_forecast"] = rows["debt_actual"].isna().to_numpy()
            out_rows["rr_day_fact_pct"] = 100 * a / debt                                   # за день: факт / долг
            out_rows["rr_day_forecast_pct"] = 100 * p / debt                               # за день: прогноз / долг
            out_rows["rr_fact_pct"] = 100 * out_rows["cum_fact"] / cum_debt                # с начала месяца
            out_rows["rr_forecast_pct"] = 100 * cum_p / cum_debt
            out_rows["rr_fact_plus_forecast_pct"] = 100 * cum / cum_debt
            out_rows["rr_total_pct"] = 100 * (f_before + cum) / tot_debt                  # с начала данных
            rr_m, rr_pure = cum[-1] / cum_debt[-1], cum_p[-1] / cum_debt[-1]
            note = " (долг — прогноз)" if out_rows["debt_is_forecast"].any() else ""
            s["recovery: прогноз без факта"] = f"{100 * rr_pure:.3f}%"
            s["recovery: факт + прогноз остатка"] = f"{100 * rr_m:.3f}%" + note
            re = rr_evals[sc][winner]
            re = re.loc[re["known_days"] == min(k, re["known_days"].max()), "err_rr"] if k < len(a) else pd.Series(dtype=float)
            s["80% интервал recovery"] = "—"
            if len(re) >= 5:
                s["80% интервал recovery"] = f"{100 * rr_m / (1 + re.quantile(0.9)):.3f}% – {100 * rr_m / (1 + re.quantile(0.1)):.3f}%"
        if T in targets:
            out.append(out_rows)
            summary.append(s)
    return pd.concat(out, ignore_index=True), pd.DataFrame(summary), models.get(cur, (None,))[0]


# ============================================================================
# ГРАФИКИ
# ============================================================================
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_backtest(curves: dict, path: str):
    """curves: {подпись: (Series ошибки по known_days, цвет, стиль линии)}"""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=SURFACE)
    _style(ax)
    for name, (s, color, ls) in curves.items():
        ax.plot(s.index, 100 * s.values, color=color, linewidth=2, linestyle=ls, label=name)
    ax.axhline(1, color=INK2, linewidth=1, linestyle=(0, (1, 2)))
    ax.annotate("цель 1 п.п.", (0, 1), xytext=(2, 4), textcoords="offset points", color=INK2, fontsize=9)
    ax.set_xlabel("Сколько дней месяца уже известно", color=INK2)
    ax.set_ylabel("Средняя |разница эффективности|, п.п.", color=INK2)
    ax.set_title("Бэктест: эффективность прогноза против эффективности факта на конец месяца",
                 color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_month(g: pd.DataFrame, path: str):
    """Эффективность на дату (и recovery rate, если есть долг): факт; прогноз на весь месяц без факта
    с коридором ±1 п.п. (цель); куда придёт месяц с учётом факта по сегодня."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    panels = [("eff", "Эффективность на дату, %", True)]
    if "rr_fact_pct" in g:
        panels.append(("rr", "Recovery rate с начала месяца, %", False))
    cols = {"eff": ("eff_fact_pct", "eff_forecast_pct", "eff_fact_plus_forecast_pct"),
            "rr": ("rr_fact_pct", "rr_forecast_pct", "rr_fact_plus_forecast_pct")}
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 4.5), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    known, rest = g[g["has_fact"]], g[~g["has_fact"]]
    for ax, (key, title, band) in zip(axes, panels):
        c_fact, c_fc, c_fpf = cols[key]
        _style(ax)
        if band:
            ax.fill_between(g["date"], g[c_fc] - 1, g[c_fc] + 1, color=BLUE, alpha=0.15, linewidth=0,
                            label="±1 п.п. от прогноза (цель)")
        ax.plot(g["date"], g[c_fc], color=BLUE, linewidth=2, linestyle=(0, (4, 3)), label="Прогноз (без факта месяца)")
        if len(known):
            ax.plot(known["date"], known[c_fact], color=INK, linewidth=2, label="Факт")
        if len(rest):
            r = pd.concat([known.tail(1), rest]) if len(known) else rest
            ax.plot(r["date"], r[c_fpf], color=ORANGE, linewidth=2, linestyle=(0, (1, 1.5)),
                    label="Факт + прогноз остатка")
        if key == "eff":
            ax.axhline(100, color=INK2, linewidth=1, linestyle=(0, (1, 2)))
        ax.set_xlim(g["date"].min() - pd.Timedelta(hours=12), g["date"].max() + pd.Timedelta(hours=12))
        ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 8, 15, 22, 29]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
        ax.set_title(f"{g['month'].iloc[0]}: {title}", color=INK, loc="left", fontsize=11)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ============================================================================
# ЗАПУСК
# ============================================================================
def _summary_table(best, evals):
    def w_str(w):
        return " / ".join(f"{x:g}" for x in w)

    def pp(x):
        return f"{x:.2f} п.п."

    rows = {}
    for name, cfg in best.items():
        e = evals[name]
        by_k = e.groupby("known_days")["err_eff_pp"]
        within = by_k.apply(lambda s: (s.abs() <= 1).mean())
        e0 = e[e["known_days"] == 0]
        rows[name] = {
            "веса: бустинг / ×уровень / реф.": w_str(cfg["w"]), "alpha": cfg["alpha"],
            **{f"конец мес., известно {k} дн.": pp(by_k.apply(lambda s: s.abs().mean()).get(k, np.nan))
               for k in REPORT_KNOWN_DAYS},
            "месяцев ≤ 1 п.п. (известно 0 / 10 / 20 дн.)": " / ".join(f"{100 * within.get(k, np.nan):.0f}%" for k in (0, 10, 20)),
            "на даты, прогноз с 1-го: ср.|разница|": pp(e0["eff_dates_pp"].mean()),
            "на даты, прогноз с 1-го: дат ≤ 1 п.п.": f"{100 * e0['eff_dates_within1'].mean():.0f}%",
            "дневной WAPE": f"{100 * e0['err_daily_wape'].mean():.1f}%",
        }
    return pd.DataFrame(rows)


def _reach(e: pd.DataFrame, share=0.8):
    """С какого числа известных дней разница эффективности на конец месяца ≤ 1 п.п. в доле share месяцев."""
    within = e.groupby("known_days")["err_eff_pp"].apply(lambda s: (s.abs() <= 1).mean())
    ok = within[within >= share]
    return int(ok.index.min()) if len(ok) else None


def _hl(h):
    return "без роста" if h == 0 else ("среднее за всю историю" if np.isinf(h) else "полураспад %g мес." % h)


def run(df: pd.DataFrame, last_fact_date=LAST_FACT_DATE, make_plots=True):
    d, last_fact_date, cur, targets = prepare(df, last_fact_date)
    has_debt = "debt" in d
    print(f"Данные: {d['date'].min():%d.%m.%Y} – {last_fact_date:%d.%m.%Y} (факт); "
          f"план загружен по {full_months(d, 'plan').index.max()}"
          + (f"; долг известен по {d.loc[d['debt_actual'].notna(), 'date'].max():%d.%m.%Y}" if has_debt else ""))

    # прогноз плана (и долга) там, где их ещё нет
    plan_cfg = tune_series(d, "plan")
    d["plan"] = fill_series(d, "plan", plan_cfg)
    print(f"Прогноз плана, когда он не загружен: последний месяц ({plan_cfg['unit']}) × темп роста ({_hl(plan_cfg['h'])}), "
          f"раскладка {plan_cfg['profile']}. На прошлых месяцах ошибка итога плана {100 * plan_cfg['total_err']:.2f}%, "
          f"раскладки по дням {100 * plan_cfg['profile_err']:.1f}% (WAPE).")
    debt_cfg = None
    if has_debt:
        debt_cfg = tune_series(d, "debt")
        d["debt"] = fill_series(d, "debt", debt_cfg)
        print(f"Прогноз долга на будущие дни: на прошлых месяцах ошибка итога {100 * debt_cfg['total_err']:.2f}%.")
    d = add_plan_features(d)
    mt = month_table(d, last_fact_date)
    d["y"] = np.where(d["ym"].map(mt["plan_known"]).astype(bool), d["fact"] / d["plan_m"].replace(0, np.nan), np.nan)

    print("Бэктест...")
    bt = backtest(d, mt, plan_cfg, debt_cfg)
    lvl_cfg, L = tune_level(bt)
    bt = add_level_method(bt, lvl_cfg, L)
    print(f"\nУровень для «{M_CAL}»: " + ("не подстраивается" if np.isinf(lvl_cfg["w0"]) else
          f"сглаживание ({_hl(lvl_cfg['h'])}), тяга к среднему {lvl_cfg['w0']:g}"))

    use_debt = False
    if has_debt:  # признаки долга в бустинге — только если они уменьшают ошибку
        s0 = tune(bt)[M_BOOST]["score"]
        bt_debt = bt.copy()
        bt_debt["pred_model"], bt_debt["pred_model_pu"] = bt["pred_model_debt"], bt["pred_model_debt_pu"]
        s1 = tune(bt_debt)[M_BOOST]["score"]
        use_debt = s1 < s0
        print(f"Признаки долга в бустинге: средняя разница эффективности без них {s0:.2f} п.п., с ними {s1:.2f} п.п. "
              f"→ {'оставляем' if use_debt else 'не используем'}.")
        if use_debt:
            bt = bt_debt

    best = {sfx: tune(bt, sfx) for sfx in SCENARIOS}
    evals = {sfx: {n: evaluate(bt, c["w"], c["alpha"], sfx) for n, c in best[sfx].items()} for sfx in SCENARIOS}
    print("\nЭффективность = факт / план и прогноз / план. Ниже — разница между ними в процентных пунктах (п.п.).")
    for sfx, label in SCENARIOS.items():
        print(f"\nБэктест, {label}:")
        print(_summary_table(best[sfx], evals[sfx]).to_string())

    winner = min(best[""], key=lambda n: best[""][n]["score"])
    for sfx, label in SCENARIOS.items():
        r = _reach(evals[sfx][winner])
        print(f"Цель «≤ 1 п.п. на конец месяца в 80% месяцев», {label}: "
              + (f"достигается, когда известно {r} дн. месяца." if r is not None else "не достигается."))
    ev = evals[""][winner]
    by_month = (ev[ev["known_days"].isin(REPORT_KNOWN_DAYS)]
                .pivot(index="ym", columns="known_days", values="err_eff_pp").round(2))
    by_month.insert(0, "эфф. факт, %", ev.groupby("ym")["eff_fact"].first().round(1))
    pe = bt.groupby("ym").agg(hat=("plan_m_hat", "first"), act=("plan_m", "first"))
    by_month["ошибка прогноза плана, %"] = (100 * (pe["hat"] / pe["act"] - 1)).rename(index=str).round(2)
    print(f"\nЛучший: {winner}. Разница эффективности на конец месяца по месяцам, план загружен, п.п. "
          f"(столбцы — сколько дней известно; + значит прогноз выше факта). Последний столбец — "
          f"насколько ошибся бы прогноз плана, если бы план не был загружен:")
    print(by_month.to_string())

    rr_evals = {}
    if has_debt:
        rr_evals = {sfx: {n: rr_evaluate(bt, c["w"], c["alpha"], sfx) for n, c in best[sfx].items()} for sfx in SCENARIOS}
        tab = {}
        for sfx, label in SCENARIOS.items():
            r = rr_evals[sfx][winner]
            tab[label] = {f"известно {k} дн.": f"{100 * r.loc[r['known_days'] == k, 'err_rr'].abs().mean():.2f}%"
                          for k in REPORT_KNOWN_DAYS}
        print(f"\nRecovery rate месяца ({winner}): средняя |относительная ошибка| (долг на неизвестные дни — прогноз):")
        print(pd.DataFrame(tab).T.to_string())

    fc, fsum, model = production(d, mt, cur, targets, use_debt, best, evals, rr_evals, winner, L, lvl_cfg)
    print("\nПрогноз:")
    print(fsum.to_string(index=False))

    if model is not None and hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=model.feats_).sort_values(ascending=False)
        print("\nВажность признаков бустинга (топ-12):")
        print(imp.head(12).round(1).to_string())

    for m, g in fc.groupby("month"):
        g.to_csv(f"{OUT_DIR}/forecast_{m}.csv", index=False, encoding="utf-8-sig")
        if make_plots:
            plot_month(g, f"{OUT_DIR}/forecast_{m}.png")
    ev.to_csv(f"{OUT_DIR}/backtest_errors.csv", index=False)
    if make_plots:
        def curve(sfx, name):
            return evals[sfx][name].groupby("known_days")["err_eff_pp"].apply(lambda s: s.abs().mean() / 100)
        curves = {f"{winner}, план загружен": (curve("", winner), BLUE, "-"),
                  f"{winner}, план ещё не загружен": (curve("_pu", winner), BLUE, (0, (4, 3)))}
        if winner != M_BOOST:
            curves[f"{M_BOOST} (первая версия), план загружен"] = (curve("", M_BOOST), AQUA, "-")
        curves[f"{M_REF}, план загружен"] = (curve("", M_REF), ORANGE, "-")
        plot_backtest(curves, f"{OUT_DIR}/backtest_error_by_day.png")
    return dict(daily=d, months=mt, backtest=bt, best=best, evals=evals, rr_evals=rr_evals, winner=winner,
                use_debt=use_debt, plan_cfg=plan_cfg, debt_cfg=debt_cfg, level_cfg=lvl_cfg, level=L,
                by_month=by_month, forecast=fc, forecast_summary=fsum, model=model)


def load_data() -> pd.DataFrame:
    """Замените на свой запрос. Нужны колонки DATE_COL, FACT_COL, PLAN_COL (и DEBT_COL, если задан)."""
    # import sqlalchemy as sa
    # engine = sa.create_engine("...")
    # return pd.read_sql("select pay_date as date, fact, plan, debt from ...", engine)
    raise NotImplementedError("Заполните load_data() своим запросом к базе")


if __name__ == "__main__":
    res = run(load_data())
