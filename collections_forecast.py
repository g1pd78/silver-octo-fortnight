"""
Прогноз сборов на следующий месяц (без его плана и факта) и проверка прогноза по факту на каждую дату.

ПОСТАНОВКА
  Прогноз на месяц T делается до его начала. В момент прогноза известны факт и план по прошлым месяцам
  и по текущему месяцу (факт — по сегодня). План и факт месяца T НЕ известны и нигде не используются.

ИДЕЯ: факт месяца = план × уровень выполнения × календарь
  План T      — прогноз по истории планов (последний известный план × сглаженный темп роста).
  Календарь   — дневная модель (градиентный бустинг) по производственному календарю: какую долю плана обычно
                собирают в каждый день при таком раскладе выходных, праздников, каникул и дней зарплат.
                Сумма по месяцу S — «календарное» выполнение.
  Уровень L   — платёжная дисциплина: факт / (план × S). Меняется медленно, прогнозируется сглаживанием
                прошлых месяцев и неполного текущего.
  Для сравнения: «тренд факта × календарь» (без плана) и ваши референсные месяцы × прогноз плана.

ПРОВЕРКА ПО ФАКТУ
  Прогноз фиксируется в момент прогноза и сравнивается с фактом на каждую дату месяца (накопленный факт против
  накопленного прогноза) и с итогом месяца. В бэктесте каждая модель обучена только на том, что было известно
  в момент прогноза. Ошибка = прогноз / факт − 1 («+» — прогноз выше факта).

ЧТО ПЕЧАТАЕТ
  [1] Ошибку итога месяца в зависимости от того, за сколько дней до начала месяца сделан прогноз.
  [2] Ошибку накопленного факта на контрольные даты для прогноза, сделанного за FORECAST_ORIGIN дней.
  [3] По месяцам: сколько ошибки дал прогноз плана, а сколько — уровень.
  [4] Уровень по месяцам и его скачки — примерный предел точности по трём колонкам.
  [5] Дни с аномальным фактом (кандидаты в EXCLUDE_DATES).
  Текущий месяц: прогноз, сделанный в прошлом месяце, против факта на сегодня. Следующий месяц: прогноз с интервалом.

Вход: DataFrame с колонками дата / факт / план — суммы ЗА ДЕНЬ (если накопленные — CUMULATIVE_INPUT = True).
Будущие даты не нужны.

Зависимости: pandas, numpy, scikit-learn, matplotlib. lightgbm — по желанию, используется, если установлен.
Запуск: заполните load_data() и выполните файл, или в Jupyter:  from collections_forecast import run;  res = run(df)
Настройки — в блоке НАСТРОЙКИ ниже. Прогон ~1–2 минуты (с lightgbm быстрее).
"""

import itertools
import warnings

import numpy as np
import pandas as pd

# ============================================================================
# НАСТРОЙКИ
# ============================================================================
DATE_COL, FACT_COL, PLAN_COL = "date", "fact", "plan"
CUMULATIVE_INPUT = False       # True, если факт и план в выгрузке накопленные с начала месяца
LAST_FACT_DATE = None          # последняя дата с ПОЛНЫМ фактом. None = последняя дата, где факт не пустой.
                               # Если в выгрузке есть неполный «сегодня», укажите вчерашнюю дату.
HORIZON_MONTHS = 1             # сколько месяцев вперёд прогнозировать (1 = следующий месяц)
FORECAST_ORIGIN = -5           # когда вы обычно делаете прогноз: за сколько дней до начала месяца
                               # (-5 = по факту на 26-е при 31 дне; 0 = по итогам прошлого месяца)
PAYDAYS = (5, 10, 15, 20, 25)  # типичные дни зарплаты/аванса клиентов; выходной -> предыдущий рабочий день
N_REF_MONTHS = 3               # сколько предыдущих месяцев брать в методе референсных месяцев
MIN_TRAIN_MONTHS = 12          # минимум полных месяцев для обучения первой модели бэктеста
RECENCY_HALFLIFE = 12          # вес наблюдения в дневной модели падает вдвое каждые N месяцев (None — без весов)
EXCLUDE_DATES = []             # разовые аномалии (продажа портфеля, крупная досрочка): не учить на них модель
                               # и не сдвигать по ним уровень, например ["2025-03-14"]
ORIGINS_REPORT = (-20, -10, -5, 0)      # моменты прогноза в таблице [1]
ORIGIN_RANGE = range(-25, 1)            # все моменты прогноза в бэктесте (по ним подбирается сглаживание)
CHECK_DAYS = (1, 3, 5, 10, 15, 20, 25)  # контрольные даты для проверки накопленного факта (+ конец месяца)
TREND_HALFLIVES = (0, 1, 3, 6, 12, np.inf)   # темп роста плана/факта: 0 = без роста, иначе полураспад весов, мес.
LEVEL_HALFLIVES = (1, 2, 3, 6, 12, np.inf)   # сглаживание уровня: вес месяца падает вдвое каждые N месяцев
LEVEL_PRIOR_WEIGHTS = (0, 1, 3, 10, np.inf)  # тяга уровня к среднему модели (inf = уровень не подстраивать)
PARTIAL_GAINS = (0, 0.5, 1, 2)               # вес неполного текущего месяца при прогнозе следующего
OUT_DIR = "."
RANDOM_STATE = 42
N_SEEDS = 5                    # сколько бустингов с разными seed усреднять (1 — быстрее, но прогноз «дрожит»)
CALENDAR_BLENDS = (0.0, 0.5, 1.0)  # доля бустинга в календарной модели (0 = только GLM); лучшая выбирается бэктестом

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


def prepare(df: pd.DataFrame, last_fact_date=None):
    """Дневная сетка до конца горизонта прогноза + календарные признаки + месячная таблица."""
    d = df[[DATE_COL, FACT_COL, PLAN_COL]].copy()
    d.columns = ["date", "fact", "plan"]
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d = d.groupby("date", as_index=False).agg(fact=("fact", lambda s: s.sum(min_count=1)),
                                              plan=("plan", lambda s: s.sum(min_count=1)))
    if last_fact_date is None:
        last_fact_date = d.loc[d["fact"].notna(), "date"].max()
    last_fact_date = pd.Timestamp(last_fact_date)
    last_plan_date = d.loc[d["plan"].notna(), "date"].max()
    start = d["date"].min()
    end = max(d["date"].max(), (last_fact_date.to_period("M") + HORIZON_MONTHS + 1).end_time.normalize())

    d = pd.DataFrame({"date": pd.date_range(start, end, freq="D")}).merge(d, on="date", how="left")
    d["ym"] = d["date"].dt.to_period("M")
    if CUMULATIVE_INPUT:  # накопленные с начала месяца -> дневные
        for col in ["fact", "plan"]:
            d[col] = d.groupby("ym")[col].transform(_cum_to_daily)
    known_f, known_p = d["date"] <= last_fact_date, d["date"] <= last_plan_date
    d["fact"] = d["fact"].where(known_f).fillna(0.0).where(known_f).round(2)
    d["plan"] = d["plan"].where(known_p).fillna(0.0).where(known_p).round(2)

    d = d.merge(build_calendar(start, end), on="date", how="left")

    mt = pd.DataFrame(index=pd.period_range(d["ym"].min(), d["ym"].max(), freq="M"))
    mt["fact_m"] = d.groupby("ym")["fact"].sum()
    mt["plan_m"] = d.groupby("ym")["plan"].sum()
    whole = mt.index.start_time >= start                                   # месяц целиком внутри данных
    mt["complete"] = whole & (mt.index.end_time.normalize() <= last_fact_date)
    mt["plan_full"] = whole & (mt.index.end_time.normalize() <= last_plan_date) & (mt["plan_m"] > 0)
    mt.loc[~mt["plan_full"], "plan_m"] = np.nan
    mt.loc[~mt["complete"], "fact_m"] = np.nan

    d["plan_m"] = d["ym"].map(mt["plan_m"])
    d["y"] = d["fact"] / d["plan_m"]  # цель дневной модели: доля плана месяца, собранная в этот день
    return d, mt, last_fact_date


# Признаки дневной модели: только календарь. Плана прогнозного месяца нет, а уровень прогнозируется отдельно,
# чтобы «календарные» месяцы (январь, май) не тянули за собой следующий.
SHAPE_FEATURES = (["dow", "day", "month", "dim", "days_to_eom", "is_workday", "is_holiday",
                   "wd_num", "wd_total", "wd_left", "wd_frac", "nonwork_before", "nonwork_after",
                   "first_wd_month", "last_wd_month", "ny_break", "days_since_pay"]
                  + [f"pay_{p}" for p in PAYDAYS])

M_FULL, M_FACT, M_REF = "План × уровень × календарь", "Тренд факта × календарь", "Референсные месяцы × прогноз плана"
METHODS = (M_FULL, M_FACT, M_REF)
SHORT = {M_FULL: "план×уровень", M_FACT: "тренд факта", M_REF: "реф. месяцы"}


# ============================================================================
# ДНЕВНАЯ МОДЕЛЬ («календарь»)
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


def glm_features(g: pd.DataFrame) -> pd.DataFrame:
    """Признаки для мультипликативной модели: профиль по числу месяца × день недели × эффекты выходных и праздников."""
    work = g["is_workday"] == 1
    X = {f"d{dd}": g["day"] == dd for dd in range(1, 32)}
    X.update({f"w{w}": g["dow"] == w for w in range(7)})
    X.update(work=work, hol=g["is_holiday"] == 1)
    for k in (1, 2, 3):
        X[f"nb{k}"] = work & (g["nonwork_before"].clip(upper=3) == k)  # первый рабочий день после k выходных
        X[f"na{k}"] = work & (g["nonwork_after"].clip(upper=3) == k)   # последний рабочий день перед k выходными
    X = pd.DataFrame(X, index=g.index).astype(float)
    X["nb_long"] = (work & (g["nonwork_before"] >= 3)) * np.log1p(g["nonwork_before"])  # после длинных каникул
    X["na_long"] = (work & (g["nonwork_after"] >= 3)) * np.log1p(g["nonwork_after"])    # перед длинными каникулами
    X["first_wd"], X["last_wd"], X["ny"] = g["first_wd_month"], g["last_wd_month"], g["ny_break"]
    for p in PAYDAYS:
        X[f"pay{p}"] = g[f"pay_{p}"]
    X["dim"] = g["dim"] - 30.5
    return X


class CalendarGLM:
    """Пуассоновская регрессия (лог-ссылка): жёстче бустинга, лучше держит сумму за месяц на короткой истории."""
    SCALE = 30.0

    def fit(self, train, sample_weight=None):
        from sklearn.linear_model import PoissonRegressor
        self.m = PoissonRegressor(alpha=1e-4, max_iter=3000).fit(
            glm_features(train), train["y"].clip(lower=0) * self.SCALE, sample_weight=sample_weight)
        return self

    def predict(self, g):
        return self.m.predict(glm_features(g)) / self.SCALE


def fit_models(train: pd.DataFrame, target_month):
    tr = train[~train["date"].isin(pd.to_datetime(EXCLUDE_DATES))]
    w = None
    if RECENCY_HALFLIFE:
        age = (target_month - tr["ym"]).apply(lambda x: x.n).to_numpy()
        w = 0.5 ** (age / RECENCY_HALFLIFE)
    boost = SeedBag().fit(tr[SHAPE_FEATURES], tr["y"], sample_weight=w)
    glm = CalendarGLM().fit(tr, sample_weight=w)
    return boost, glm


def walk_forward(d: pd.DataFrame, train_months: list, origin):
    """Для каждого месяца k: модели обучены только на полных месяцах ДО k и прогнозируют месяцы k и k+1
    (последние, рабочие модели — до конца горизонта). Прогноз — доля плана месяца на каждый день.
    Возвращает прогнозы бустинга и GLM отдельно: их смесь выбирается по бэктесту."""
    by_month = {m: g.reset_index(drop=True) for m, g in d.groupby("ym")}
    first = train_months[0] + MIN_TRAIN_MONTHS
    if first > origin:
        raise ValueError(f"Мало истории: нужно хотя бы {MIN_TRAIN_MONTHS} полных месяцев (уменьшите MIN_TRAIN_MONTHS)")
    pb, pg = {}, {}
    for k in pd.period_range(first, origin, freq="M"):
        train = d[(d["ym"] < k) & d["ym"].isin(train_months)]
        boost, glm = fit_models(train, k)
        last = k + 1 if k < origin else origin + HORIZON_MONTHS
        for t in pd.period_range(k, last, freq="M"):
            if t in by_month and len(by_month[t]) == t.days_in_month:
                g = by_month[t]
                pb[(k, t)] = np.clip(boost.predict(g[SHAPE_FEATURES]), 0, None)
                pg[(k, t)] = np.clip(glm.predict(g), 0, None)
        print(f"  модели {k}: обучены на {train['ym'].nunique()} мес.")
    return pb, pg, by_month


# ============================================================================
# ПЛАН, УРОВЕНЬ, ТРЕНД
# ============================================================================
def adjusted_fact(g: pd.DataFrame, expected: np.ndarray) -> np.ndarray:
    """Факт, в котором дни из EXCLUDE_DATES заменены ожиданием модели (чтобы разовые суммы не сдвигали уровень)."""
    excl = g["date"].isin(pd.to_datetime(EXCLUDE_DATES)).to_numpy()
    return np.where(excl, expected, g["fact"].to_numpy())


def level_history(preds, by_month, mt, train_months) -> pd.DataFrame:
    rows = []
    for m in train_months:
        if (m, m) not in preds:
            continue
        g = by_month[m]
        plan_m = mt.loc[m, "plan_m"]
        share = preds[(m, m)]
        fa = adjusted_fact(g, share * plan_m).sum()
        rows.append(dict(ym=m, P=plan_m, F=g["fact"].sum(), R=g["fact"].sum() / plan_m, S=share.sum(),
                         L=fa / (plan_m * share.sum()), A=fa / share.sum()))
    lv = pd.DataFrame(rows).set_index("ym")
    lv["dL"] = lv["L"].pct_change()
    lv["dP"] = lv["P"].pct_change()
    return lv


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


def plan_forecast(case, cfg) -> float:
    s = case["plans"]
    if cfg["unit"] == "день":
        return trend_forecast(s / s.index.days_in_month, case["T"], cfg["h"]) * case["T"].days_in_month
    return trend_forecast(s, case["T"], cfg["h"])


def level_estimate(case, h, w0, gain) -> float:
    """Уровень на прогнозный месяц: взвешенное среднее уровней прошлых месяцев (вес падает с давностью),
    неполного текущего месяца (вес ~ его собранной доле × gain) и «нейтрального» уровня 1 с весом w0."""
    if np.isinf(w0):
        return 1.0
    decay = (lambda a: 0.5 ** (np.asarray(a, float) / h)) if np.isfinite(h) else (lambda a: np.ones_like(np.asarray(a, float)))
    w = decay(case["ages"])
    num, den = w0 + (w * case["Ls"]).sum(), w0 + w.sum()
    if case["partial"] is not None and gain > 0:
        lp, age, frac = case["partial"]
        wp = gain * frac * float(decay(age))
        num, den = num + wp * lp, den + wp
    return num / den if den > 0 else 1.0


# ============================================================================
# ПРОГНОЗ НА МЕСЯЦ ИЗ ЗАДАННОГО МОМЕНТА
# ============================================================================
def reference_curve(by_month, T, refs) -> np.ndarray:
    """Текущий подход: средняя кривая выполнения (накопл. факт / план месяца) референсных месяцев
    по календарному дню. Возвращает дневные доли плана."""
    n = T.days_in_month
    curves = []
    for r in refs:
        g = by_month.get(r)
        if g is None or g["fact"].isna().any() or not g["plan_m"].iloc[0] > 0:
            continue
        c = (g["fact"].cumsum() / g["plan_m"]).to_numpy()
        curves.append(np.concatenate([c, np.repeat(c[-1], max(0, n - len(c)))])[:n])
    if not curves:
        return np.full(n, np.nan)
    return np.diff(np.r_[0.0, np.mean(curves, axis=0)])


def build_case(T, Op, c, ctx):
    """Всё, что известно о месяце T в момент прогноза: идёт месяц Op, по нему известно c дней факта
    (c = 0 — прогноз по итогам месяца перед Op). План и факт месяца T не используются."""
    preds, by_month, lv, mt = ctx["preds"], ctx["by_month"], ctx["lv"], ctx["mt"]
    g = by_month[T]
    k = c - sum(p.days_in_month for p in pd.period_range(Op, T - 1, freq="M")) if T > Op else 0
    hist = lv[lv.index < Op]
    known_plans = mt.loc[mt["plan_full"] & (mt.index <= min(Op, T - 1)), "plan_m"]
    case = dict(T=T, Op=Op, c=c, k=k, share=preds[(Op, T)], Ls=hist["L"].to_numpy(),
                ages=np.array([(T - m).n for m in hist.index], float), A=hist["A"], plans=known_plans,
                partial=None, dates=g["date"].to_numpy(), actual=g["fact"].to_numpy(),
                plan_actual=mt.loc[T, "plan_m"])
    if c > 0 and mt.loc[Op, "plan_full"] and (Op, Op) in preds:
        go = by_month[Op]
        sh = preds[(Op, Op)]
        e = sh * mt.loc[Op, "plan_m"]
        a = adjusted_fact(go, e)
        case["partial"] = (a[:c].sum() / e[:c].sum(), (T - Op).n, sh[:c].sum() / sh.sum())
    refs = [m for m in ctx["train_months"] if m < Op][-N_REF_MONTHS:]
    case["ref"] = reference_curve(by_month, T, refs)
    return case


def forecast_for(method, case, cfg):
    p_hat = plan_forecast(case, cfg["plan"])
    if method == M_REF:
        return p_hat * case["ref"]
    if method == M_FACT:
        return trend_forecast(case["A"], case["T"], cfg["fact_h"]) * case["share"]
    lv = cfg["level"]
    return p_hat * level_estimate(case, lv["h"], lv["w0"], lv["gain"]) * case["share"]


# ============================================================================
# БЭКТЕСТ: прогноз фиксируется в момент прогноза и сверяется с фактом месяца
# ============================================================================
def backtest_cases(ctx):
    cases = []
    for T in ctx["train_months"]:
        if not all(key in ctx["preds"] for key in [(T - 1, T), (T, T), (T - 1, T - 1)]):
            continue
        if not (ctx["lv"].index < T - 1).any():  # нужен хотя бы один прошлый месяц с уровнем — одинаково для всех моментов
            continue
        for k in ORIGIN_RANGE:
            Op, c = (T, 0) if k == 0 else (T - 1, (T - 1).days_in_month + k)
            if c >= 1 or k == 0:
                case = build_case(T, Op, c, ctx)
                if not case["plans"].empty:
                    cases.append(case)
    if not cases:
        raise ValueError("Нет месяцев для бэктеста: мало истории")
    return cases


def tune(cases):
    """Подбор параметров по бэктесту: план — по ошибке плана, тренд факта и уровень — по ошибке итога."""
    act = np.array([c["actual"].sum() for c in cases])
    plan_act = np.array([c["plan_actual"] for c in cases])
    S = np.array([c["share"].sum() for c in cases])

    best_p = None
    for unit, h in itertools.product(("месяц", "день"), TREND_HALFLIVES):
        p = np.array([plan_forecast(c, dict(unit=unit, h=h)) for c in cases])
        score = np.nanmean(np.abs(p / plan_act - 1))
        if best_p is None or score < best_p["score"] - 1e-12:
            best_p = dict(unit=unit, h=h, score=score)
    p_hat = np.array([plan_forecast(c, best_p) for c in cases])

    best_f = None
    for h in TREND_HALFLIVES:
        a = np.array([trend_forecast(c["A"], c["T"], h) for c in cases])
        score = np.nanmean(np.abs(a * S / act - 1))
        if best_f is None or score < best_f[1] - 1e-12:
            best_f = (h, score)

    best_l = None
    for h, w0, gain in itertools.product(LEVEL_HALFLIVES, LEVEL_PRIOR_WEIGHTS, PARTIAL_GAINS):
        lvl = np.array([level_estimate(c, h, w0, gain) for c in cases])
        score = np.nanmean(np.abs(p_hat * lvl * S / act - 1))
        if best_l is None or score < best_l["score"] - 1e-12:
            best_l = dict(h=h, w0=w0, gain=gain, score=score)
    return dict(plan=best_p, fact_h=best_f[0], level=best_l)


def evaluate(cases, cfg):
    report_k = set(ORIGINS_REPORT) | {FORECAST_ORIGIN}
    tot, days = [], []
    for c in cases:
        a = c["actual"]
        ca = np.cumsum(a)
        p_hat = plan_forecast(c, cfg["plan"])
        for name in METHODS:
            f = forecast_for(name, c, cfg)
            cf = np.cumsum(f)
            with np.errstate(divide="ignore", invalid="ignore"):
                ce = np.where(ca > 0, cf / ca - 1, np.nan)
            row = dict(method=name, ym=str(c["T"]), k=c["k"], err_total=cf[-1] / ca[-1] - 1,
                       err_plan=p_hat / c["plan_actual"] - 1,
                       mean_abs_cum=np.nanmean(np.abs(ce)), wape_daily=np.abs(f - a).sum() / a.sum())
            if name == M_FULL:  # итог = план × уровень × календарь, поэтому ошибка раскладывается на множители
                row["err_level"] = (1 + row["err_total"]) / (1 + row["err_plan"]) - 1
            tot.append(row)
            if c["k"] in report_k:
                days.append(pd.DataFrame(dict(method=name, ym=str(c["T"]), k=c["k"], date=c["dates"],
                                              day=np.arange(1, len(a) + 1), fact=a, forecast=f,
                                              cum_fact=ca, cum_forecast=cf, cum_err=ce)))
    return pd.DataFrame(tot), pd.concat(days, ignore_index=True)


# ============================================================================
# ОТЧЁТ
# ============================================================================
def _k_label(k):
    return "по итогам прошлого месяца" if k == 0 else f"за {-k} дн. до начала месяца"


def _k_col(k):
    return f"за {-k} дн."


def _pct(x, nd=2):
    return "—" if pd.isna(x) else f"{100 * x:+.{nd}f}%".replace("+-", "-")


def _apct(x, nd=2):
    return "—" if pd.isna(x) else f"{100 * x:.{nd}f}%"


def report_backtest(tot, days, best_name):
    t = tot[tot["k"].isin(ORIGINS_REPORT)]
    a = t.groupby(["method", "k"])["err_total"].agg(
        mae=lambda s: s.abs().mean(), within=lambda s: (s.abs() < 0.01).mean()).reset_index()
    a["v"] = a.apply(lambda r: f"{100 * r.mae:.2f}% / {100 * r.within:.0f}%", axis=1)
    tab = a.pivot(index="method", columns="k", values="v").reindex(index=list(METHODS), columns=list(ORIGINS_REPORT))
    tab.columns = [_k_col(k) for k in tab.columns]
    print(f"\n[1] Ошибка ИТОГА месяца в зависимости от того, за сколько дней до начала месяца сделан прогноз "
          f"({t['ym'].nunique()} мес. бэктеста).")
    print("    Средняя |ошибка| / доля месяцев с |ошибкой| < 1%:")
    print(tab.to_string())

    d0 = days[days["k"] == FORECAST_ORIGIN]
    rows = {}
    for name in METHODS:
        g = d0[d0["method"] == name]
        r = {f"{dd}-е": g.loc[g["day"] == dd, "cum_err"].abs().mean() for dd in CHECK_DAYS}
        r["конец"] = g.groupby("ym")["cum_err"].last().abs().mean()
        r["все дни"] = g["cum_err"].abs().mean()
        rows[name] = r
    tb = pd.DataFrame(rows).T.apply(lambda col: col.map(_apct))
    tb["дней с |ош.|<1%"] = [f"{100 * (d0.loc[d0['method'] == n, 'cum_err'].abs() < 0.01).mean():.0f}%" for n in METHODS]
    print(f"\n[2] Прогноз {_k_label(FORECAST_ORIGIN)}. Средняя |ошибка| НАКОПЛЕННОГО факта на дату месяца:")
    print(tb.to_string())

    t0 = tot[tot["k"] == FORECAST_ORIGIN]
    full = t0[t0["method"] == M_FULL].set_index("ym")
    per = pd.DataFrame({"ош. плана": full["err_plan"].map(_pct), "ош. уровня": full["err_level"].map(_pct)})
    for name in METHODS:
        per[f"итог: {SHORT[name]}"] = t0[t0["method"] == name].set_index("ym")["err_total"].map(_pct)
    per["ср.|ош.| на даты"] = t0[t0["method"] == best_name].set_index("ym")["mean_abs_cum"].map(_apct)
    print(f"\n[3] По месяцам, прогноз {_k_label(FORECAST_ORIGIN)} (+ = прогноз выше факта).")
    print("    Для «план × уровень × календарь»: (1 + ош. итога) = (1 + ош. плана) × (1 + ош. уровня).")
    print(per.to_string())
    print(f"    Средняя |ошибка| прогноза плана: {_apct(full['err_plan'].abs().mean())}, "
          f"уровня: {_apct(full['err_level'].abs().mean())}.")


def report_level(lv):
    t = pd.DataFrame({
        "рост плана": lv["dP"].map(_pct),
        "выполнение": lv["R"].map(lambda x: _apct(x, 1)),
        "ожидание по календарю": lv["S"].map(lambda x: _apct(x, 1)),
        "уровень L": lv["L"].map(lambda x: f"{x:.3f}"),
        "скачок L": lv["dL"].map(_pct),
        "скачок выполнения": lv["R"].pct_change().map(_pct),
    })
    t.index = t.index.astype(str)
    print("\n[4] По месяцам. «Ожидание по календарю» — выполнение, которого модель ждала до начала месяца")
    print("    при среднем уровне; уровень L = выполнение / ожидание. Скачки L календарём не объясняются.")
    print(t.to_string())
    print(f"\n    Средний |скачок| выполнения месяц к месяцу: {_apct(lv['R'].pct_change().abs().mean())}; "
          f"после поправки на календарь (|скачок L|): {_apct(lv['dL'].abs().mean())}.")
    print("    |Скачок L| — примерный предел точности прогноза следующего месяца по дате, факту и плану.")


def report_anomalies(preds, by_month, lv, top=8):
    rows = []
    for m in lv.index:
        g = by_month[m]
        exp = preds[(m, m)] * lv.loc[m, "P"]
        f = g["fact"].to_numpy()
        for i in np.flatnonzero(exp > 0):
            rows.append(dict(date=g["date"].iloc[i], ratio=f[i] / exp[i], share=f[i] / f.sum()))
    an = pd.DataFrame(rows)
    an = an[an["ratio"] > 2].sort_values("ratio", ascending=False).head(top)
    if an.empty:
        return
    print("\n[5] Дни, где факт больше ожидания модели более чем вдвое (если это разовые суммы — внесите в EXCLUDE_DATES):")
    print(pd.DataFrame({"дата": an["date"].dt.strftime("%d.%m.%Y %a"), "факт / ожидание": an["ratio"].round(1),
                        "доля факта месяца": an["share"].map(lambda x: _apct(x, 1))}).to_string(index=False))


def _fmt_money(x):
    return f"{x:,.0f}".replace(",", " ")


# ============================================================================
# РАБОЧИЙ ПРОГНОЗ
# ============================================================================
def production(ctx, origin, c, last_fact_date, cfg, best_name, tot, days):
    q = (days[(days["method"] == best_name) & (days["k"] == FORECAST_ORIGIN)]
         .groupby("day")["cum_err"].quantile([0.1, 0.9]).unstack())
    mon = last_fact_date.to_period("M")  # месяц, по которому есть факт: сверяем с ним прогноз, сделанный заранее
    jobs = []
    if FORECAST_ORIGIN == 0:
        mon_origin = (mon, 0)
    else:
        prev = mon - 1
        mon_origin = (prev, prev.days_in_month + FORECAST_ORIGIN)
    if (mon_origin[0], mon) in ctx["preds"] and mon in ctx["by_month"]:
        jobs.append(("check", mon, *mon_origin))
    for i in range(1, HORIZON_MONTHS + 1):
        jobs.append(("forecast", mon + i, origin, c))

    out, summary = [], []
    for kind, T, Op, cc in jobs:
        case = build_case(T, Op, cc, ctx)
        f = forecast_for(best_name, case, cfg)
        p_hat = plan_forecast(case, cfg["plan"])
        cf = np.cumsum(f)
        a = case["actual"]
        ca = np.where(np.isnan(a), np.nan, np.nancumsum(a))
        qq = q.reindex(np.arange(1, len(f) + 1)).ffill().bfill()
        rows = pd.DataFrame({"month": str(T), "date": case["dates"], "forecast": f, "cum_forecast": cf,
                             "fact": a, "cum_fact": ca, "err_cum_pct": 100 * (cf / ca - 1),
                             "corridor_low": cf / (1 + qq[0.9].to_numpy()), "corridor_high": cf / (1 + qq[0.1].to_numpy())})
        out.append(rows)
        k = case["k"]
        e = tot[(tot["method"] == best_name) & (tot["k"] == max(k, min(ORIGIN_RANGE)))]["err_total"]
        lo, hi = (cf[-1] / (1 + e.quantile(0.9)), cf[-1] / (1 + e.quantile(0.1))) if len(e) >= 5 else (np.nan, np.nan)
        made = pd.Timestamp(Op.start_time) + pd.Timedelta(days=cc - 1) if cc > 0 else pd.Timestamp(Op.start_time) - pd.Timedelta(days=1)
        summary.append({"месяц": str(T), "прогноз по факту на": f"{made:%d.%m.%Y}",
                        "прогноз плана": _fmt_money(p_hat), "прогноз факта": _fmt_money(cf[-1]),
                        "80% интервал факта": (f"{_fmt_money(lo)} – {_fmt_money(hi)}"
                                               + (" (прибл.: дальше, чем бэктест)" if k < min(ORIGIN_RANGE) else ""))
                        if np.isfinite(lo) else "—",
                        "выполнение (к прогнозу плана)": _apct(cf[-1] / p_hat, 1)})
        if kind == "check":
            n = int(np.sum(~np.isnan(a)))
            dev = cf[n - 1] / ca[n - 1] - 1
            lo_d, hi_d = (q.loc[n, 0.1], q.loc[n, 0.9]) if n in q.index else (np.nan, np.nan)
            ok = lo_d <= dev <= hi_d
            print(f"\nПРОВЕРКА ПО ФАКТУ, {T}: прогноз сделан {made:%d.%m.%Y} (без плана и факта {T}), факт по "
                  f"{pd.Timestamp(case['dates'][n - 1]):%d.%m.%Y}.")
            print(f"  Накопленный прогноз {_fmt_money(cf[n - 1])}, факт {_fmt_money(ca[n - 1])}: ошибка {_pct(dev)}. "
                  f"В бэктесте на {n}-й день у 80% месяцев ошибка была от {_pct(lo_d)} до {_pct(hi_d)} — "
                  + ("в пределах обычного." if ok else "ВНЕ обычного коридора, стоит разобраться."))
            if np.isfinite(case["plan_actual"]):
                print(f"  План {T} оказался {_fmt_money(case['plan_actual'])}, прогноз плана был {_fmt_money(p_hat)} "
                      f"({_pct(p_hat / case['plan_actual'] - 1)}).")
    return pd.concat(out, ignore_index=True), pd.DataFrame(summary)


# ============================================================================
# ГРАФИКИ
# ============================================================================
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = {M_FULL: "#2a78d6", M_FACT: "#eb6834", M_REF: "#1baf7a"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_backtest(days, path):
    import matplotlib.pyplot as plt
    d0 = days[days["k"] == FORECAST_ORIGIN]
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=SURFACE)
    _style(ax)
    for name in METHODS:
        s = d0[d0["method"] == name].groupby("day")["cum_err"].apply(lambda x: x.abs().mean())
        ax.plot(s.index, 100 * s.values, color=SERIES[name], linewidth=2, label=name)
    ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10, 20, 50, 100]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.axhline(1, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_xlabel("День месяца", color=INK2)
    ax.set_ylabel("Средняя |ошибка| накопленного факта (лог. шкала)", color=INK2)
    ax.set_title(f"Бэктест: прогноз {_k_label(FORECAST_ORIGIN)}, сверка с фактом на каждую дату",
                 color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_forecast(fc, path):
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    months = list(dict.fromkeys(fc["month"]))[:2]
    fig, axes = plt.subplots(1, len(months), figsize=(5.5 * len(months), 4.5), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    color = SERIES[M_FULL]
    for ax, m in zip(axes, months):
        _style(ax)
        g = fc[fc["month"] == m]
        ax.fill_between(g["date"], g["corridor_low"] / 1e6, g["corridor_high"] / 1e6, color=color, alpha=0.12,
                        linewidth=0, label="Коридор для факта (80% месяцев бэктеста)")
        ax.plot(g["date"], g["cum_forecast"] / 1e6, color=color, linewidth=2, linestyle=(0, (4, 3)), label="Прогноз")
        if g["cum_fact"].notna().any():
            ax.plot(g["date"], g["cum_fact"] / 1e6, color=INK, linewidth=2, label="Факт")
        ax.set_xlim(g["date"].min() - pd.Timedelta(hours=12), g["date"].max() + pd.Timedelta(hours=12))
        ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 8, 15, 22, 29]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
        has_fact = g["cum_fact"].notna().any()
        ax.set_title(f"{m}: " + ("прогноз против факта" if has_fact else "прогноз"), color=INK, loc="left", fontsize=11)
        ax.set_ylabel("Накопленный факт, млн", color=INK2)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ============================================================================
# ЗАПУСК
# ============================================================================
def run(df: pd.DataFrame, last_fact_date=LAST_FACT_DATE, make_plots=True):
    d, mt, last_fact_date = prepare(df, last_fact_date)
    origin = (last_fact_date + pd.Timedelta(days=1)).to_period("M")   # первый месяц без полного факта
    c = (last_fact_date + pd.Timedelta(days=1)).day - 1                # сколько дней этого месяца известно
    train_months = [m for m in mt.index[mt["complete"] & mt["plan_full"]] if m < origin]
    print(f"Данные: {d.loc[d['fact'].notna(), 'date'].min():%d.%m.%Y} – {last_fact_date:%d.%m.%Y}; "
          f"план известен по {mt.index[mt['plan_full']].max()} включительно")

    print("Обучение моделей шаг за шагом (каждая — только на прошлом)...")
    pb, pg, by_month = walk_forward(d, train_months, origin)

    variants = {}
    for wb in CALENDAR_BLENDS:  # календарная модель: GLM, бустинг или смесь — выбирает бэктест
        preds = {key: wb * pb[key] + (1 - wb) * pg[key] for key in pb}
        lv = level_history(preds, by_month, mt, train_months)
        ctx = dict(preds=preds, by_month=by_month, lv=lv, mt=mt, train_months=train_months)
        cases = backtest_cases(ctx)
        cfg = tune(cases)
        tot, days = evaluate(cases, cfg)
        score = tot.groupby("method")["err_total"].apply(lambda s: s.abs().mean())
        variants[wb] = dict(ctx=ctx, lv=lv, cfg=cfg, tot=tot, days=days, score=score)
    wb = min(variants, key=lambda w: variants[w]["score"].min())
    v = variants[wb]
    ctx, lv, cfg, tot, days = v["ctx"], v["lv"], v["cfg"], v["tot"], v["days"]
    best_name = v["score"].idxmin()

    cal_name = lambda w: "GLM" if w == 0 else ("бустинг" if w == 1 else f"смесь ({100 * w:.0f}% бустинга)")
    comp = pd.DataFrame({cal_name(w): variants[w]["score"].reindex(list(METHODS)).map(_apct) for w in variants}).T
    print("\nКалендарная модель: средняя |ошибка| итога месяца по всем моментам прогноза бэктеста")
    print(comp.to_string())
    print(f"Выбрано: {cal_name(wb)}")

    p, lvc = cfg["plan"], cfg["level"]

    def hh(h):
        return "без роста" if h == 0 else ("средний за всю историю" if np.isinf(h) else "полураспад %g мес." % h)

    unit = "в день" if p["unit"] == "день" else "за месяц"
    if np.isinf(lvc["w0"]):
        lvl_txt = "не подстраивается"
    else:
        smooth = "без затухания" if np.isinf(lvc["h"]) else "полураспад %g мес." % lvc["h"]
        lvl_txt = "сглаживание %s, тяга к среднему %g, вес неполного месяца %g" % (smooth, lvc["w0"], lvc["gain"])
    print(f"\nПодобрано по бэктесту: план — последний известный ({unit}) × темп роста ({hh(p['h'])}); "
          f"тренд факта — {hh(cfg['fact_h'])}; уровень — {lvl_txt}")

    report_backtest(tot, days, best_name)
    report_level(lv)
    report_anomalies(ctx["preds"], by_month, lv)

    print(f"\nЛучший по ошибке итога месяца: {best_name}")
    fc, fsum = production(ctx, origin, c, last_fact_date, cfg, best_name, tot, days)
    print("\nПрогноз:")
    print(fsum.to_string(index=False))

    fc.to_csv(f"{OUT_DIR}/forecast_daily.csv", index=False)
    days[days["method"].isin([best_name, M_REF])].to_csv(f"{OUT_DIR}/backtest_daily.csv", index=False)
    tot.to_csv(f"{OUT_DIR}/backtest_months.csv", index=False)
    if make_plots:
        plot_backtest(days, f"{OUT_DIR}/backtest_by_date.png")
        plot_forecast(fc, f"{OUT_DIR}/forecast_vs_fact.png")
    return dict(daily=d, months=mt, level=lv, cfg=cfg, calendar_blend=wb, backtest_months=tot, backtest_daily=days,
                best=best_name, forecast=fc, forecast_summary=fsum, variants=variants)


def load_data() -> pd.DataFrame:
    """Замените на свой запрос. Нужны колонки DATE_COL, FACT_COL, PLAN_COL."""
    # import sqlalchemy as sa
    # engine = sa.create_engine("...")
    # return pd.read_sql("select pay_date as date, fact, plan from ...", engine)
    raise NotImplementedError("Заполните load_data() своим запросом к базе")


if __name__ == "__main__":
    res = run(load_data())

# ----------------------------------------------------------------------------
# Если в [3] большая «ошибка уровня», а в [4] скачки L заметно больше 1%, по трём колонкам точнее не станет.
# Что обычно объясняет уровень заранее (добавляется отдельным слагаемым или признаком уровня):
#   - досрочные погашения отдельно от плановых платежей: они не привязаны к плану и сильнее всего «шумят»;
#   - просроченная задолженность на начало месяца по корзинам: сборы с просрочки зависят от неё, а не от плана;
#   - разбивка по продуктам / сегментам: у каждого свой уровень, итог = сумма прогнозов.
# Если большая «ошибка плана» — план следующего месяца лучше брать из графика платежей по действующим договорам
# (он известен заранее), а не экстраполировать.
# ----------------------------------------------------------------------------
