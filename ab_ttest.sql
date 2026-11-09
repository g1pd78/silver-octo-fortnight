-- A/B-тест: только p_value и is_significant, без фильтров и поправок.
-- SQLite 3.25+ с SQRT/EXP. Шаблонизация Superset не требуется.
-- Наблюдение = независимая строка; повторные наблюдения одного объекта
-- нужно предварительно агрегировать до единицы рандомизации.
-- group_name = <эксперимент>_<группа>; контроль заканчивается на _control.
-- Метрики: SUM(x) / SUM(y). Линеаризация: L = x - k*y, k из контроля.
-- Сохранён исходный метод: статистика Уэлча + НОРМАЛЬНОЕ приближение
-- двустороннего p-value (не точное распределение Стьюдента для малых n).
-- Значимость: p < 0.05. Одна строка результата на метрику и пару групп.
-- Идентификаторы используются только внутри расчёта и не выводятся.
-- NULL означает, что тест не определён: мало данных, нулевой знаменатель
-- либо нулевая стандартная ошибка. NULL не заменяется на «нет значимости».

WITH source_rows AS (
    SELECT
        *,
        RTRIM(RTRIM(group_name, REPLACE(group_name, '_', '')), '_') AS experiment,
        CASE WHEN group_name LIKE '%\_control' ESCAPE '\' THEN 1 ELSE 0 END AS is_control
    FROM your_table
),
metric_rows AS (
    SELECT 'mail_cum' AS metric_name, experiment, group_name, is_control,
           mail_cum * 1.0 AS x, acc_count * 1.0 AS y
    FROM source_rows
    UNION ALL
    SELECT 'recovery_rate', experiment, group_name, is_control,
           COALESCE(od_fact, 0) * 1.0, COALESCE(od_start, 0) * 1.0
    FROM source_rows
    UNION ALL
    SELECT 'email_cum', experiment, group_name, is_control,
           email_cum * 1.0, acc_count * 1.0
    FROM source_rows
),
valid_rows AS (
    -- Один и тот же набор непустых пар используется для k, среднего и n.
    SELECT * FROM metric_rows WHERE x IS NOT NULL AND y IS NOT NULL
),
kappa AS (
    SELECT metric_name, experiment, SUM(x) / NULLIF(SUM(y), 0) AS k
    FROM valid_rows
    WHERE is_control = 1
    GROUP BY metric_name, experiment
),
linearized AS (
    SELECT r.metric_name, r.experiment, r.group_name, r.is_control, r.y,
           r.x - k.k * r.y AS l
    FROM valid_rows r
    JOIN kappa k ON k.metric_name = r.metric_name AND k.experiment = r.experiment
),
centered AS (
    SELECT *, AVG(l) OVER (PARTITION BY metric_name, experiment, group_name) AS mean_l
    FROM linearized
),
group_stats AS (
    SELECT metric_name, experiment, group_name, MAX(is_control) AS is_control,
           COUNT(l) AS n, MAX(mean_l) AS mean_l, AVG(y) AS mean_y,
           -- Выборочная дисперсия (n-1), без нестандартной функции VARIANCE.
           SUM((l - mean_l) * (l - mean_l)) / NULLIF(COUNT(l) - 1, 0) AS var_l
    FROM centered
    GROUP BY metric_name, experiment, group_name
),
comparisons AS (
    SELECT CASE WHEN t.n >= 2 AND c.n >= 2 AND t.mean_y <> 0 AND c.mean_y <> 0
                THEN (t.mean_l - c.mean_l)
                     / NULLIF(SQRT(t.var_l / t.n + c.var_l / c.n), 0)
           END AS t_stat
    FROM group_stats t
    JOIN group_stats c
      ON c.metric_name = t.metric_name AND c.experiment = t.experiment AND c.is_control = 1
    WHERE t.is_control = 0
),
normal_tail AS (
    SELECT *, 1.0 / (1.0 + 0.2316419 * ABS(t_stat)) AS a
    FROM comparisons
),
p_values AS (
    SELECT MIN(1.0, MAX(0.0,
               2 * 0.3989422804 * EXP(-t_stat * t_stat / 2)
                 * a * (0.319381530 + a * (-0.356563782
                 + a * (1.781477937 + a * (-1.821255978 + a * 1.330274429))))
           )) AS p_value
    FROM normal_tail
)
SELECT
    p_value,
    CASE WHEN p_value IS NOT NULL THEN p_value < 0.05 END AS is_significant
FROM p_values;
