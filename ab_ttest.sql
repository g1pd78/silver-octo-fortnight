-- A/B-тест для SQLite с функциями SQRT и EXP.
-- your_table содержит обе группы. Сопоставляем _pilot и _control
-- по pilot_id и общей части group_name перед суффиксом.
-- Для флагов 0/1 знаменатель — число строк группы, включая разные дни entity.
-- Для recovery_rate знаменатель — SUM(b10_start) по непустым парам с od_fact.
-- Разброс считаем по entity, объединяя её дни внутри группы.
-- Предполагается, что entity принадлежит одной группе эксперимента.
-- Пропущенный или некорректный флаг делает сравнение по метрике неопределённым.
-- Двусторонний p-value использует нормальное приближение.
-- NULL означает, что сравнение невозможно, например без контроля или при нулевой дисперсии.

WITH source_rows AS (
    SELECT
        pilot_id,
        entity,
        RTRIM(RTRIM(group_name, REPLACE(group_name, '_', '')), '_') AS experiment,
        CASE WHEN group_name LIKE '%\_control' ESCAPE '\' THEN 1 ELSE 0 END AS is_control,
        sms_cum,
        email_cum,
        voice_cum,
        mail_cum,
        call_robot,
        call_oper,
        call_incon,
        call_any,
        cvisit_cum,
        chat_cum,
        od_fact,
        b10_start
    FROM your_table
    WHERE group_name LIKE '%\_control' ESCAPE '\'
       OR group_name LIKE '%\_pilot' ESCAPE '\'
),
flag_rows AS (
    SELECT pilot_id, experiment, is_control, entity, 'sms_cum' AS metric_name, sms_cum AS flag
    FROM source_rows
    UNION ALL
    SELECT pilot_id, experiment, is_control, entity, 'email_cum', email_cum
    FROM source_rows
    UNION ALL
    SELECT pilot_id, experiment, is_control, entity, 'voice_cum', voice_cum
    FROM source_rows
    UNION ALL
    SELECT pilot_id, experiment, is_control, entity, 'mail_cum', mail_cum
    FROM source_rows
    UNION ALL
    SELECT pilot_id, experiment, is_control, entity, 'call_robot', call_robot
    FROM source_rows
    UNION ALL
    SELECT pilot_id, experiment, is_control, entity, 'call_oper', call_oper
    FROM source_rows
    UNION ALL
    SELECT pilot_id, experiment, is_control, entity, 'call_incon', call_incon
    FROM source_rows
    UNION ALL
    SELECT pilot_id, experiment, is_control, entity, 'call_any', call_any
    FROM source_rows
    UNION ALL
    SELECT pilot_id, experiment, is_control, entity, 'cvisit_cum', cvisit_cum
    FROM source_rows
    UNION ALL
    SELECT pilot_id, experiment, is_control, entity, 'chat_cum', chat_cum
    FROM source_rows
),
metric_rows AS (
    SELECT pilot_id, experiment, is_control, entity, metric_name,
           CASE WHEN flag IN (0, 1, '0', '1') THEN CAST(flag AS REAL) END AS x,
           1.0 AS y
    FROM flag_rows
    UNION ALL
    SELECT pilot_id, experiment, is_control, entity, 'recovery_rate',
           CASE WHEN b10_start IS NOT NULL THEN od_fact * 1.0 END,
           CASE WHEN od_fact IS NOT NULL THEN b10_start * 1.0 END
    FROM source_rows
),
entity_totals AS (
    SELECT pilot_id, experiment, is_control, entity, metric_name,
           CASE WHEN COUNT(x) = COUNT(y) THEN SUM(x) END AS x,
           SUM(y) AS y
    FROM metric_rows
    GROUP BY pilot_id, experiment, is_control, entity, metric_name
),
pooled AS (
    SELECT *,
           SUM(x) OVER pair / NULLIF(SUM(y) OVER pair, 0) AS k
    FROM entity_totals
    WINDOW pair AS (PARTITION BY pilot_id, experiment, metric_name)
),
linearized AS (
    SELECT pilot_id, experiment, is_control, metric_name, x, y,
           x - k * y AS l
    FROM pooled
),
centered AS (
    SELECT *, AVG(l) OVER (PARTITION BY pilot_id, experiment, metric_name, is_control) AS mean_l
    FROM linearized
),
group_stats AS (
    SELECT pilot_id, experiment, is_control, metric_name,
           COUNT(l) AS n,
           CASE WHEN COUNT(x) = COUNT(y)
                THEN SUM(x) / NULLIF(SUM(y), 0)
           END AS metric_value,
           AVG(y) AS mean_y,
           SUM((l - mean_l) * (l - mean_l)) / NULLIF(COUNT(l) - 1, 0) AS var_l
    FROM centered
    GROUP BY pilot_id, experiment, is_control, metric_name
),
comparisons AS (
    SELECT t.pilot_id, t.experiment, t.metric_name,
           CASE WHEN t.n >= 2 AND c.n >= 2 AND t.mean_y <> 0 AND c.mean_y <> 0
                THEN (t.metric_value - c.metric_value) / NULLIF(SQRT(
                         t.var_l / (t.n * t.mean_y * t.mean_y)
                         + c.var_l / (c.n * c.mean_y * c.mean_y)
                     ), 0)
           END AS z
    FROM group_stats t
    LEFT JOIN group_stats c
      ON c.pilot_id = t.pilot_id AND c.experiment = t.experiment
     AND c.metric_name = t.metric_name AND c.is_control = 1
    WHERE t.is_control = 0
),
normal_tail AS (
    SELECT *, 1.0 / (1.0 + 0.2316419 * ABS(z)) AS a
    FROM comparisons
),
p_values AS (
    SELECT pilot_id, experiment, metric_name,
           CASE WHEN z = 0 THEN 1.0
                ELSE MIN(1.0, MAX(0.0,
                    2 * 0.3989422804 * EXP(-z * z / 2)
                      * a * (0.319381530 + a * (-0.356563782
                      + a * (1.781477937 + a * (-1.821255978 + a * 1.330274429))))
                ))
           END AS p_value
    FROM normal_tail
)
SELECT
    metric_name,
    p_value,
    CASE WHEN p_value IS NOT NULL THEN p_value < 0.05 END AS is_significant
FROM p_values
ORDER BY pilot_id, experiment, metric_name;
