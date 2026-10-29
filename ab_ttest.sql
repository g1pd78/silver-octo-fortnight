-- =====================================================================
-- A/B-тест в Superset: t-тест с линеаризацией, MDE, мощность,
-- несколько пилотов, guardrail-метрики (без FDR)
--
-- СУБД: SQLite 3.25+ (нужны оконные функции и функция VARIANCE).
-- Используется как virtual dataset в Superset (нужен ENABLE_TEMPLATE_PROCESSING).
--
-- Наблюдение = строка таблицы (без свёртки по entity).
-- Контроль = группа 'control', все остальные группы = пилоты.
-- Метрики вида SUM(числитель) / SUM(знаменатель).
--
-- Общая логика:
--   1. Для каждой метрики берём числитель x и знаменатель y по строкам.
--   2. Линеаризуем: L = x − κ·y, где κ — значение метрики в контроле.
--      Это превращает отношение сумм в обычное среднее, не меняя метрику.
--   3. К L применяем t-тест Уэлча: шум, t-статистика, p-value, интервал.
--   4. Переводим результат обратно в единицы метрики (п.п., рубли, %).
--   5. Добавляем MDE, мощность, поправку Бонферрони на число пилотов
--      и статус гардрейлов.
-- =====================================================================

WITH

-- ---------------------------------------------------------------------
-- 0. СПРАВОЧНИК МЕТРИК
-- Для каждой метрики: какая у неё роль и как её читать.
--   role        primary   — по ней принимаем решение
--               secondary — смотрим, но решение по ней не принимаем
--               guardrail — не должна ухудшиться
--   direction    1 — рост метрики хорошо, −1 — рост плохо
--   margin      для гардрейлов: допустимое относительное ухудшение (0.02 = 2%)
--   target_lift ожидаемый эффект для расчёта мощности;
--               для гардрейла обычно равен margin
-- Метрика без строки здесь считается secondary с direction = 1.
-- ---------------------------------------------------------------------
metric_dict AS (
    SELECT 'mail_cum' AS metric_name, 'primary'   AS role,  1 AS direction, NULL AS margin, 0.03 AS target_lift
    UNION ALL
    SELECT 'recovery_rate',                'secondary',          1,              NULL,           0.03
    UNION ALL
    SELECT 'email_cum',               'guardrail',          1,              0.02,           0.02
),

-- ---------------------------------------------------------------------
-- 1. ФИЛЬТРЫ ДАШБОРДА
-- Применяем фильтры по продукту и бакету ДО любых расчётов.
-- Иначе κ и L посчитались бы по всем данным, а не по выбранному срезу.
-- remove_filter=True говорит Superset: фильтр применён здесь,
-- снаружи к результату его повторно не добавлять.
-- Если в фильтре ничего не выбрано, условие не добавляется — берём всё.
-- ---------------------------------------------------------------------
filtered AS (
    SELECT *
    FROM your_table
    WHERE 1 = 1                                   -- заглушка, чтобы дальше все условия начинались с AND
    {% if filter_values('product', remove_filter=True) %}
      AND product IN {{ filter_values('product', remove_filter=True) | where_in }}
    {% endif %}
    {% if filter_values('bucket', remove_filter=True) %}
      AND bucket IN {{ filter_values('bucket', remove_filter=True) | where_in }}
    {% endif %}
),

-- ---------------------------------------------------------------------
-- 2. ЧИСЛИТЕЛЬ И ЗНАМЕНАТЕЛЬ КАЖДОЙ МЕТРИКИ
-- Переводим таблицу в длинный формат: одна строка = (метрика, группа, x, y).
-- Метрика группы потом считается как SUM(x) / SUM(y) — ровно как в
-- текущих отчётах. Новая метрика добавляется ещё одним блоком UNION ALL
-- и строкой в metric_dict.
-- * 1.0 — перевод в дробное число: в SQLite 3 / 20 = 0, а 3.0 / 20 = 0.15.
-- COALESCE(…, 0) — пустое значение считаем нулём, иначе NULL выпадет из сумм.
-- ---------------------------------------------------------------------
step1_rows AS (
    -- покрытие mail: доля дней с отметкой
    SELECT
        'mail_cum'                  AS metric_name,
        group_name,
        mail_cum  * 1.0                  AS x,       -- числитель: отметка 1/0
        acc_count * 1.0                  AS y        -- знаменатель
    FROM filtered

    UNION ALL

    -- recovery rate: сколько ОД собрано от ОД на старте
    SELECT
        'recovery_rate'                  AS metric_name,
        group_name,
        COALESCE(od_fact, 0)  * 1.0      AS x,
        COALESCE(od_start, 0) * 1.0      AS y
    FROM filtered

    UNION ALL

    -- покрытие email (гардрейл): доля дней с отметкой email
    SELECT
        'email_cum'                 AS metric_name,
        group_name,
        email_cum * 1.0                  AS x,       -- числитель: отметка 1/0
        acc_count * 1.0                  AS y        -- знаменатель
    FROM filtered
),

-- ---------------------------------------------------------------------
-- 3. κ (КАППА) — «НОРМА» КАЖДОЙ МЕТРИКИ
-- κ = значение метрики в контроле = ΣX / ΣY по контролю.
-- Это общая линейка для всех групп: с ней сравнивается каждая строка.
-- NULLIF(…, 0): если знаменатель 0, получаем NULL вместо ошибки деления.
-- ---------------------------------------------------------------------
step2_kappa AS (
    SELECT
        metric_name,
        SUM(x) / NULLIF(SUM(y), 0) AS k
    FROM step1_rows
    WHERE group_name = 'control'
    GROUP BY metric_name
),

-- ---------------------------------------------------------------------
-- 4. ЛИНЕАРИЗАЦИЯ — ГЛАВНЫЙ ШАГ
-- L = x − κ·y = фактический числитель − ожидаемый при уровне контроля.
-- Пример: κ = 0.12, y = 20, x = 3 → ожидалось 2.4, факт 3, L = +0.6.
--
-- Почему это работает:
--   среднее L = среднее x − κ·среднее y
--             = среднее y · (ΣX/ΣY − κ)
--             = среднее y · (метрика группы − κ)
--   В контроле метрика = κ, поэтому среднее L = 0.
--   В пилоте среднее L = среднее y · (метрика пилота − метрика контроля).
-- Значит, «отличаются ли метрики» = «отличаются ли средние L»,
-- а для средних есть готовая формула шума (t-тест).
-- JOIN по metric_name: каждая метрика линеаризуется своим κ.
-- ---------------------------------------------------------------------
step3_lin AS (
    SELECT
        e.metric_name,
        e.group_name,
        e.x,
        e.y,
        k.k,
        e.x - k.k * e.y AS l
    FROM step1_rows e
    JOIN step2_kappa k
      ON k.metric_name = e.metric_name
),

-- ---------------------------------------------------------------------
-- 5. СТАТИСТИКИ ПО КАЖДОЙ МЕТРИКЕ И ГРУППЕ
-- Всё, что нужно t-тесту, плюс значение метрики для отображения.
-- ---------------------------------------------------------------------
step4_grp AS (
    SELECT
        metric_name,
        group_name,
        COUNT(*)                    AS n,          -- число наблюдений
        AVG(l)                      AS mean_l,     -- среднее L (в контроле ≈ 0)
        VARIANCE(l)                 AS var_l,      -- разброс L: Σ(L − среднее)² / (n − 1)
        AVG(y)                      AS mean_y,     -- средний знаменатель: нужен для перевода обратно
        SUM(x) / NULLIF(SUM(y), 0)  AS metric_val, -- значение метрики, как в текущих отчётах
        MAX(k)                      AS k           -- κ одинаковый во всех строках, MAX просто достаёт его
    FROM step3_lin
    GROUP BY metric_name, group_name
),

-- ---------------------------------------------------------------------
-- 6. КАЖДЫЙ ПИЛОТ В ПАРЕ С КОНТРОЛЕМ + ШУМ РАЗНИЦЫ
-- Self-join: строки пилотов (t) соединяем со строкой контроля (c)
-- той же метрики. Получаем одну строку на пару «метрика × пилот».
--
-- diff_l — разница средних L (сигнал).
-- se_l   — шум этой разницы, формула t-теста Уэлча:
--          √(дисперсия_пилота / n_пилота + дисперсия_контроля / n_контроля)
--          var / n — шум среднего: падает с ростом выборки;
--          шумы двух групп складываются, потому что группы независимы;
--          Уэлч не требует одинакового разброса в группах.
-- ---------------------------------------------------------------------
step5_t AS (
    SELECT
        t.metric_name,
        t.group_name                                AS pilot,
        c.n                                       AS n_c,
        t.n                                       AS n_t,
        c.metric_val                              AS metric_c,
        t.metric_val                              AS metric_t,
        c.mean_l                                  AS mean_l_c,
        t.mean_l                                  AS mean_l_t,
        c.var_l                                   AS var_l_c,
        t.var_l                                   AS var_l_t,
        t.mean_y                                  AS mean_y_t,
        c.k                                       AS k,
        t.mean_l - c.mean_l                       AS diff_l,
        SQRT(t.var_l / t.n + c.var_l / c.n)       AS se_l
    FROM step4_grp t
    JOIN step4_grp c
      ON c.metric_name = t.metric_name
     AND c.group_name    = 'control'
    WHERE t.group_name <> 'control'
),

-- ---------------------------------------------------------------------
-- 7. t-СТАТИСТИКА И ВСПОМОГАТЕЛЬНЫЕ ВЕЛИЧИНЫ
-- t_stat   = diff_l / se_l: во сколько раз эффект больше шума.
-- n_pilots — сколько пилотов сравнивается с контролем по этой метрике
--            (нужно для поправки Бонферрони).
-- a        — промежуточная величина для формулы p-value
--            (приближение Abramowitz–Stegun 26.2.17, см. шаг 9).
-- z_pw     — насколько ожидаемый эффект target_lift заходит за порог
--            значимости 1.96, в единицах шума (для мощности).
--            target_lift · mean_y_t · κ — это target_lift в «единицах L».
-- LEFT JOIN со справочником: метрика без описания всё равно считается.
-- ---------------------------------------------------------------------
step6_aux AS (
    SELECT
        s.*,
        COALESCE(d.role, 'secondary')                           AS role,
        COALESCE(d.direction, 1)                                AS direction,
        d.margin,
        COALESCE(d.target_lift, 0.03)                           AS target_lift,
        COUNT(*) OVER (PARTITION BY s.metric_name)              AS n_pilots,
        diff_l / NULLIF(se_l, 0)                                AS t_stat,
        1.0 / (1.0 + 0.2316419 * ABS(diff_l / NULLIF(se_l, 0))) AS a,
        ABS(COALESCE(d.target_lift, 0.03)) * mean_y_t * k / NULLIF(se_l, 0) - 1.96 AS z_pw
    FROM step5_t s
    LEFT JOIN metric_dict d
      ON d.metric_name = s.metric_name
),

-- ---------------------------------------------------------------------
-- 8. ХВОСТ НОРМАЛЬНОГО РАСПРЕДЕЛЕНИЯ ДЛЯ МОЩНОСТИ
-- tail_pw = площадь хвоста нормального распределения за |z_pw|.
-- Та же аппроксимация, что для p-value. В SQLite нет встроенной
-- функции нормального распределения, поэтому используем многочлен:
--   0.3989422804 = 1/√(2π) — высота «колокола» в нуле;
--   EXP(−z²/2)   — форма колокола;
--   0.2316419 и 0.319381530 … 1.330274429 — табличные коэффициенты,
--   ошибка меньше 0.0000001.
-- Множитель 1/(1 + 0.2316419·|z|) расписан заново в каждом слагаемом,
-- потому что SQLite не даёт сослаться на колонку того же SELECT.
-- ---------------------------------------------------------------------
step7_pw AS (
    SELECT
        *,
        0.3989422804 * EXP(-z_pw * z_pw / 2)
          * (1.0 / (1.0 + 0.2316419 * ABS(z_pw)))
          * (0.319381530
          + (1.0 / (1.0 + 0.2316419 * ABS(z_pw))) * (-0.356563782
          + (1.0 / (1.0 + 0.2316419 * ABS(z_pw))) * (1.781477937
          + (1.0 / (1.0 + 0.2316419 * ABS(z_pw))) * (-1.821255978
          + (1.0 / (1.0 + 0.2316419 * ABS(z_pw))) * 1.330274429))))   AS tail_pw
    FROM step6_aux
),

-- ---------------------------------------------------------------------
-- 9. ЭФФЕКТЫ, ИНТЕРВАЛЫ, p-VALUE, MDE, МОЩНОСТЬ
--
-- Перевод из «единиц L» в единицы метрики:
--   diff_l = mean_y_t · (metric_t − metric_c), поэтому
--   абсолютный эффект = diff_l / mean_y_t            (п.п., рубли)
--   lift              = diff_l / (mean_y_t · κ)      (% к контролю)
--   Границы интервала переводятся тем же делением.
--
-- 1.96 — 95% значений нормального распределения лежат в пределах ±1.96 шума:
--   интервал = эффект ± 1.96 · шум; |t| > 1.96 ⇔ p < 0.05.
-- p_value = 2 · площадь хвоста за |t| (двусторонний тест: эффект
--   может быть и в плюс, и в минус).
-- MDE = 2.8 · шум: 1.96 (порог 5%) + 0.84 (мощность 80%).
--   Эффекты меньше MDE тест, скорее всего, не заметит.
-- power — вероятность заметить эффект размера target_lift.
-- n_*_needed — сколько наблюдений нужно, чтобы MDE = target_lift:
--   шум падает как 1/√n, поэтому n растёт как (MDE / target)².
-- ---------------------------------------------------------------------
step8_res AS (
    SELECT
        metric_name,
        role,
        direction,
        margin,
        pilot,
        n_pilots,
        n_c,
        n_t,
        metric_c,
        metric_t,
        diff_l / mean_y_t                               AS abs_diff,
        (diff_l - 1.96 * se_l) / mean_y_t               AS abs_ci_low,
        (diff_l + 1.96 * se_l) / mean_y_t               AS abs_ci_high,
        diff_l / (mean_y_t * k)                         AS lift,
        (diff_l - 1.96 * se_l) / (mean_y_t * k)         AS lift_ci_low,
        (diff_l + 1.96 * se_l) / (mean_y_t * k)         AS lift_ci_high,
        t_stat,
        2 * 0.3989422804 * EXP(-t_stat * t_stat / 2)
          * a * (0.319381530
          + a * (-0.356563782
          + a * (1.781477937
          + a * (-1.821255978
          + a * 1.330274429))))                         AS p_value,
        CASE WHEN ABS(t_stat) > 1.96 THEN 1 ELSE 0 END  AS is_significant,

        2.8 * se_l / mean_y_t                           AS mde_abs,
        2.8 * se_l / (mean_y_t * k)                     AS mde_rel,
        target_lift,
        CASE WHEN z_pw >= 0 THEN 1 - tail_pw ELSE tail_pw END  AS power,
        CAST(n_c * (2.8 * se_l / (mean_y_t * k) / ABS(target_lift))
                 * (2.8 * se_l / (mean_y_t * k) / ABS(target_lift)) AS INTEGER) AS n_c_needed,
        CAST(n_t * (2.8 * se_l / (mean_y_t * k) / ABS(target_lift))
                 * (2.8 * se_l / (mean_y_t * k) / ABS(target_lift)) AS INTEGER) AS n_t_needed
    FROM step7_pw
),

-- ---------------------------------------------------------------------
-- 10. ФЛАГИ: ПОПРАВКА НА ПИЛОТЫ, НАПРАВЛЕНИЕ ЭФФЕКТА, ГАРДРЕЙЛЫ
--
-- Бонферрони: порог 0.05 делим на число пилотов. При 3 пилотах — 0.0167.
--   Без поправки шанс случайного прокраса хотя бы одного пилота ≈ 14%.
--
-- «Хорошее направление»: умножаем эффект на direction, чтобы плюс
--   всегда значил «лучше». Для direction = −1 границы интервала
--   меняются местами и знак переворачивается.
--
-- Статус гардрейла (по худшей правдоподобной границе):
--   нарушен             — весь интервал в плохой стороне: ухудшение значимо;
--   ок                  — худшая граница не хуже допуска margin
--                         (без margin: просто нет значимого ухудшения);
--   недостаточно данных — интервал слишком широкий, ухудшение больше
--                         допуска исключить нельзя.
-- ---------------------------------------------------------------------
step9_flags AS (
    SELECT
        *,
        0.05 / n_pilots                                           AS alpha_adj,
        CASE WHEN p_value < 0.05 / n_pilots THEN 1 ELSE 0 END     AS is_significant_adj,

        direction * lift                                          AS lift_good,
        CASE WHEN direction = 1 THEN lift_ci_low  ELSE -lift_ci_high END AS lift_good_ci_low,
        CASE WHEN direction = 1 THEN lift_ci_high ELSE -lift_ci_low  END AS lift_good_ci_high,

        CASE
            WHEN p_value >= 0.05 / n_pilots THEN 'нет различий'
            WHEN direction * lift > 0        THEN 'лучше'
            ELSE                                  'хуже'
        END                                                       AS effect_verdict,

        CASE
            WHEN role <> 'guardrail' THEN NULL
            WHEN (CASE WHEN direction = 1 THEN lift_ci_high ELSE -lift_ci_low END) < 0
                THEN 'нарушен'
            WHEN margin IS NULL
                THEN 'ок'
            WHEN (CASE WHEN direction = 1 THEN lift_ci_low ELSE -lift_ci_high END) > -margin
                THEN 'ок'
            ELSE 'недостаточно данных'
        END                                                       AS guardrail_status
    FROM step8_res
)

-- ---------------------------------------------------------------------
-- 11. РЕЗУЛЬТАТ
-- Колонки для дашборда плюс сводка по пилоту: сколько гардрейлов у него
-- нарушено и сколько под вопросом (оконная функция по pilot — одно и то же
-- число во всех строках пилота).
-- product и bucket — какие фильтры выбраны на дашборде (для подписи).
-- Решение: для primary и secondary — effect_verdict (с поправкой Бонферрони
-- на число пилотов), для гардрейлов — guardrail_status.
-- Сортировка: по пилоту, внутри — primary, гардрейлы, остальные.
-- ---------------------------------------------------------------------
SELECT
    metric_name,
    role,
    pilot,
    '{{ filter_values("product") | join(", ") or "все" }}'  AS product,
    '{{ filter_values("bucket")  | join(", ") or "все" }}'  AS bucket,
    n_c,
    n_t,
    metric_c,
    metric_t,
    abs_diff,
    abs_ci_low,
    abs_ci_high,
    lift,
    lift_ci_low,
    lift_ci_high,
    t_stat,
    p_value,
    is_significant,

    n_pilots,
    alpha_adj,
    is_significant_adj,
    effect_verdict,

    lift_good,
    lift_good_ci_low,
    lift_good_ci_high,
    margin,
    guardrail_status,
    SUM(CASE WHEN guardrail_status = 'нарушен' THEN 1 ELSE 0 END)
        OVER (PARTITION BY pilot)                             AS pilot_guardrails_violated,
    SUM(CASE WHEN guardrail_status = 'недостаточно данных' THEN 1 ELSE 0 END)
        OVER (PARTITION BY pilot)                             AS pilot_guardrails_unclear,

    mde_abs,
    mde_rel,
    target_lift,
    power,
    n_c_needed,
    n_t_needed
FROM step9_flags
ORDER BY pilot,
         CASE role WHEN 'primary' THEN 1 WHEN 'guardrail' THEN 2 ELSE 3 END,
         metric_name;
