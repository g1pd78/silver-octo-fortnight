# A/B-тест в Superset: t-тест с линеаризацией, MDE, мощность и бутстрап для сверки

СУБД: SQLite. Наблюдение — строка таблицы (без свёртки по entity).
Группы: `control` и `pilot`. Метрики вида `SUM(числитель) / SUM(знаменатель)`.

Содержание:

1. Основной запрос (t-тест + линеаризация + MDE + мощность) — для virtual dataset в Superset
2. Бутстрап — для сверки вне дашборда
3. Сверка t-теста и бутстрапа
4. Константы

---

## 1. Основной запрос

Используется как virtual dataset в Superset. Фильтры по продукту и бакету подставляются через Jinja до расчёта.
Требуется флаг `ENABLE_TEMPLATE_PROCESSING` в конфиге Superset.

```sql
WITH
params AS (
    SELECT 0.03 AS target_lift        -- ожидаемый относительный эффект
),

filtered AS (
    SELECT *
    FROM your_table
    WHERE 1 = 1
    {% if filter_values('product', remove_filter=True) %}
      AND product IN {{ filter_values('product', remove_filter=True) | where_in }}
    {% endif %}
    {% if filter_values('bucket', remove_filter=True) %}
      AND bucket IN {{ filter_values('bucket', remove_filter=True) | where_in }}
    {% endif %}
),

-- 1. Строки: числитель и знаменатель по блоку на метрику
step1_entity AS (
    SELECT
        'mail_coverage'           AS metric_name,
        ab_group,
        mail_cum  * 1.0           AS x,
        acc_count * 1.0           AS y
    FROM filtered

    UNION ALL

    SELECT
        'recovery_rate'           AS metric_name,
        ab_group,
        COALESCE(od_fact, 0)  * 1.0      AS x,
        COALESCE(od_start, 0) * 1.0      AS y
    FROM filtered

    -- UNION ALL ... следующие метрики по тому же шаблону
),

-- 2. κ для каждой метрики
step2_kappa AS (
    SELECT
        metric_name,
        SUM(x) / NULLIF(SUM(y), 0) AS k
    FROM step1_entity
    WHERE ab_group = 'control'
    GROUP BY metric_name
),

-- 3. Линеаризация: κ своей метрики
step3_lin AS (
    SELECT
        e.metric_name,
        e.ab_group,
        e.x,
        e.y,
        k.k,
        e.x - k.k * e.y AS l
    FROM step1_entity e
    JOIN step2_kappa k
      ON k.metric_name = e.metric_name
),

-- 4. Статистики по метрике и группе
step4_grp AS (
    SELECT
        metric_name,
        ab_group,
        COUNT(*)                    AS n,
        AVG(l)                      AS mean_l,
        VARIANCE(l)                 AS var_l,
        AVG(y)                      AS mean_y,
        SUM(x) / NULLIF(SUM(y), 0)  AS metric_val,
        MAX(k)                      AS k
    FROM step3_lin
    GROUP BY metric_name, ab_group
),

-- 5. Пилот и контроль в одну строку на метрику
step5_t AS (
    SELECT
        *,
        mean_l_t - mean_l_c                   AS diff_l,
        SQRT(var_l_t / n_t + var_l_c / n_c)   AS se_l
    FROM (
        SELECT
            metric_name,
            MAX(CASE WHEN ab_group = 'control' THEN n          END) AS n_c,
            MAX(CASE WHEN ab_group = 'pilot'   THEN n          END) AS n_t,
            MAX(CASE WHEN ab_group = 'control' THEN metric_val END) AS metric_c,
            MAX(CASE WHEN ab_group = 'pilot'   THEN metric_val END) AS metric_t,
            MAX(CASE WHEN ab_group = 'control' THEN mean_l     END) AS mean_l_c,
            MAX(CASE WHEN ab_group = 'pilot'   THEN mean_l     END) AS mean_l_t,
            MAX(CASE WHEN ab_group = 'control' THEN var_l      END) AS var_l_c,
            MAX(CASE WHEN ab_group = 'pilot'   THEN var_l      END) AS var_l_t,
            MAX(CASE WHEN ab_group = 'pilot'   THEN mean_y     END) AS mean_y_t,
            MAX(k)                                                  AS k
        FROM step4_grp
        GROUP BY metric_name
    )
),

-- 6. t-статистика и вспомогательные величины для p-value и мощности
step6_aux AS (
    SELECT
        s.*,
        p.target_lift,
        diff_l / NULLIF(se_l, 0)                                AS t_stat,
        1.0 / (1.0 + 0.2316419 * ABS(diff_l / NULLIF(se_l, 0))) AS a,
        ABS(p.target_lift) * mean_y_t * k / NULLIF(se_l, 0) - 1.96 AS z_pw
    FROM step5_t s
    CROSS JOIN params p
),

-- 7. Площадь хвоста для мощности
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
)

SELECT
    metric_name,
    '{{ filter_values("product") | join(", ") or "все" }}'  AS product,
    '{{ filter_values("bucket")  | join(", ") or "все" }}'  AS bucket,
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

    -- MDE при мощности 80% и уровне 5%
    2.8 * se_l / mean_y_t                           AS mde_abs,
    2.8 * se_l / (mean_y_t * k)                     AS mde_rel,

    -- мощность для ожидаемого эффекта target_lift
    target_lift,
    CASE WHEN z_pw >= 0 THEN 1 - tail_pw ELSE tail_pw END  AS power,

    -- сколько строк нужно, чтобы MDE = target_lift (при тех же пропорциях групп)
    CAST(n_c * (2.8 * se_l / (mean_y_t * k) / ABS(target_lift))
             * (2.8 * se_l / (mean_y_t * k) / ABS(target_lift)) AS INTEGER) AS n_c_needed,
    CAST(n_t * (2.8 * se_l / (mean_y_t * k) / ABS(target_lift))
             * (2.8 * se_l / (mean_y_t * k) / ABS(target_lift)) AS INTEGER) AS n_t_needed
FROM step7_pw
ORDER BY metric_name;
```

### Выходные колонки

| Колонка | Смысл |
|---|---|
| `metric_name` | Метрика |
| `product`, `bucket` | Выбранные на дашборде фильтры (`все`, если не выбраны) |
| `n_c`, `n_t` | Число строк в контроле и пилоте |
| `metric_c`, `metric_t` | Значение метрики в контроле и пилоте |
| `abs_diff` | Абсолютный эффект: `metric_t − metric_c` |
| `abs_ci_low`, `abs_ci_high` | 95% доверительный интервал абсолютного эффекта |
| `lift` | Относительный эффект: `(metric_t − metric_c) / metric_c` |
| `lift_ci_low`, `lift_ci_high` | 95% доверительный интервал относительного эффекта |
| `t_stat` | Во сколько раз эффект больше шума |
| `p_value` | Двусторонний p-value |
| `is_significant` | 1, если `|t| > 1.96` (эквивалентно `p_value < 0.05`) |
| `mde_abs`, `mde_rel` | Минимальный эффект, который тест замечает с мощностью 80% |
| `target_lift` | Ожидаемый эффект из `params` |
| `power` | Вероятность заметить эффект размера `target_lift` |
| `n_c_needed`, `n_t_needed` | Сколько строк нужно, чтобы MDE сравнялся с `target_lift` |

### Пошаговая проверка

Замените финальный `SELECT` на просмотр нужного шага:

| Запрос | Что должно быть |
|---|---|
| `SELECT metric_name, ab_group, SUM(x) / SUM(y) FROM step1_entity GROUP BY 1, 2;` | Значения метрик совпадают с текущими расчётами |
| `SELECT * FROM step2_kappa;` | `k` равно значению метрики в контроле |
| `SELECT metric_name, ab_group, AVG(l) FROM step3_lin GROUP BY 1, 2;` | В контроле ≈ 0 |
| `SELECT * FROM step4_grp;` | По две строки на метрику, `var_l > 0` |
| `SELECT metric_name, metric_t - metric_c, diff_l / mean_y_t FROM step5_t;` | Два числа совпадают |

### Замечания

- `target_lift` сейчас один на все метрики. Когда появится справочник метрик, `params` заменяется на джойн к нему по `metric_name`.
- `n_c`, `n_t`, `n_c_needed`, `n_t_needed` — в строках.
- Не подставляйте в `target_lift` фактически полученный `lift`: «наблюдаемая мощность» однозначно выражается через p-value и новой информации не даёт.

---

## 2. Бутстрап для сверки

Пуассоновский бутстрап: в каждой итерации каждая строка получает случайный вес 0, 1, 2… (распределение Пуассона со средним 1), метрика считается как взвешенная `SUM(w·x) / SUM(w·y)`.

Запускается **в одной сессии** SQLite-клиента (sqlite3, DB Browser и т.п.): временные таблицы живут до закрытия соединения. В SQL Lab Superset каждый запрос может идти в новом соединении, и временные таблицы между запросами пропадут.

Перед запуском оцените объём: строк в `boot_u` будет `число строк × число метрик × число итераций`.

```sql
SELECT COUNT(*) FROM your_table WHERE ab_group IN ('control', 'pilot');
```

Если выходит больше 50–100 млн, уменьшите число итераций.

### B0. Исходные строки

```sql
DROP TABLE IF EXISTS boot_rows;

CREATE TEMP TABLE boot_rows AS
SELECT 'mail_coverage' AS metric_name, ab_group,
       mail_cum  * 1.0 AS x,
       acc_count * 1.0 AS y
FROM your_table
WHERE ab_group IN ('control', 'pilot')
  -- AND product IN (...)
  -- AND bucket IN (...)

UNION ALL

SELECT 'recovery_rate', ab_group,
       COALESCE(od_fact, 0)  * 1.0,
       COALESCE(od_start, 0) * 1.0
FROM your_table
WHERE ab_group IN ('control', 'pilot');
  -- AND product IN (...)
  -- AND bucket IN (...)
```

Фильтры должны совпадать с теми, что выбраны на дашборде при сверке.

### B1. Итерации

```sql
DROP TABLE IF EXISTS iters;

CREATE TEMP TABLE iters AS
WITH RECURSIVE s(it) AS (
    SELECT 1
    UNION ALL
    SELECT it + 1 FROM s WHERE it < 200
)
SELECT it FROM s;
```

### B2. Случайные числа (фиксируем в таблице)

```sql
DROP TABLE IF EXISTS boot_u;

CREATE TEMP TABLE boot_u AS
SELECT
    i.it,
    r.metric_name,
    r.ab_group,
    r.x,
    r.y,
    RANDOM() / 18446744073709551616.0 + 0.5 AS u     -- равномерное от 0 до 1
FROM boot_rows r
CROSS JOIN iters i;
```

Случайные числа сохраняются в таблицу, потому что иначе оптимизатор SQLite может пересчитать `RANDOM()` в каждом сравнении `CASE` на следующем шаге и испортить веса.

### B3. Веса и разница метрик в каждой итерации

```sql
DROP TABLE IF EXISTS boot_diff;

CREATE TEMP TABLE boot_diff AS
SELECT
    metric_name,
    it,
    SUM(CASE WHEN ab_group = 'pilot'   THEN w * x END) * 1.0
      / NULLIF(SUM(CASE WHEN ab_group = 'pilot'   THEN w * y END), 0)
  - SUM(CASE WHEN ab_group = 'control' THEN w * x END) * 1.0
      / NULLIF(SUM(CASE WHEN ab_group = 'control' THEN w * y END), 0) AS diff
FROM (
    SELECT
        metric_name, it, ab_group, x, y,
        CASE                                   -- Пуассон(1) через равномерное u
            WHEN u < 0.367879 THEN 0
            WHEN u < 0.735759 THEN 1
            WHEN u < 0.919699 THEN 2
            WHEN u < 0.981012 THEN 3
            WHEN u < 0.996340 THEN 4
            WHEN u < 0.999406 THEN 5
            ELSE 6
        END AS w
    FROM boot_u
)
GROUP BY metric_name, it;
```

Проверка: по одной строке на метрику и итерацию, среднее `diff` близко к `metric_t − metric_c`.

```sql
SELECT metric_name, COUNT(*), AVG(diff), MIN(diff), MAX(diff)
FROM boot_diff
GROUP BY metric_name;
```

### B4. Итоги бутстрапа

```sql
DROP TABLE IF EXISTS boot_stats;

CREATE TEMP TABLE boot_stats AS
WITH ranked AS (
    SELECT
        metric_name,
        diff,
        ROW_NUMBER() OVER (PARTITION BY metric_name ORDER BY diff) AS rn,
        COUNT(*)     OVER (PARTITION BY metric_name)               AS n
    FROM boot_diff
)
SELECT
    metric_name,
    MAX(n)                                                                AS n_iter,
    AVG(diff)                                                             AS boot_mean_diff,
    SQRT(VARIANCE(diff))                                                  AS boot_se,
    MAX(CASE WHEN rn = MAX(CAST(0.025 * n AS INTEGER), 1) THEN diff END) AS boot_ci_low,
    MAX(CASE WHEN rn = CAST(0.975 * n AS INTEGER)         THEN diff END) AS boot_ci_high,
    2 * MIN(
          AVG(CASE WHEN diff <= 0 THEN 1.0 ELSE 0.0 END),
          AVG(CASE WHEN diff >= 0 THEN 1.0 ELSE 0.0 END)
        )                                                                 AS boot_p_value
FROM ranked
GROUP BY metric_name;
```

- `boot_se` — разброс разниц по итерациям, то есть шум, измеренный напрямую.
- `boot_ci_low`, `boot_ci_high` — перцентильный 95% интервал: отрезаем по 2,5% итераций с каждой стороны.
- `boot_p_value` — доля итераций по другую сторону от нуля, умноженная на 2 (двусторонний тест).
- `MAX(a, b)` и `MIN(a, b)` с двумя аргументами — обычный выбор большего/меньшего числа, а не агрегаты.

---

## 3. Сверка t-теста и бутстрапа

Выполняется в той же сессии после B0–B4. t-тест с линеаризацией считается по тем же строкам `boot_rows`, результаты ставятся рядом с бутстрапом.

```sql
WITH
k AS (
    SELECT metric_name, SUM(x) / NULLIF(SUM(y), 0) AS k
    FROM boot_rows
    WHERE ab_group = 'control'
    GROUP BY metric_name
),
lin AS (
    SELECT r.metric_name, r.ab_group, r.x, r.y, r.x - k.k * r.y AS l
    FROM boot_rows r
    JOIN k ON k.metric_name = r.metric_name
),
g AS (
    SELECT metric_name, ab_group,
           COUNT(*)    AS n,
           AVG(l)      AS mean_l,
           VARIANCE(l) AS var_l,
           AVG(y)      AS mean_y,
           SUM(x) / NULLIF(SUM(y), 0) AS metric_val
    FROM lin
    GROUP BY metric_name, ab_group
),
p AS (
    SELECT
        metric_name,
        MAX(CASE WHEN ab_group = 'control' THEN n          END) AS n_c,
        MAX(CASE WHEN ab_group = 'pilot'   THEN n          END) AS n_t,
        MAX(CASE WHEN ab_group = 'control' THEN metric_val END) AS metric_c,
        MAX(CASE WHEN ab_group = 'pilot'   THEN metric_val END) AS metric_t,
        MAX(CASE WHEN ab_group = 'control' THEN mean_l     END) AS mean_l_c,
        MAX(CASE WHEN ab_group = 'pilot'   THEN mean_l     END) AS mean_l_t,
        MAX(CASE WHEN ab_group = 'control' THEN var_l      END) AS var_l_c,
        MAX(CASE WHEN ab_group = 'pilot'   THEN var_l      END) AS var_l_t,
        MAX(CASE WHEN ab_group = 'pilot'   THEN mean_y     END) AS mean_y_t
    FROM g
    GROUP BY metric_name
),
t AS (
    SELECT
        *,
        mean_l_t - mean_l_c                   AS diff_l,
        SQRT(var_l_t / n_t + var_l_c / n_c)   AS se_l
    FROM p
)
SELECT
    t.metric_name,
    t.metric_t - t.metric_c                         AS diff,

    t.se_l / t.mean_y_t                             AS lin_se,
    b.boot_se,
    (t.se_l / t.mean_y_t) / NULLIF(b.boot_se, 0)    AS se_ratio,       -- должно быть ≈ 1

    (t.diff_l - 1.96 * t.se_l) / t.mean_y_t         AS lin_ci_low,
    b.boot_ci_low,
    (t.diff_l + 1.96 * t.se_l) / t.mean_y_t         AS lin_ci_high,
    b.boot_ci_high,

    b.boot_p_value,
    b.n_iter
FROM t
JOIN boot_stats b ON b.metric_name = t.metric_name
ORDER BY t.metric_name;
```

### Как читать сверку

- **`se_ratio` ≈ 1** (в пределах 0,9–1,1 при 200 итерациях) — формула шума t-теста посчитана верно.
- **Границы интервалов** близки друг к другу. Небольшие расхождения нормальны: бутстрап случайный и слегка меняется от запуска к запуску.
- **p-value** сравнивайте на незначимых и пограничных метриках. При 200 итерациях шаг p-value бутстрапа 2/200 = 0,01, поэтому сильный эффект он покажет как 0.
- **Если сильно расходятся** для денежной метрики (recovery rate): вероятно, тяжёлые хвосты — несколько очень больших значений. Бутстрап покажет несимметричный интервал. Нужна обрезка экстремальных значений.
- Оба расчёта предполагают независимость строк. Их совпадение подтверждает правильность формул, но не проверяет это предположение.

---

## 4. Константы

### t-тест, MDE, мощность

| Константа | Что это | Зачем |
|---|---|---|
| `* 1.0` | Перевод в дробное число | В SQLite `3 / 20 = 0`, а `3.0 / 20 = 0.15` |
| `NULLIF(…, 0)` | Замена нуля на NULL | Пустой знаменатель даёт NULL, а не ошибку |
| `1.96` | 95% значений нормального распределения в пределах ±1.96 | Уровень уверенности 95% для интервала и флага |
| `2 *` в p-value | Двусторонний тест | Эффект может быть и в плюс, и в минус |
| `0.3989422804` | 1 / √(2π) | Высота нормального распределения в нуле |
| `EXP(-t*t/2)` | Форма нормального распределения | Как быстро падает вероятность больших отклонений |
| `0.2316419`, `0.319381530`, `−0.356563782`, `1.781477937`, `−1.821255978`, `1.330274429` | Коэффициенты приближения хвоста нормального распределения (Abramowitz & Stegun, 26.2.17), ошибка < 1e-7 | Замена встроенной функции нормального распределения, которой нет в SQLite |
| `0.84` | Точка, левее которой 80% нормального распределения | Мощность 80% |
| `2.8` | 1.96 + 0.84 | Порог значимости плюс запас на мощность. Для мощности 90% — `3.24` |
| `0.03` | Ожидаемый эффект (`target_lift`) | Задаётся бизнесом |

### Бутстрап

| Константа | Что это | Зачем |
|---|---|---|
| `200` | Число итераций | Компромисс скорости и точности в SQLite |
| `18446744073709551616`, `+ 0.5` | 2⁶⁴ и сдвиг | Перевод `RANDOM()` (от −2⁶³ до 2⁶³) в число от 0 до 1 |
| `0.367879`, `0.735759`, `0.919699`, `0.981012`, `0.996340`, `0.999406` | Накопленные вероятности Пуассона(1): P(0) = e⁻¹, P(0)+P(1), … | Сколько раз строка «попала» в переигровку. Веса больше 6 (вероятность ~0.00008) объединены в 6 |
| `0.025`, `0.975` | Края 95% интервала | Тот же уровень 95%, что и 1.96 в t-тесте |

---

## Что подставить

- `your_table` — имя таблицы
- `ab_group`, `'control'`, `'pilot'` — колонка группы и её значения
- `product`, `bucket` — колонки фильтров
- `mail_cum`, `acc_count`, `od_fact`, `od_start` — числители и знаменатели метрик
- Новые метрики добавляются блоком `UNION ALL` в `step1_entity` (основной запрос) и в `boot_rows` (бутстрап)
