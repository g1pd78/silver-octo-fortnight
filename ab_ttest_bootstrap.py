"""Verify ab_ttest.sql with a cluster bootstrap.

Install numpy, then run:
    python ab_ttest_bootstrap.py ab_platform_operreport_example.csv > bootstrap.csv

The CSV output has the same columns as the SQL. Undefined results are blank.
Each resample draws entities within each arm, keeping all their days together.
"""

import argparse
import csv
from pathlib import Path
import sys

import numpy as np


FLAGS = [
    "sms_cum", "email_cum", "voice_cum", "mail_cum", "call_robot",
    "call_oper", "call_incon", "call_any", "cvisit_cum", "chat_cum",
]
RECOVERY_COLUMNS = ["od_fact", "b10_start"]
METRICS = sorted([*FLAGS, "recovery_rate"])
FLAG_VALUES = {"0": 0, "1": 1, "0.0": 0, "1.0": 1}
FLAG_POSITIONS = {name: METRICS.index(name) for name in FLAGS}
RECOVERY_POSITION = METRICS.index("recovery_rate")
OUTPUT_COLUMNS = ("metric_name", "p_value", "is_significant")
REQUIRED_COLUMNS = {"pilot_id", "group_name", "entity", *FLAGS, *RECOVERY_COLUMNS}


def read_totals(input_csv, encoding):
    experiments = {}
    with input_csv.open(encoding=encoding, newline="") as source:
        header = source.readline()
        if not header:
            raise ValueError("input CSV is empty")
        dialect = csv.Sniffer().sniff(header, delimiters=",;\t")
        source.seek(0)
        rows = csv.DictReader(source, dialect=dialect)
        missing = REQUIRED_COLUMNS.difference(rows.fieldnames or [])
        if missing:
            raise ValueError("missing CSV columns: " + ", ".join(sorted(missing)))

        for row in rows:
            if None in row or None in row.values():
                raise ValueError(f"wrong number of fields on CSV line {rows.line_num}")
            experiment, separator, arm = row["group_name"].rpartition("_")
            arm = arm.lower()
            if not separator or arm not in ("pilot", "control"):
                continue
            pilot_id, entity = row["pilot_id"], row["entity"]
            if not pilot_id or not entity:
                raise ValueError(f"missing pilot_id or entity on CSV line {rows.line_num}")
            if pilot_id.isdecimal():
                pilot_id = int(pilot_id)
            arms = experiments.setdefault((pilot_id, experiment.rstrip("_")), {})
            entities = arms.setdefault(arm, {})
            if entity not in entities:
                entities[entity] = np.zeros((len(METRICS), 2))
                entities[entity][RECOVERY_POSITION] = np.nan
            totals = entities[entity]

            for metric, index in FLAG_POSITIONS.items():
                flag = row[metric].strip()
                totals[index, 0] += FLAG_VALUES.get(flag, np.nan)
                totals[index, 1] += 1

            fact, balance = (row[name].strip() for name in RECOVERY_COLUMNS)
            if fact and balance:
                recovery = totals[RECOVERY_POSITION]
                amounts = [float(value.replace(",", ".")) for value in (fact, balance)]
                if not np.isfinite(amounts).all():
                    raise ValueError(f"nonfinite recovery amount on CSV line {rows.line_num}")
                if np.isnan(recovery[0]):
                    recovery[:] = 0
                recovery += amounts
    return experiments


def bootstrap_ratios(residuals, denominators, resamples, rng):
    results = np.full(resamples, np.nan)
    n = len(residuals)
    batch_size = max(1, min(256, 1_000_000 // n))
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        indices = rng.integers(n, size=(stop - start, n))
        sampled_x = residuals[indices].sum(axis=1)
        sampled_y = denominators[indices].sum(axis=1)
        np.divide(sampled_x, sampled_y, out=results[start:stop], where=sampled_y != 0)
    return results


def bootstrap_p_value(pilot, control, resamples, rng):
    if pilot is None or control is None or min(len(pilot), len(control)) < 2:
        return None

    if not np.isfinite(pilot).all() or not np.isfinite(control).all():
        return None
    x_t, y_t = pilot.T
    x_c, y_c = control.T
    sum_y_t, sum_y_c = y_t.sum(), y_c.sum()
    if sum_y_t == 0 or sum_y_c == 0:
        return None

    ratio_t, ratio_c = x_t.sum() / sum_y_t, x_c.sum() / sum_y_c
    observed = abs(ratio_t - ratio_c)
    # Center each arm at its observed ratio to simulate the null of equal ratios.
    residuals_t = x_t - ratio_t * y_t
    residuals_c = x_c - ratio_c * y_c
    if np.ptp(residuals_t) == 0 and np.ptp(residuals_c) == 0:
        return None

    null_diffs = bootstrap_ratios(residuals_t, y_t, resamples, rng)
    null_diffs -= bootstrap_ratios(residuals_c, y_c, resamples, rng)
    valid = np.isfinite(null_diffs)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return None
    extreme = int(np.count_nonzero(np.abs(null_diffs[valid]) >= observed))
    return (extreme + 1) / (n_valid + 1)


def compare(experiments, resamples, seed):
    rng = np.random.default_rng(seed)
    keys = sorted(experiments, key=lambda key: (isinstance(key[0], str), *key))
    for key in keys:
        arms = experiments[key]
        if "pilot" not in arms:
            continue
        pilot = np.stack(list(arms["pilot"].values()))
        control = np.stack(list(arms["control"].values())) if "control" in arms else None
        for index, metric in enumerate(METRICS):
            pilot_totals = pilot[:, index]
            control_totals = control[:, index] if control is not None else None
            if metric == "recovery_rate" and control_totals is not None:
                pilot_totals = pilot_totals[~np.isnan(pilot_totals[:, 1])]
                control_totals = control_totals[~np.isnan(control_totals[:, 1])]
            p_value = bootstrap_p_value(pilot_totals, control_totals, resamples, rng)
            yield metric, p_value, None if p_value is None else int(p_value < 0.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoding", default="utf-8-sig")
    args = parser.parse_args()
    if args.resamples < 1 or args.seed < 0:
        parser.error("--resamples must be positive and --seed must be nonnegative")

    try:
        experiments = read_totals(args.input_csv, args.encoding)
        writer = csv.writer(sys.stdout, lineterminator="\n")
        writer.writerow(OUTPUT_COLUMNS)
        writer.writerows(compare(experiments, args.resamples, args.seed))
    except (csv.Error, OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
