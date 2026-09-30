#!/usr/bin/env python3
"""Display all existing acceptance conventions; never select the closest one."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ORDER = ('gsm8k', 'math500', 'aime25', 'humaneval', 'lcb', 'mtbench', 'alpaca')
PAPER = {
    0: dict(zip(ORDER, (7.36, 8.88, 8.94, 6.90, 7.80, 4.40, 3.69))),
    1: dict(zip(ORDER, (6.82, 7.50, 5.41, 6.30, 7.60, 4.16, 3.56))),
}
METRICS = ('accepted_per_round_pooled', 'acceptance_length_with_bonus_pooled',
           'acceptance_length_with_bonus_request_mean')


def summarize(rows):
    groups = defaultdict(dict)
    for row in rows:
        temp = float(row['temperature'])
        if temp not in PAPER or row['benchmark'] not in ORDER:
            continue
        key = (row['method'], int(temp))
        if row['benchmark'] in groups[key]:
            raise ValueError(f'Duplicate benchmark {key}: {row["benchmark"]}')
        groups[key][row['benchmark']] = row
    report = {'groups': [], 'interpretation': [
        'Paper text defines accepted draft tokens per verification cycle. This is accepted_per_round_pooled.',
        'The public DFlash runtime also exposes committed tokens with a bonus; author ReTrace counter code is unavailable.',
        'Keep the +bonus and request-average variants separate; do not choose a convention by its closeness to Table 2.',
        'Blank comparison_valid/greedy_parity fields mean validation was not supplied, not that it passed.',
        'Length-limited requests should be reported, not dropped to improve acceptance statistics.',
    ]}
    for (method, temperature), data in sorted(groups.items()):
        entries = []
        for benchmark in ORDER:
            if benchmark not in data:
                continue
            row = data[benchmark]
            entries.append({'benchmark': benchmark, 'paper_retrace_tau': PAPER[temperature][benchmark],
                **{name: float(row[name]) if row.get(name) else None for name in METRICS},
                'requests': int(row['requests']),
                'length_limited_requests': int(row.get('length_limited_requests') or 0),
                'comparison_valid': row.get('comparison_valid') or 'NOT PROVIDED',
                'greedy_parity': row.get('greedy_parity') or 'NOT PROVIDED'})
        complete = set(data) == set(ORDER)
        macro = {name: sum(x[name] for x in entries) / len(ORDER)
                 for name in METRICS if complete and all(x[name] is not None for x in entries)}
        report['groups'].append({'method': method, 'temperature': temperature,
            'complete_benchmarks': complete, 'benchmarks': entries, 'macro_average': macro,
            'paper_retrace_macro': 6.85 if temperature == 0 else 5.91})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with args.csv.open(newline='') as f:
        report = summarize(list(csv.DictReader(f)))
    text = json.dumps(report, indent=2, allow_nan=False)
    print(text)
    if args.output:
        with args.output.open('x') as f:
            f.write(text + '\n')


if __name__ == '__main__':
    main()
