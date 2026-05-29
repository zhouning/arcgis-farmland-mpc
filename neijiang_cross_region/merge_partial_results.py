"""Merge partial MPC eval results from 3 logs into final 5-seed json.

Sources:
  - seed 0,1 from eval_partial.log.paused_20260510 (UTF-8)
  - seed 2,3 from eval_partial_seeds234.log.paused_20260511 (UTF-8)
  - seed 4   from eval_partial_seed4.log (UTF-16-LE, PowerShell redirect)
"""
import re
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
LOG_S01 = ROOT / "eval_partial.log.paused_20260510"
LOG_S23 = ROOT / "eval_partial_seeds234.log.paused_20260511"
LOG_S4 = ROOT / "eval_partial_seed4.log"
OUT_JSON = ROOT / "5seed_multiobj_results_partial.json"

LINE_RE = re.compile(
    r"seed=(?P<seed>\d+) ep=(?P<ep>\d+): "
    r"slope=(?P<slope>[-+\d.]+)% "
    r"cont_delta=(?P<cd>[-+\d.]+) "
    r"\((?P<cp>[-+\d.]+)%\) "
    r"baimu_count(?P<bc>[-+]\d+) "
    r"area(?P<ba>[-+\d.]+)ha "
    r"reward=(?P<rw>[-+\d.]+) "
    r"time=(?P<t>[\d.]+)s"
)


def read_text_auto(path: Path) -> str:
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8", errors="replace")


def parse_log(path: Path, target_seeds: set[int]) -> dict[int, dict]:
    text = read_text_auto(path)
    out: dict[int, dict] = {
        s: {'slopes': [], 'rewards': [], 'times': [],
            'cont_deltas': [], 'cont_pcts': [],
            'baimu_count_deltas': [], 'baimu_area_deltas_ha': []}
        for s in target_seeds
    }
    for m in LINE_RE.finditer(text):
        seed = int(m['seed'])
        if seed not in target_seeds:
            continue
        d = out[seed]
        d['slopes'].append(float(m['slope']))
        d['rewards'].append(float(m['rw']))
        d['times'].append(float(m['t']))
        d['cont_deltas'].append(float(m['cd']))
        d['cont_pcts'].append(float(m['cp']))
        d['baimu_count_deltas'].append(int(m['bc']))
        d['baimu_area_deltas_ha'].append(float(m['ba']))
    return out


def cross_seed_stats(per_seed: dict[str, dict]) -> dict:
    seeds = sorted(per_seed.keys(), key=int)
    slope = [float(np.mean(per_seed[s]['slopes'])) for s in seeds]
    cp = [float(np.mean(per_seed[s]['cont_pcts'])) for s in seeds]
    cd = [float(np.mean(per_seed[s]['cont_deltas'])) for s in seeds]
    bc = [float(np.mean(per_seed[s]['baimu_count_deltas'])) for s in seeds]
    ba = [float(np.mean(per_seed[s]['baimu_area_deltas_ha'])) for s in seeds]
    rw = [float(np.mean(per_seed[s]['rewards'])) for s in seeds]

    def _ms(arr):
        a = np.array(arr, dtype=float)
        return float(a.mean()), float(a.std())

    sm, ss = _ms(slope)
    cpm, cps = _ms(cp)
    cdm, cds = _ms(cd)
    bcm, bcs = _ms(bc)
    bam, bas = _ms(ba)
    rwm, rws = _ms(rw)
    return {
        'slope_pct_mean': sm, 'slope_pct_std': ss,
        'cont_pct_mean': cpm, 'cont_pct_std': cps,
        'cont_raw_delta_mean': cdm, 'cont_raw_delta_std': cds,
        'baimu_count_delta_mean': bcm, 'baimu_count_delta_std': bcs,
        'baimu_area_delta_ha_mean': bam, 'baimu_area_delta_ha_std': bas,
        'reward_mean': rwm, 'reward_std': rws,
    }


def main():
    for p in (LOG_S01, LOG_S23, LOG_S4):
        if not p.exists():
            sys.exit(f"missing {p}")

    sources = [
        (LOG_S01, {0, 1}),
        (LOG_S23, {2, 3}),
        (LOG_S4, {4}),
    ]
    per_seed: dict[str, dict] = {}
    for path, seeds in sources:
        parsed = parse_log(path, seeds)
        for s in seeds:
            n = len(parsed[s]['slopes'])
            if n != 5:
                sys.exit(f"seed {s} from {path.name}: parsed {n} eps, expected 5")
            per_seed[str(s)] = parsed[s]

    if set(per_seed.keys()) != {'0', '1', '2', '3', '4'}:
        sys.exit(f"seed keys {sorted(per_seed.keys())} != 0..4")

    merged = {
        'region': 'Neijiang Dongxing',
        'mode': 'partial',
        'lambda_rank': 5.0,
        'n_seeds': 5,
        'eval_episodes_per_seed': 5,
        'per_seed': {s: per_seed[s] for s in sorted(per_seed, key=int)},
        'cross_seed': cross_seed_stats(per_seed),
        'bishan_reference_slope': -1.289,
        'bishan_reference_slope_std': 0.079,
        'merged_from': {
            'seed_0_1': LOG_S01.name,
            'seed_2_3': LOG_S23.name,
            'seed_4': LOG_S4.name,
        },
    }
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, default=str)
    print(f"Wrote {OUT_JSON}")
    cs = merged['cross_seed']
    print(f"  slope%      = {cs['slope_pct_mean']:+.4f} ± {cs['slope_pct_std']:.4f}")
    print(f"  cont_delta  = {cs['cont_raw_delta_mean']:+.4f} ± {cs['cont_raw_delta_std']:.4f}")
    print(f"  cont_pct    = {cs['cont_pct_mean']:+.4f}% ± {cs['cont_pct_std']:.4f}")
    print(f"  baimu_#     = {cs['baimu_count_delta_mean']:+.2f} ± {cs['baimu_count_delta_std']:.2f}")
    print(f"  baimu_ha    = {cs['baimu_area_delta_ha_mean']:+.2f} ± {cs['baimu_area_delta_ha_std']:.2f}")
    print(f"  reward      = {cs['reward_mean']:+.2f} ± {cs['reward_std']:.2f}")


if __name__ == '__main__':
    main()
