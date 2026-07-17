"""
Generate the PATCHED v2.3.1 reference baseline at the SAME reduced config as
the v3 save scripts, so v3.0 vs v2.3.1 is apples-to-apples (both engines with
the fixed multiscale-CIN-regate behaviour).

Reuses plot_fig56.py (the original v2 script) VERBATIM for make_sim /
make_screening / the partner_count + dwelltime_by_genotype analyzers, pinning
the frozen v2.3.1 engine ahead of any editable install (editable-trap).

Writes, into results/v2.3.1_baseline/:
  - fig5_partners.csv          (india, females, lifetime casual-partner bins)
  - fig5_dwelltime_summary.csv (india age at causal infection / CIN2+ / cancer)
  - fig6_asr.csv               (ASR cancer incidence 2020 by screening coverage)

Config (matches save_fig5/6_baseline_v3.py): n_agents=20k, dt=0.25,
start=1960, stop=2020, ms_agent_ratio=100, seeds [0,1,2],
end_probs [0.0, 0.1, 0.2].

Run with the v2.3.1 reference venv:
  .venv-v2/Scripts/python.exe run_v2ref_reduced.py
"""
import sys
sys.path.insert(0, r'C:\Users\ryanhu\PycharmProjects\hpvsim_v23_frozen')

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import hpvsim as hpv

assert hpv.__version__ == '2.3.1', f'expected frozen 2.3.1, got {hpv.__version__} from {hpv.__file__}'
import plot_fig56 as p  # noqa: E402

LOCATION = 'india'
BINS = np.concatenate([np.arange(21), [100]])


def run_fig5(seeds):
    partner_arrs = []
    age_causal, age_cin, age_cancer = [], [], []
    for s in seeds:
        sim = p.make_sim(location=LOCATION, seed=s, add_analyzers=True)
        sim.run(verbose=0)
        pc = sim.get_analyzer('partner_count')
        partner_arrs.append(np.asarray(pc.partners['f']))
        dt = sim.get_analyzer('dwelltime_by_genotype')
        age_causal += list(dt.age_causal)
        age_cin += list(dt.age_cin)
        age_cancer += list(dt.age_cancer)
        print(f'[fig5] seed {s}: n_females={len(pc.partners["f"])} n_cancers={len(dt.age_cancer)}', flush=True)

    counts = np.concatenate(partner_arrs)
    hist, _ = np.histogram(counts, bins=BINS)
    prob = hist / hist.sum()
    partners_df = pd.DataFrame({'location': LOCATION, 'sex': 'f',
                                'partner_count_bin': np.arange(21), 'probability': prob})

    age_causal = np.array(age_causal); age_cin = np.array(age_cin); age_cancer = np.array(age_cancer)
    rows = []
    for ev, arr in [
        ('Causal HPV infection', age_causal[age_causal < 50]),
        ('CIN2+',                age_cin[age_causal < 65]),
        ('Cancer',               age_cancer[age_causal < 90]),
    ]:
        rows.append(pd.DataFrame({'Age': arr, 'Health event': ev, 'location': LOCATION.capitalize()}))
    dwelltime_df = pd.concat(rows)
    dwell_summary = (dwelltime_df.groupby(['location', 'Health event'])['Age']
                     .describe(percentiles=[0.25, 0.5, 0.75]).reset_index())
    return partners_df, dwell_summary


def run_fig6(seeds, end_probs):
    rows = []
    for end_prob in end_probs:
        per_seed = []
        for s in seeds:
            if end_prob > 0:
                algos = p.make_screening(end_probs=[end_prob])
                interventions = algos[end_prob]
            else:
                interventions = None
            sim = p.make_sim(location=LOCATION, seed=s, interventions=interventions)
            sim.run(verbose=0)
            asr = float(np.asarray(sim.results['asr_cancer_incidence'])[-1])
            per_seed.append(asr)
            print(f'[fig6] end_prob={end_prob} seed={s}: asr2020={asr:.2f}', flush=True)
        per_seed = np.array(per_seed)
        rows.append({
            'scenario': f'Screening {end_prob}',
            'asr_cancer_incidence_2020': float(per_seed.mean()),
            'asr_cancer_incidence_low_2020': float(per_seed.min()),
            'asr_cancer_incidence_high_2020': float(per_seed.max()),
        })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default='results/v2.3.1_baseline')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--end-probs', type=float, nargs='+', default=[0.0, 0.1, 0.2])
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    partners_df, dwell_summary = run_fig5(args.seeds)
    partners_df.to_csv(outdir / 'fig5_partners.csv', index=False)
    dwell_summary.to_csv(outdir / 'fig5_dwelltime_summary.csv', index=False)

    fig6_df = run_fig6(args.seeds, args.end_probs)
    fig6_df.to_csv(outdir / 'fig6_asr.csv', index=False)

    (outdir / 'manifest.json').write_text(json.dumps({
        'figures': ['fig5', 'fig6'],
        'hpvsim_version': hpv.__version__,
        'engine': 'hpvsim_v23_frozen @ fix-multiscale-cin-regate (patched v2.3.1)',
        'seeds': args.seeds, 'end_probs': args.end_probs,
        'config': 'n_agents=20000 dt=0.25 start=1960 stop=2020 ms_agent_ratio=100',
        'date': date.today().isoformat(),
    }, indent=2))

    print(f'\nSaved v2.3.1 reference baseline to {outdir}')
    print(dwell_summary.to_string(index=False))
    print(fig6_df.to_string(index=False))
