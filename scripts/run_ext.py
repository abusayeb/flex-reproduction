# Table 2: FLEX against FLEX-V on standard and personal-preference scenarios.
import numpy as np
from flex.config import N_DEMOS
from flex.editing import flex
from flex.experiment import file_tag, markdown_table, mean_and_error, run_one, save_results
from flex.flexv import flex_v
from flex.llm import make_llm
from flex.scenarios import all_scenarios, personal_scenarios

METHODS = {'FLEX': flex, 'FLEX-V': flex_v}
GENERAL_PROPERTIES = ('heat_source', 'dangerous_terrain')

def kept_general_feature(row):
    for name in row['features']:
        for prop in GENERAL_PROPERTIES:
            if prop in name:
                return True
    return False

def summary_row(kind, method, rows):
    kept = np.mean([kept_general_feature(r) for r in rows])
    asked = np.mean([r['asked'] for r in rows])
    return [kind, method, mean_and_error([r['score'] for r in rows]), mean_and_error([r['transfer'] for r in rows]),
            f'{asked:.2f}', f'{kept:.0%}']

def main(llm='oracle', seeds=5, demos=N_DEMOS, kinds=('standard', 'personal'), temperature=0.0):
    spec, llm, kinds = llm, make_llm(llm, temperature), list(kinds)
    scenarios = []
    if 'standard' in kinds:
        scenarios += [('standard', s) for s in all_scenarios() if 'generalize' in s.ops]
    if 'personal' in kinds:
        scenarios += [('personal', s) for s in personal_scenarios()]

    rows = []
    for kind, scenario in scenarios:
        for seed in range(seeds):
            for method_name, method in METHODS.items():
                r = run_one(scenario, seed, method, method_name, llm, demos)
                r['kind'] = kind
                r['verify'] = [e for e in r['log'] if e['tag'] == 'verify']
                r['asked'] = sum(v['decision'] == 'asked' for v in r['verify'])
                rows.append(r)

    groups = {}
    for r in rows:
        group = (r['kind'], r['method'])
        if group not in groups:
            groups[group] = []
        groups[group].append(r)
    header = ['Scenarios', 'Method', 'Train R̂', 'Transfer R̂', 'Questions / run', 'Generalised feature kept']
    body = [summary_row(kind, method, groups[(kind, method)]) for kind, method in sorted(groups)]
    md = f'# Extension, LLM `{spec}`, T={temperature}, {seeds} seeds, {demos} demos\n\n'
    md += markdown_table(header, body)

    name = f'ext_{file_tag(spec, temperature)}_d{demos}'
    if kinds != ['standard', 'personal']:
        name += '_' + '_'.join(kinds)
    save_results(name, rows, md)
    return rows, llm
