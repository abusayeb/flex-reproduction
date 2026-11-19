# Table 1: FLEX against the baselines on the seven scenarios.
import time

from flex.config import N_DEMOS
from flex.editing import METHODS
from flex.experiment import file_tag, markdown_table, mean_and_error, run_one, save_results
from flex.llm import make_llm
from flex.scenarios import SCENARIOS, all_scenarios

# the paper's Table 1, one value per scenario
PAPER = {
    'ALGAE': [1.00, .59, .95, .77, .59, .63, .69],
    'MFE': [.95, .70, 1.00, .75, .43, .67, .83],
    'USF': [.64, .72, .90, .80, .99, .78, .78],
    'FLEX': [1.00, .91, 1.00, 1.00, 1.00, 1.00, .99],
}


def results_table(rows, worlds, key='score'):
    groups = {}
    for r in rows:
        if r['world'] in worlds:
            group = (r['method'], r['scenario'])
            if group not in groups:
                groups[group] = []
            groups[group].append(r[key])
    body = []
    for method in METHODS:
        body.append([method] + [mean_and_error(groups.get((method, s), [])) for s in SCENARIOS])
    return markdown_table(['Method'] + list(SCENARIOS), body)


def main(llm='oracle', seeds=5, demos=N_DEMOS, worlds=('kitchen', 'lavaland'), temperature=0.0):
    spec, llm = llm, make_llm(llm, temperature)
    t0 = time.time()

    rows = []
    for scenario in all_scenarios(worlds):
        for seed in range(seeds):
            for method_name, method in METHODS.items():
                rows.append(run_one(scenario, seed, method, method_name, llm, demos))

    new_calls = 0
    for _, cached in llm.calls:
        if not cached:
            new_calls += 1
    fallbacks = sum(r['fallbacks'] for r in rows)
    dropped = sum(r['dropped'] for r in rows)
    paper = markdown_table(['Method'] + list(SCENARIOS),
                           [[m] + [f'{v:.2f}' for v in values] for m, values in PAPER.items()])

    md = f'# RQ1, LLM `{spec}`, T={temperature}, {seeds} seeds, {demos} demos\n'
    md += '\n## Both worlds\n\n' + results_table(rows, worlds)
    for world in worlds:
        md += f'\n## {world}\n\n' + results_table(rows, [world])
    md += '\n## Transfer to a new layout with an unseen hazard\n\n' + results_table(rows, worlds, 'transfer')
    md += '\n## Paper (Table 1)\n\n' + paper
    md += (f'\nLLM calls: {new_calls} new, {len(llm.calls) - new_calls} cached. '
           f'Code fallbacks: {fallbacks}. Dropped features: {dropped}. Runtime {time.time() - t0:.0f}s.\n')

    save_results(f'rq1_{file_tag(spec, temperature)}_d{demos}', rows, md)
    return rows, llm
