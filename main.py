# Reruns the experiments in the report and checks every number against it.
#   python main.py            everything, about a minute
#   python main.py table1     one part: basics, table1, op_errors, table2 or figures
import sys
import time

import numpy as np

from flex import irl
from flex.config import DEMO_SEED_OFFSET, GOAL_WEIGHT, HAZARD_WEIGHT, N_DEMOS
from flex.features import distance_feature, feature_table, goal_feature
from flex.scenarios import SCENARIOS, all_scenarios
from flex.world import goal_index, start_index
from scripts import make_figures, run_ext, run_op_errors, run_rq1

MISMATCHES = []
LLMS = []


def as_text(values, fmt):
    text = ''
    for value in values:
        text += f'{fmt.format(value):<12}'
    return text


def check(label, got, want, fmt='{:.2f}', paper=''):
    # compare with the number in the report, as printed
    got_text, want_text = as_text(got, fmt), as_text(want, fmt)
    if paper:
        paper = f'{paper:<12}'
    if got_text == want_text:
        print(f'  {label:<28}{paper}{got_text}ok')
    else:
        print(f'  {label:<28}{paper}{got_text}MISMATCH, report: {want_text}')
        MISMATCHES.append(label)


def expect(label, value, ok, rule):
    # for numbers that only need to fall in a range
    result = 'ok' if ok else 'FAILED'
    print(f'  {label:<38}{value:<12}{result:<8}{rule}')
    if not ok:
        MISMATCHES.append(label)


def header(*columns):
    print(f'  {"":<28}' + as_text(columns, '{}'))


def mean_scores(rows, method, kind=None):
    # mean score on the training layout and on the new layout
    train, new = [], []
    for r in rows:
        if r['method'] == method and (kind is None or r['kind'] == kind):
            train.append(r['score'])
            new.append(r['transfer'])
    return np.mean(train), np.mean(new)


def error_rates(rows):
    rates = []
    for edit in ['add', 'generalize', 'mask']:
        errors = [r['error'] for r in rows if r['op'] == edit]
        rates.append(100 * np.mean(errors))
    return rates


def basics():
    # IRL alone: with the right features the robot should be perfect, worse without them
    demo_scores, ratios = [], []
    scores_by_features = {'right': [], 'goal only': [], 'stove': []}
    for scenario in all_scenarios():
        if scenario.name != 'Gen. Only':
            continue
        feature_sets = {
            'right': [goal_feature(), distance_feature(scenario.truth)],
            'goal only': [goal_feature()],
            'stove': [goal_feature(), distance_feature(scenario.mentioned)],
        }
        for seed in range(5):
            env = scenario.env(seed)
            start, goal = start_index(env), goal_index(env)
            reward = scenario.true_reward(env)
            demos = irl.sample_demos(reward, start, goal, N_DEMOS, np.random.default_rng(DEMO_SEED_OFFSET + seed))
            for demo in demos:
                demo_scores.append(irl.normalized_reward(reward, demo, start, goal))
            for key, features in feature_sets.items():
                table = feature_table(features, env)
                weights, _ = irl.learn_weights(table, demos, start, goal)
                path, _ = irl.best_path(table @ weights, start, goal)
                scores_by_features[key].append(irl.normalized_reward(reward, path, start, goal))
                if key == 'right':
                    ratios.append(weights[1] / -weights[0])

    demo_score = np.mean(demo_scores)
    ratio = np.mean(ratios)
    right = np.mean(scores_by_features['right'])
    goal_only = np.mean(scores_by_features['goal only'])
    stove = np.mean(scores_by_features['stove'])
    true_ratio = HAZARD_WEIGHT / -GOAL_WEIGHT
    expect('Demo score', f'{demo_score:.2f}', 0.9 < demo_score < 1, 'expected 0.9-1')
    expect('Learned weights, heat source : goal', f'{ratio:.1f} : 1', abs(ratio - true_ratio) < 1,
           f'true {true_ratio:.0f} : 1')
    expect('Score, goal + heat source', f'{right:.2f}', right > 0.995, 'expected 1.00')
    expect('Score, goal + stove', f'{stove:.2f}', goal_only < stove < right, 'expected in between')
    expect('Score, goal only', f'{goal_only:.2f}', goal_only < 0.8, 'expected < 0.8')


def table1():
    header('paper', 'train', 'new layout')
    rows, _ = run_rq1.main(llm='oracle', demos=10)
    report = {'ALGAE': (0.94, 0.86), 'MFE': (1.00, 0.86), 'USF': (1.00, 1.00), 'FLEX': (1.00, 0.94)}
    for method, want in report.items():
        paper = np.mean(run_rq1.PAPER[method])
        check(method, mean_scores(rows, method), want, paper=f'{paper:.2f}')
    rows_3_demos, _ = run_rq1.main(llm='oracle', demos=3)
    check('FLEX, 3 demos', mean_scores(rows_3_demos, 'FLEX'), (1.00, 0.93), paper='-')

    # Generalize only matters in the scenarios that need it
    print('  Scenarios with Generalize, new layout:')
    for method, want in [('ALGAE', 0.86), ('FLEX', 0.99)]:
        new_scores = []
        for r in rows:
            edits_needed = SCENARIOS[r['scenario']][0]
            if r['method'] == method and 'generalize' in edits_needed:
                new_scores.append(r['transfer'])
        check(f'  {method}', [np.mean(new_scores)], [want])


def op_errors():
    header('add', 'generalize', 'mask')
    rows, _ = run_op_errors.main(llm='oracle')
    check('Oracle', error_rates(rows), (0, 0, 0), '{:.0f}%')
    models = [('ollama:qwen2.5:7b-instruct', 'Qwen 2.5 7B', (57, 29, 57)),
              ('ollama:qwen2.5:14b-instruct', 'Qwen 2.5 14B', (57, 62, 57)),
              ('ollama:gemma4:26b', 'Gemma 4 26B', (0, 14, 14))]
    for spec, label, want in models:
        rows, llm = run_op_errors.main(llm=spec, seeds=[5, 6, 7])
        check(label, error_rates(rows), want, '{:.0f}%')
        LLMS.append(llm)
    gpt = [f'{rate}%' for rate in make_figures.GPT4O_PAPER]
    print(f'  {"GPT-4o (paper)":<28}' + as_text(gpt, '{}'))


def table2():
    header('train', 'new layout')
    rows, _ = run_ext.main(llm='oracle')
    check('Oracle, FLEX', mean_scores(rows, 'FLEX', 'personal'), (1.00, 1.00))
    check('Oracle, FLEX-V', mean_scores(rows, 'FLEX-V', 'personal'), (1.00, 1.00))

    rows, _ = run_ext.main(llm='oracle:overgeneralize')
    check('Over-general oracle, FLEX', mean_scores(rows, 'FLEX', 'personal'), (0.99, 0.94))
    check('Over-general oracle, FLEX-V', mean_scores(rows, 'FLEX-V', 'personal'), (1.00, 1.00))

    rows, llm = run_ext.main(llm='ollama:gemma4:26b', seeds=2, kinds=['personal'])
    check('Gemma 4 26B, FLEX', mean_scores(rows, 'FLEX', 'personal'), (0.88, 0.94))
    check('Gemma 4 26B, FLEX-V', mean_scores(rows, 'FLEX-V', 'personal'), (0.97, 0.94))
    LLMS.append(llm)


def figures():
    make_figures.main()
    print('  Saved report/fig_example.png, fig_op_errors.png, fig_delta.png')


STEPS = {
    'basics': ('0. IRL with fixed features, no LLM (Gen. Only, 2 worlds x 5 layouts)', basics),
    'table1': ('1. Table 1, oracle LLM (7 scenarios x 2 worlds x 5 layouts)', table1),
    'op_errors': ('2. Error rate per operation, Fig. 2 (7 scenarios x 3 layouts = 21 trials)', op_errors),
    'table2': ('3. Table 2, personal preferences: FLEX vs FLEX-V', table2),
    'figures': ('4. Figures', figures),
}


def main():
    steps = sys.argv[1:]
    if not steps:
        steps = list(STEPS)
    for name in steps:
        if name not in STEPS:
            sys.exit(f'unknown step {name!r}, choose from: {", ".join(STEPS)}')

    t0 = time.time()
    print('FLEX reproduction (Lee et al., CoRL 2026), 2D navigation')
    print('Score = normalized reward, 1.00 = best path. "ok" = same as in the report.')
    for name in steps:
        title, run = STEPS[name]
        print(f'\n{title}')
        run()

    print()
    if LLMS:
        calls, cached = 0, 0
        for llm in LLMS:
            for _, from_cache in llm.calls:
                calls += 1
                if from_cache:
                    cached += 1
        print(f'LLM replies: {cached}/{calls} from cache/')
    if MISMATCHES:
        print(f'{len(MISMATCHES)} numbers differ from the report: {", ".join(MISMATCHES)}')
    else:
        print('All numbers match the report.')
    print(f'Done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
