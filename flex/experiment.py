import json
import numpy as np
from flex import irl
from flex.config import DEMO_SEED_OFFSET, N_DEMOS, RESULTS_DIR
from flex.editing import Context
from flex.features import feature_table
from flex.world import goal_index, start_index

def run_one(scenario, seed, method, method_name, llm, n_demos=N_DEMOS):
    # demos -> edit the features -> learn weights -> plan -> score on this layout and on a new one
    env = scenario.env(seed)
    start, goal = start_index(env), goal_index(env)
    reward = scenario.true_reward(env)
    demos = irl.sample_demos(reward, start, goal, n_demos, np.random.default_rng(DEMO_SEED_OFFSET + seed))

    context = Context(env, llm, scenario, demos)
    features, error = scenario.initial_features(), None
    try:
        features = method(features, context)
    except Exception as e:
        error = repr(e)[:300]

    table = feature_table(features, env)
    weights, _ = irl.learn_weights(table, demos, start, goal)
    path, _ = irl.best_path(table @ weights, start, goal)
    score = irl.normalized_reward(reward, path, start, goal)

    # same features and weights on a new layout the robot has never seen
    new_env = scenario.new_env(seed)
    new_start, new_goal = start_index(new_env), goal_index(new_env)
    try:
        new_path, _ = irl.best_path(feature_table(features, new_env) @ weights, new_start, new_goal)
        new_score = irl.normalized_reward(scenario.true_reward(new_env), new_path, new_start, new_goal)
    except Exception:
        new_score = 0.0

    fallbacks, dropped = 0, 0
    for entry in context.log:
        if entry['tag'] == 'fallback':
            fallbacks += 1
        if entry['tag'] == 'drop':
            dropped += 1

    return {
        'world': scenario.world, 'scenario': scenario.name, 'seed': seed, 'method': method_name,
        'score': score, 'transfer': new_score,
        'features': [f.name for f in features], 'sources': [f.source for f in features],
        'theta': [round(float(w), 4) for w in weights], 'error': error,
        'fallbacks': fallbacks, 'dropped': dropped, 'log': context.log,
    }

def mean_and_error(values):
    # 'mean ± standard error'
    values = np.array(values, dtype=float)
    if len(values) == 0:
        return '-'
    error = 0.0
    if len(values) > 1:
        error = values.std(ddof=1) / np.sqrt(len(values))
    return f'{values.mean():.2f} ± {error:.2f}'

def markdown_table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    for row in rows:
        lines.append('| ' + ' | '.join(str(x) for x in row) + ' |')
    return '\n'.join(lines) + '\n'

def file_tag(spec, temperature):
    return spec.replace(':', '_').replace('/', '_') + f'_t{temperature}'

def save_results(name, rows, markdown):
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f'{name}.json').write_text(json.dumps(rows, indent=1))
    (RESULTS_DIR / f'{name}.md').write_text(markdown)
