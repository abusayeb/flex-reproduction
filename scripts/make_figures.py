# The three report figures, drawn from the files in results/.
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from flex import irl
from flex.config import DELTA, DEMO_SEED_OFFSET, N_DEMOS, REPORT_DIR, RESULTS_DIR
from flex.editing import METHODS, Context
from flex.features import feature_table
from flex.llm import make_llm
from flex.scenarios import all_scenarios
from flex.viz import GOAL_COLOR, draw_layout
from flex.world import goal_index, start_index

BLUE, ORANGE, AQUA, YELLOW, MUTED = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#666666'
MODELS = [('ollama_qwen2.5_7b-instruct', 'Qwen 2.5 7B', BLUE),
          ('ollama_qwen2.5_14b-instruct', 'Qwen 2.5 14B', AQUA),
          ('ollama_gemma4_26b', 'Gemma 4 26B', ORANGE)]
GPT4O_PAPER = [8.6, 5.7, 3.8]

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#888888'})

def load(name):
    return json.loads((RESULTS_DIR / name).read_text())

def fig_example():
    # ALGAE vs FLEX on kitchen Gen. Only, on the layout where ALGAE does worst
    rows = load('rq1_oracle_t0.0_d10.json')
    algae_rows = [r for r in rows if r['world'] == 'kitchen' and r['scenario'] == 'Gen. Only'
                  and r['method'] == 'ALGAE']
    worst = algae_rows[0]
    for r in algae_rows:
        if r['score'] < worst['score']:
            worst = r
    seed = worst['seed']

    scenario = [s for s in all_scenarios(['kitchen']) if s.name == 'Gen. Only'][0]
    env, new_env = scenario.env(seed), scenario.new_env(seed)
    start, goal = start_index(env), goal_index(env)
    new_start, new_goal = start_index(new_env), goal_index(new_env)
    reward, new_reward = scenario.true_reward(env), scenario.true_reward(new_env)
    demos = irl.sample_demos(reward, start, goal, N_DEMOS, np.random.default_rng(DEMO_SEED_OFFSET + seed))
    llm = make_llm('oracle')

    train_paths = [(irl.best_path(reward, start, goal)[0], 'true best', GOAL_COLOR, '-')]
    new_paths = [(irl.best_path(new_reward, new_start, new_goal)[0], 'true best', GOAL_COLOR, '-')]
    for method, color in [('ALGAE', BLUE), ('FLEX', ORANGE)]:
        features = METHODS[method](scenario.initial_features(), Context(env, llm, scenario, demos))
        table = feature_table(features, env)
        weights, _ = irl.learn_weights(table, demos, start, goal)
        path, _ = irl.best_path(table @ weights, start, goal)
        new_path, _ = irl.best_path(feature_table(features, new_env) @ weights, new_start, new_goal)
        score = irl.normalized_reward(reward, path, start, goal)
        new_score = irl.normalized_reward(new_reward, new_path, new_start, new_goal)
        train_paths.append((path, f'{method}  ({score:.2f})', color, '--'))
        new_paths.append((new_path, f'{method}  ({new_score:.2f})', color, '--'))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9))
    draw_layout(axes[0], env, train_paths, 'Training layout', scenario.hazard)
    draw_layout(axes[1], new_env, new_paths, 'New layout: toaster replaces kettle', scenario.hazard)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / 'fig_example.png', dpi=220, bbox_inches='tight')

def fig_op_errors():
    edits = ['add', 'generalize', 'mask']
    labels, rates, colors = [], [], []
    for tag, label, color in MODELS:
        rows = load(f'op_errors_{tag}_t0.0_s5-6-7.json')
        labels.append(label)
        rates.append([100 * np.mean([r['error'] for r in rows if r['op'] == e]) for e in edits])
        colors.append(color)
    labels.append('GPT-4o (paper)')
    rates.append(GPT4O_PAPER)
    colors.append(YELLOW)

    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    width, x = 0.2, np.arange(3)
    for i in range(len(labels)):
        xs = x + (i - 1.5) * width
        ax.bar(xs, rates[i], width * 0.9, color=colors[i], label=labels[i], zorder=3)
        for bar_x, rate in zip(xs, rates[i]):
            text = f'{rate:.0f}' if rate >= 1 or rate == 0 else f'{rate:.1f}'
            ax.text(bar_x, rate + 1.2, text, ha='center', fontsize=7)
    ax.set_xticks(x, ['Add', 'Generalize', 'Mask'])
    ax.set_ylabel('error rate (%)')
    ax.set_ylim(0, 72)
    ax.grid(axis='y', color='#e5e5e5', zorder=0)
    ax.legend(ncol=4, fontsize=7.5, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    fig.savefig(REPORT_DIR / 'fig_op_errors.png', dpi=220, bbox_inches='tight')

def fig_delta():
    # wrong decisions and questions asked as delta changes
    rows = load('ext_oracle_overgeneralize_t0.0_d10.json')
    gaps, should_keep = [], []
    for r in rows:
        if r['method'] == 'FLEX-V':
            for check in r['verify']:
                gaps.append(check['ll_general'] - check['ll_specific'])
                should_keep.append(r['kind'] == 'standard')  # generalizing is right only in standard scenarios
    gaps, should_keep = np.array(gaps), np.array(should_keep)

    deltas = np.linspace(0, 12, 121)
    asked, wrong = [], []
    for delta in deltas:
        ask = np.abs(gaps) < delta
        decided_wrong = (np.abs(gaps) >= delta) & ((gaps > 0) != should_keep)
        asked.append(100 * ask.mean())
        wrong.append(100 * decided_wrong.mean())

    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    ax.plot(deltas, wrong, color=ORANGE, lw=2, label='wrong decisions')
    ax.plot(deltas, asked, color=BLUE, lw=2, label='questions asked')
    ax.axvline(DELTA, color=MUTED, lw=1, ls=':')
    ax.text(DELTA + 0.15, 55, f'δ = {DELTA:g} (used)', fontsize=7.5, color=MUTED)
    ax.set_xlabel('δ, BIC margin in nats (decide from demos when |Δ| ≥ δ, else ask)')
    ax.set_ylabel('% of Generalize checks')
    ax.set_ylim(0, 100)
    ax.grid(color='#eeeeee')
    ax.legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(0.02, 1.0))
    fig.tight_layout()
    fig.savefig(REPORT_DIR / 'fig_delta.png', dpi=220, bbox_inches='tight')

def main():
    REPORT_DIR.mkdir(exist_ok=True)
    fig_example()
    fig_op_errors()
    fig_delta()
