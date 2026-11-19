# Error rate of each edit on its own (Fig. 2).
from flex.editing import Context, add, generalize, mask
from flex.experiment import file_tag, markdown_table, save_results
from flex.features import property_of
from flex.llm import make_llm
from flex.scenarios import all_scenarios

EDITS = {'add': add, 'generalize': generalize, 'mask': mask}
PAPER = {'add': '8.6%', 'generalize': '5.7%', 'mask': '3.8%'}
DEFAULT_SEEDS = list(range(5, 20))


def properties(features, env):
    return sorted(str(property_of(f, env)) for f in features)


def expected(edit, scenario, before):
    # the properties after a correct edit
    after = list(before)
    if edit == 'add' and 'add' in scenario.ops:
        after.append(scenario.mentioned)
    if edit == 'generalize' and 'generalize' in scenario.ops:
        after = [scenario.hazard if p == scenario.mentioned else p for p in after]
    if edit == 'mask' and 'mask' in scenario.ops:
        after = [p for p in after if p != scenario.distractor]
    return sorted(after)


def is_contradictory(edit, scenario, context):
    # Add says NO NEW FEATURE but names the missing object in its reason
    if edit != 'add' or 'add' not in scenario.ops:
        return False
    for entry in context.log:
        if entry['tag'] == 'add':
            reply = entry['reply']
            return 'NO NEW FEATURE' in reply.upper() and scenario.mentioned in reply.lower()
    return False


def run_trial(edit, scenario, seed, llm):
    env = scenario.env(seed)
    context = Context(env, llm, scenario)
    features = scenario.initial_features()
    before = properties(features, env)
    try:
        got = properties(EDITS[edit](features, context), env)
    except Exception as e:
        got = [f'ERROR {e!r}'[:80]]
    want = expected(edit, scenario, before)
    return {'op': edit, 'scenario': scenario.name, 'seed': seed, 'init': before, 'got': got, 'want': want,
            'error': got != want, 'contradictory': is_contradictory(edit, scenario, context)}


def main(llm='oracle', seeds=DEFAULT_SEEDS, temperature=0.0):
    spec, llm, seeds = llm, make_llm(llm, temperature), list(seeds)

    rows = []
    for edit in EDITS:
        for scenario in all_scenarios(['kitchen']):
            for seed in seeds:
                rows.append(run_trial(edit, scenario, seed, llm))

    body = []
    for edit in EDITS:
        trials = [r for r in rows if r['op'] == edit]
        errors = sum(r['error'] for r in trials)
        body.append([edit, f'{errors}/{len(trials)}', f'{errors / len(trials):.1%}', PAPER[edit]])

    # count each kind of mistake, most common first
    counts = {}
    for r in rows:
        if r['error']:
            kind = (r['op'], r['scenario'], tuple(r['got']), tuple(r['want']))
            counts[kind] = counts.get(kind, 0) + 1
    common = sorted(counts.items(), key=lambda item: -item[1])[:12]

    md = f'# Per-operation error rates, LLM `{spec}`, T={temperature}\n\n'
    md += markdown_table(['Operation', 'Errors', 'Rate', 'Paper (GPT-4o)'], body)
    md += f"\nContradictory Add replies: {sum(r['contradictory'] for r in rows)}\n"
    md += '\nMost common errors (op, scenario, got, expected): count\n'
    for kind, count in common:
        md += f'- {kind}: {count}\n'

    name = f'op_errors_{file_tag(spec, temperature)}'
    if seeds != DEFAULT_SEEDS:
        name += '_s' + '-'.join(str(s) for s in seeds)
    save_results(name, rows, md)
    return rows, llm
