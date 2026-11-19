# FLEX's three edits (Add, Generalize, Mask), FLEX itself, and the Table 1 baselines.
import re
from flex import prompts
from flex.features import Feature, feature_per_cell, is_additive, property_of, template_feature
from flex.world import env_description

class Context:
    # everything one editing run needs, plus a log of every LLM call
    def __init__(self, env, llm, scenario, demos=None):
        self.env = env
        self.llm = llm
        self.scenario = scenario
        self.demos = demos
        self.log = []

    def prompt_fields(self):
        return {'env': env_description(self.env), 'instruction': self.scenario.instruction,
                'history': [], 'correction': self.scenario.correction}

    def ask(self, tag, messages, **info):
        info['scenario'] = self.scenario
        messages = [{'role': 'system', 'content': prompts.SYSTEM_2D}] + messages
        reply = self.llm.chat(messages, tag=tag, info=info)
        self.log.append({'tag': tag, 'prompt': messages[-1]['content'][-400:], 'reply': reply})
        return reply

def user(text):
    return {'role': 'user', 'content': text}

def exchange(prompt, reply):
    return [user(prompt), {'role': 'assistant', 'content': reply}]

def names(features):
    return [f.name for f in features]

def clean_name(text):
    # '**Distance to the stove**' -> 'distance_to_the_stove'
    text = text.strip()
    if not text:
        return ''
    text = text.strip('*`"\' ').splitlines()[0]
    text = re.sub(r'^(feature( name)?\s*[:=]\s*|\d+[.)]\s*|-\s*)', '', text, flags=re.I)
    text = re.split(r'[(:]', text)[0]
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')

def make_feature(context, conversation, name):
    # the LLM writes the feature code; if it crashes or isn't additive, use the template for that property
    reply = context.ask('codegen', conversation + [user(prompts.fill(prompts.CODEGEN, feature=name))], feature=name)
    name_match = re.search(r'Feature name:\s*"?([\w\- ]+)"?', reply)
    code_match = re.search(r'```(?:python)?\s*\n(.*?)```', reply, flags=re.S)
    new_name = clean_name(name_match.group(1)) if name_match else name
    feature = Feature(new_name, code_match.group(1) if code_match else '')
    try:
        if not is_additive(feature, context.env, feature_per_cell(feature, context.env)):
            raise ValueError('not additive over timesteps')
        return feature
    except Exception as error:
        backup = template_feature(new_name, context.env)
        if backup is None:
            backup = template_feature(name, context.env)
        if backup:
            context.log.append({'tag': 'fallback', 'feature': new_name, 'reason': repr(error)[:200]})
        else:
            context.log.append({'tag': 'drop', 'feature': name, 'reason': repr(error)[:200]})
        return backup


def one_per_property(features, context):
    # if two features measure the same property, keep the first
    seen, kept = set(), []
    for f in features:
        key = property_of(f, context.env) or f.name
        if key not in seen:
            seen.add(key)
            kept.append(f)
    return kept

def features_from_list(context, prompt, reply):
    # one feature per line of the reply; lines that fail are left out
    features = []
    for line in reply.splitlines():
        name = clean_name(line)
        if name and not name.startswith(('here', 'feature_name', 'the_', 'these')) and len(name) < 60:
            feature = make_feature(context, exchange(prompt, reply), name)
            if feature:
                features.append(feature)
    return features

def add(features, context):
    prompt = prompts.fill(prompts.ADD, features=names(features), **context.prompt_fields())
    reply = context.ask('add', [user(prompt)], features=names(features))
    name = clean_name(reply)
    if 'NO NEW FEATURE' in reply.upper() or not name:
        return features
    feature = make_feature(context, exchange(prompt, reply), name)
    if feature is None:
        return features
    return one_per_property(features + [feature], context)

def read_renames(reply):
    # 'old -> new' lines from the Generalize reply, skipping NO CHANGE
    renames = {}
    for line in reply.splitlines():
        if '->' not in line:
            continue
        old, new = line.split('->', 1)
        old, new = clean_name(old), new.strip().strip('*`"\' ')
        if old and new and 'NO CHANGE' not in new.upper():
            renames[old] = clean_name(new)
    return renames

def generalize(features, context):
    intent = context.ask('intent', [user(prompts.fill(prompts.INTENT, **context.prompt_fields()))])
    prompt = prompts.fill(prompts.GENERALIZE, features=names(features), intent=intent, **context.prompt_fields())
    reply = context.ask('generalize', [user(prompt)], features=names(features))
    renames = read_renames(reply)

    result, made = [], {}
    for f in features:
        new_name = renames.get(f.name)
        if not new_name or new_name == f.name:
            result.append(f)
            continue
        if new_name not in made:
            made[new_name] = make_feature(context, exchange(prompt, reply), new_name)
        result.append(made[new_name] or f)
    return one_per_property(result, context)

def mask(features, context):
    kept = []
    for f in features:
        prompt = prompts.fill(prompts.MASK, features=names(features), feature=f.name, **context.prompt_fields())
        reply = context.ask('mask', [user(prompt)], features=names(features), feature=f.name)
        digits = re.findall(r'\b([01])\b', reply)
        if not digits or digits[-1] == '1':
            kept.append(f)
    return kept

def flex(features, context):
    return mask(generalize(add(features, context), context), context)

def algae(features, context):
    # simplified ALGAE: Add only, no contrastive demos
    return add(features, context)

def mfe(features, context):
    # list every plausible feature of the scene, then mask
    prompt = prompts.fill(prompts.MFE_LIST, env=env_description(context.env), features=names(features))
    reply = context.ask('mfe_list', [user(prompt)], features=names(features))
    return mask(one_per_property(features + features_from_list(context, prompt, reply), context), context)

def usf(features, context):
    # one LLM call writes the whole feature list
    prompt = prompts.fill(prompts.USF_LIST, **context.prompt_fields())
    reply = context.ask('usf_list', [user(prompt)], features=names(features))
    return one_per_property(features_from_list(context, prompt, reply), context)

METHODS = {'ALGAE': algae, 'MFE': mfe, 'USF': usf, 'FLEX': flex}
