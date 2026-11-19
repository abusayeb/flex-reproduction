import hashlib
import json
import os
import time
import urllib.request
from flex.config import CACHE_DIR, LLM_TIMEOUT, OLLAMA_HOST
from flex.features import distance_feature, goal_feature

def post_json(url, payload, headers, tries=5):
    data = json.dumps(payload).encode()
    headers = dict(headers)
    headers['Content-Type'] = 'application/json'
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(request, timeout=LLM_TIMEOUT) as response:
                return json.loads(response.read())
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)

class LLM:
    name = 'base'

    def __init__(self, temperature=0.0, cache_file=None):
        self.temperature = temperature
        self.cache_file = cache_file
        self.cache = {}
        self.calls = []  # (tag, came from cache) for every call
        if cache_file and cache_file.exists():
            for line in cache_file.read_text().splitlines():
                record = json.loads(line)
                self.cache[record['key']] = record['reply']

    def chat(self, messages, tag='', info=None):
        # the cache key is a hash of the model, temperature and messages
        text = json.dumps([self.name, self.temperature, 0, messages], sort_keys=True)
        key = hashlib.sha256(text.encode()).hexdigest()
        if key in self.cache:
            self.calls.append((tag, True))
            return self.cache[key]

        reply = self.reply(messages)
        self.calls.append((tag, False))
        self.cache[key] = reply
        if self.cache_file:
            self.cache_file.parent.mkdir(exist_ok=True)
            with self.cache_file.open('a') as f:
                record = {'key': key, 'tag': tag, 'messages': messages, 'reply': reply}
                f.write(json.dumps(record) + '\n')
        return reply

class Ollama(LLM):
    def __init__(self, model, **kwargs):
        self.model = model
        self.name = f'ollama:{model}'
        super().__init__(**kwargs)

    def reply(self, messages):
        payload = {'model': self.model, 'messages': messages, 'stream': False,
                   'options': {'temperature': self.temperature}}
        return post_json(f'{OLLAMA_HOST}/api/chat', payload, {})['message']['content']

class OpenAI(LLM):
    def __init__(self, model, **kwargs):
        self.model = model
        self.name = f'openai:{model}'
        super().__init__(**kwargs)

    def reply(self, messages):
        payload = {'model': self.model, 'messages': messages, 'temperature': self.temperature}
        headers = {'Authorization': 'Bearer ' + os.environ['OPENAI_API_KEY']}
        answer = post_json('https://api.openai.com/v1/chat/completions', payload, headers)
        return answer['choices'][0]['message']['content']

class Oracle(LLM):
    # always gives the edit the scenario expects, so it is an upper bound, not a result
    name = 'oracle'

    def __init__(self, overgeneralize=False, **kwargs):
        self.overgeneralize = overgeneralize
        super().__init__(**kwargs)

    def chat(self, messages, tag='', info=None):
        # never cached: two scenarios can send the same prompt but expect different edits
        self.calls.append((tag, False))
        scenario = info['scenario']
        features = info.get('features', [])

        if tag == 'add':
            already_there = False
            for name in features:
                if scenario.mentioned in name or scenario.hazard in name:
                    already_there = True
            if 'add' in scenario.ops and not already_there:
                return f'distance_to_{scenario.mentioned}\nThe addendum says the {scenario.mentioned} matters.'
            return 'NO NEW FEATURE'

        if tag == 'intent':
            return f"The human wants the agent to stay away from {scenario.hazard.replace('_', ' ')} objects."

        if tag == 'generalize':
            should_generalize = 'generalize' in scenario.ops or self.overgeneralize
            lines = []
            for name in features:
                if should_generalize and scenario.mentioned in name:
                    lines.append(f'{name} -> distance_to_{scenario.hazard}')
                else:
                    lines.append(f'{name} -> NO CHANGE')
            return '\n'.join(lines)

        if tag == 'mask':
            if 'mask' in scenario.ops and scenario.distractor in info['feature']:
                return '0'
            return '1'

        if tag == 'mfe_list':
            return '\n'.join(f'distance_to_{o} (0.8)' for o in scenario.object_types)

        if tag == 'usf_list':
            return f'distance_to_goal (1.0)\ndistance_to_{scenario.hazard} (0.9)'

        if tag == 'codegen':
            name = info['feature']
            if 'goal' in name:
                feature = goal_feature()
            else:
                feature = distance_feature(name.replace('distance_to_', ''))
            return f'Feature name: "{feature.name}"\nFeature code:\n```python\n{feature.code}\n```'

        raise ValueError(tag)

def make_llm(spec, temperature=0.0):
    # spec: 'oracle', 'oracle:overgeneralize', 'ollama:<model>' or 'openai:<model>'
    if spec == 'oracle':
        return Oracle(temperature=temperature)
    if spec == 'oracle:overgeneralize':
        return Oracle(overgeneralize=True, temperature=temperature)
    kind, model = spec.split(':', 1)
    cache_file = CACHE_DIR / (spec.replace(':', '_').replace('/', '_') + f'_t{temperature}.jsonl')
    if kind == 'ollama':
        return Ollama(model, temperature=temperature, cache_file=cache_file)
    if kind == 'openai':
        return OpenAI(model, temperature=temperature, cache_file=cache_file)
    raise ValueError(f'unknown model {spec}')
