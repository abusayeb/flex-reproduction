import math
import re
import textwrap
import numpy as np
from flex import world
from flex.config import DISTANCE_CAP, GRID_SIZE
from flex.world import all_cells, all_properties, get_objects_with_property

# the only builtins the LLM's feature code may use
SAFE_BUILTINS = {
    'abs': abs, 'min': min, 'max': max, 'len': len, 'range': range, 'float': float, 'int': int, 'sum': sum,
    'enumerate': enumerate, 'zip': zip, 'list': list, 'tuple': tuple, 'dict': dict, 'set': set,
    'round': round, 'isinstance': isinstance, 'print': print, 'ValueError': ValueError,
    'Exception': Exception, 'sorted': sorted, 'any': any, 'all': all, 'map': map, 'pow': pow,
}

class Feature:
    def __init__(self, name, code, source='lm'):
        self.name = name
        self.code = code
        self.source = source

def safe_import(name, *args, **kwargs):
    if name == 'numpy':
        return np
    if name == 'math':
        return math
    # the paper's code imports get_objects_with_property from its own package
    if name.endswith('env_utils') or name.startswith('flexible_features'):
        return world
    raise ImportError(f'import of {name!r} not allowed in feature code')

def load_function(feature):
    # run the feature's code and return its function: the one named like the feature, else the last one
    allowed = dict(SAFE_BUILTINS)
    allowed['__import__'] = safe_import
    namespace = {'__builtins__': allowed, 'np': np, 'numpy': np, 'math': math,
                 'get_objects_with_property': get_objects_with_property}
    code = textwrap.dedent(feature.code)
    exec(code, namespace)
    if callable(namespace.get(feature.name)):
        return namespace[feature.name]
    function_names = re.findall(r'^def\s+(\w+)\s*\(', code, flags=re.M)
    for name in reversed(function_names):
        if callable(namespace.get(name)):
            return namespace[name]
    raise ValueError('no function defined in feature code')

def feature_per_cell(feature, env):
    # the feature's value in each of the 81 cells
    function = load_function(feature)
    values = np.zeros(GRID_SIZE * GRID_SIZE)
    for index, (x, y) in enumerate(all_cells()):
        value = np.asarray(function(env, [float(x), float(y)]), dtype=np.float64).reshape(-1)
        if value.size != 1 or not np.isfinite(value[0]):
            raise ValueError(f'feature returned {value!r} at {(x, y)}')
        values[index] = value[0]
    return values

def feature_table(features, env):
    # one row per cell, one column per feature
    table = np.zeros((GRID_SIZE * GRID_SIZE, len(features)))
    for column, feature in enumerate(features):
        table[:, column] = feature_per_cell(feature, env)
    return table

def is_additive(feature, env, per_cell):
    # f(path) must equal the sum of f over its cells, otherwise one value per cell is not enough
    rng = np.random.default_rng(0)
    function = load_function(feature)
    for _ in range(3):
        cells = rng.integers(0, GRID_SIZE, size=(5, 2))
        path = []
        for x, y in cells:
            path += [float(x), float(y)]
        whole = float(np.asarray(function(env, path), dtype=np.float64).reshape(-1)[0])
        parts = 0
        for x, y in cells:
            parts += per_cell[int(y) * GRID_SIZE + int(x)]
        if not np.isclose(whole, parts, rtol=1e-6, atol=1e-6):
            return False
    return True

def goal_feature():
    code = '''
def distance_to_goal(environment, trajectory):
    goals = get_objects_with_property("goal", environment)
    feat = np.zeros(1)
    for i in range(0, len(trajectory) - 1, 2):
        for g in goals:
            feat += np.hypot(trajectory[i] - g[0], trajectory[i + 1] - g[1])
    return feat / len(goals)
'''
    return Feature('distance_to_goal', code, source='builtin')

def distance_feature(prop, source='template'):
    name = f'distance_to_{prop}'
    code = f'''
def {name}(environment, trajectory, r={DISTANCE_CAP}):
    objs = get_objects_with_property("{prop}", environment)
    feat = np.zeros(1)
    if not objs:
        return feat
    for i in range(0, len(trajectory) - 1, 2):
        for o in objs:
            d = np.hypot(trajectory[i] - o[0], trajectory[i + 1] - o[1])
            feat += min(d, r)
    return feat / len(objs)
'''
    return Feature(name, code, source=source)

def to_snake(text):
    return re.sub(r'[^a-z0-9]+', '_', text.lower())

def property_in_name(name, env):
    # the env property a feature name talks about; the longest match wins
    name = to_snake(name)
    if 'goal' in name:
        return 'goal'
    best = None
    for prop in all_properties(env):
        word = to_snake(prop)
        for spelling in [word, word.rstrip('s'), word + 's', word.replace('_', '')]:
            if spelling and spelling in name and (best is None or len(prop) > len(best)):
                best = prop
    return best

def template_feature(name, env):
    prop = property_in_name(name, env)
    if prop is None:
        return None
    if prop == 'goal':
        return goal_feature()
    return distance_feature(prop)

def property_of(feature, env):
    # the property a feature measures: read from its code, else guessed from its name
    found = re.findall(r'get_objects_with_property\(\s*[\'"]([^\'"]+)[\'"]', feature.code)
    if found:
        return found[0]
    return property_in_name(feature.name, env)
