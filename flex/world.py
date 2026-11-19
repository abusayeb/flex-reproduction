import numpy as np

from flex.config import GRID_SIZE, PATH_LENGTH

KITCHEN_OBJECTS = {
    'fridge': {'color': 'blue', 'property': ['fridge', 'light_source', 'icy', 'stationary', 'appliance']},
    'stove': {'color': 'red', 'property': ['stove', 'heat_source', 'metal', 'stationary', 'appliance']},
    'kettle': {'color': 'orange', 'property': ['kettle', 'portable', 'heat_source', 'appliance']},
}

LAVA_OBJECTS = {
    'meadow': {'color': 'green', 'property': ['meadow', 'grass', 'soft', 'terrain']},
    'lava': {'color': 'red', 'property': ['lava', 'dangerous_terrain', 'hot', 'terrain']},
    'swamp': {'color': 'brown', 'property': ['swamp', 'dangerous_terrain', 'wet', 'terrain']},
}

# the object that replaces the second hazard in the new layout
NEW_OBJECTS = {
    'kitchen': ('toaster', {'color': 'silver', 'property': ['toaster', 'heat_source', 'portable', 'appliance']}),
    'lavaland': ('quicksand', {'color': 'yellow', 'property': ['quicksand', 'dangerous_terrain', 'sandy', 'terrain']}),
}

# world: (objects, hazard property, object named in the correction, distractor)
WORLDS = {
    'kitchen': (KITCHEN_OBJECTS, 'heat_source', 'stove', 'fridge'),
    'lavaland': (LAVA_OBJECTS, 'dangerous_terrain', 'lava', 'meadow'),
}

INSTRUCTION = 'Go to the goal.'

CORRECTIONS = {
    'kitchen': "Don't burn yourself on the stove.",
    'lavaland': "Don't walk into the lava.",
}

PERSONAL_CORRECTIONS = {
    'kitchen': ('kettle', 'Stay away from the kettle.'),
    'lavaland': ('swamp', 'Stay away from the swamp.'),
}

def cell_to_index(cell):
    x, y = cell
    return y * GRID_SIZE + x

def index_to_cell(index):
    return (index % GRID_SIZE, index // GRID_SIZE)

def all_cells():
    cells = []
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            cells.append((x, y))
    return cells

def move_table():
    # can_move[a, b] is True if the robot can go from cell a to cell b in one step (staying counts)
    n_cells = GRID_SIZE * GRID_SIZE
    can_move = np.zeros((n_cells, n_cells), dtype=bool)
    for x, y in all_cells():
        here = cell_to_index((x, y))
        can_move[here, here] = True
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if 0 <= x + dx < GRID_SIZE and 0 <= y + dy < GRID_SIZE:
                can_move[here, cell_to_index((x + dx, y + dy))] = True
    return can_move

def make_env(world, seed):
    # start in the top-left 3x3, goal in the bottom-right 3x3, objects in the central 5x5
    objects = WORLDS[world][0]
    rng = np.random.default_rng(seed)

    while True:
        start = (int(rng.integers(0, 3)), int(rng.integers(0, 3)))
        goal = (int(rng.integers(6, 9)), int(rng.integers(6, 9)))
        steps_needed = abs(goal[0] - start[0]) + abs(goal[1] - start[1])
        if steps_needed <= PATH_LENGTH - 1 - 4:
            break

    central = []
    for x in range(2, 7):
        for y in range(2, 7):
            central.append((x, y))
    picks = rng.choice(len(central), size=len(objects), replace=False)

    env = {'type': 'gridrobot', 'X': GRID_SIZE, 'Y': GRID_SIZE, 'starts': [list(start)],
           'goals_pos': [list(goal)], 'goals_color': 'green', 'goals_type': 'cross'}
    slot = 1
    for name, info in objects.items():
        env[f'objects{slot}_pos'] = [list(central[picks[slot - 1]])]
        env[f'objects{slot}_color'] = info['color']
        env[f'objects{slot}_type'] = name
        env[f'objects{slot}_property'] = list(info['property'])
        slot += 1
    return env

def list_objects(env):
    # (type, properties, positions) for every object slot that is not empty
    objects = []
    for slot in range(1, 4):
        positions = env.get(f'objects{slot}_pos') or []
        if positions:
            objects.append((env[f'objects{slot}_type'], env.get(f'objects{slot}_property', []), positions))
    return objects

def all_properties(env):
    properties = set()
    for object_type, object_properties, _ in list_objects(env):
        properties.add(object_type)
        properties.update(object_properties)
    return properties

def get_objects_with_property(prop, environment):
    # the LLM's feature code calls this, so the name and arguments must stay as they are
    prop = prop.lower().strip()
    if prop in ('goal', 'goals', environment.get('goals_type', '')):
        return [list(p) for p in environment['goals_pos']]
    found = []
    for object_type, object_properties, positions in list_objects(environment):
        if prop == object_type or prop in object_properties:
            for p in positions:
                found.append(list(p))
    return found

def env_description(env):
    # the env text in every prompt; changing it would miss the LLM cache
    lines = ['env:']
    for key, value in env.items():
        lines.append(f'  {key}: {value}')
    return '\n'.join(lines)

def start_index(env):
    return cell_to_index(env['starts'][0])

def goal_index(env):
    return cell_to_index(env['goals_pos'][0])
