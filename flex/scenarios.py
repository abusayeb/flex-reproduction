# The seven Table 1 scenarios, and the personal-preference ones used for FLEX-V.
from dataclasses import dataclass
from flex.config import GOAL_WEIGHT, HAZARD_WEIGHT, NEW_LAYOUT_SEED_OFFSET
from flex.features import distance_feature, feature_per_cell, goal_feature
from flex.world import (CORRECTIONS, INSTRUCTION, NEW_OBJECTS, PERSONAL_CORRECTIONS, WORLDS,
                        make_env)

# name: (edits needed, starting features)
SCENARIOS = {
    'Add Only': ({'add'}, ['goal']),
    'Gen. Only': ({'generalize'}, ['goal', 'mentioned']),
    'Mask Only': ({'mask'}, ['goal', 'mentioned', 'distractor']),
    'Add+Gen.': ({'add', 'generalize'}, ['goal', 'distractor']),
    'Add+Mask': ({'add', 'mask'}, ['goal', 'distractor']),
    'Mask+Gen.': ({'mask', 'generalize'}, ['goal', 'mentioned', 'distractor']),
    'All Three': ({'add', 'generalize', 'mask'}, ['goal', 'distractor']),
}

@dataclass
class Scenario:
    world: str
    name: str
    ops: set
    initial: list
    mentioned: str  # the object named in the correction
    hazard: str  # the property the correction is really about
    distractor: str  # the object Mask should drop
    object_types: list
    truth: str  # what the true reward avoids
    correction: str
    instruction: str = INSTRUCTION

    def env(self, seed):
        # the second hazard is only kept when Generalize (or a personal test) needs it
        env = make_env(self.world, seed)
        if 'generalize' in self.ops or 'personal' in self.ops:
            return env
        for slot in range(1, 4):
            if env[f'objects{slot}_type'] not in (self.mentioned, self.distractor):
                env[f'objects{slot}_pos'] = []
                env[f'objects{slot}_color'] = ''
                env[f'objects{slot}_type'] = ''
                env[f'objects{slot}_property'] = []
        return env

    def new_env(self, seed):
        # a new layout where the second hazard is a type never seen (toaster / quicksand)
        env = make_env(self.world, seed + NEW_LAYOUT_SEED_OFFSET)
        new_type, info = NEW_OBJECTS[self.world]
        for slot in range(1, 4):
            if env[f'objects{slot}_type'] not in (self.mentioned, self.distractor):
                env[f'objects{slot}_type'] = new_type
                env[f'objects{slot}_color'] = info['color']
                env[f'objects{slot}_property'] = list(info['property'])
        return env

    def initial_features(self):
        features = []
        for name in self.initial:
            if name == 'goal':
                features.append(goal_feature())
            elif name == 'mentioned':
                features.append(distance_feature(self.mentioned, source='builtin'))
            else:
                features.append(distance_feature(self.distractor, source='builtin'))
        return features

    def true_reward(self, env):
        return (GOAL_WEIGHT * feature_per_cell(goal_feature(), env)
                + HAZARD_WEIGHT * feature_per_cell(distance_feature(self.truth), env))

def all_scenarios(worlds=('kitchen', 'lavaland')):
    scenarios = []
    for world in worlds:
        objects, hazard, mentioned, distractor = WORLDS[world]
        for name, (ops, initial) in SCENARIOS.items():
            scenarios.append(Scenario(world, name, ops, initial, mentioned, hazard, distractor, list(objects),
                                      truth=hazard, correction=CORRECTIONS[world]))
    return scenarios

def personal_scenarios():
    # the user means one object only, so generalizing to its property is wrong
    scenarios = []
    for world, (thing, correction) in PERSONAL_CORRECTIONS.items():
        objects, hazard, _, distractor = WORLDS[world]
        for name in ('Add Only', 'Add+Mask'):
            ops, initial = SCENARIOS[name]
            ops = set(ops)
            ops.add('personal')
            scenarios.append(Scenario(world, f'Personal/{name}', ops, initial, thing, hazard, distractor,
                                      list(objects), truth=thing, correction=correction))
    return scenarios
