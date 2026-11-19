from flex.features import Feature, distance_feature, feature_per_cell, is_additive
from flex.world import make_env


def test_distance_feature_is_additive_and_capped():
    env = make_env('kitchen', 0)
    feature = distance_feature('heat_source')
    values = feature_per_cell(feature, env)
    assert is_additive(feature, env, values)
    assert values.min() >= 0 and values.max() <= 3.0 + 1e-9


def test_non_additive_code_is_rejected():
    code = '''
def last_x(environment, trajectory):
    return np.array([trajectory[-2]])
'''
    env = make_env('kitchen', 0)
    feature = Feature('last_x', code)
    assert not is_additive(feature, env, feature_per_cell(feature, env))
