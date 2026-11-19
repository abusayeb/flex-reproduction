import numpy as np

from flex import irl
from flex.features import distance_feature, feature_per_cell, feature_table, goal_feature
from flex.scenarios import all_scenarios
from flex.world import goal_index, index_to_cell, make_env, start_index


def test_paths_start_and_end_right():
    env = make_env('lavaland', 3)
    start, goal = start_index(env), goal_index(env)
    reward = -feature_per_cell(goal_feature(), env)
    best, _ = irl.best_path(reward, start, goal)
    assert best[0] == start and best[-1] == goal
    for path in irl.sample_demos(reward, start, goal, 5, np.random.default_rng(0)):
        assert path[0] == start and path[-1] == goal
        for a, b in zip(path, path[1:]):
            (x1, y1), (x2, y2) = index_to_cell(a), index_to_cell(b)
            assert abs(x1 - x2) + abs(y1 - y2) <= 1


def test_expected_features_match_sampling():
    env = make_env('kitchen', 1)
    start, goal = start_index(env), goal_index(env)
    table = feature_table([goal_feature(), distance_feature('stove')], env)
    reward = table @ np.array([-0.5, 1.0])
    exact = irl.expected_visits(reward, start, goal).sum(axis=0) @ table
    demos = irl.sample_demos(reward, start, goal, 3000, np.random.default_rng(0))
    sampled = np.mean([table[path].sum(axis=0) for path in demos], axis=0)
    assert np.allclose(exact, sampled, rtol=0.03)


def test_irl_recovers_true_behaviour():
    scenario = all_scenarios(['kitchen'])[0]
    env = scenario.env(0)
    start, goal = start_index(env), goal_index(env)
    reward = scenario.true_reward(env)
    demos = irl.sample_demos(reward, start, goal, 10, np.random.default_rng(0))
    table = feature_table([goal_feature(), distance_feature('heat_source')], env)
    weights, _ = irl.learn_weights(table, demos, start, goal)
    path, _ = irl.best_path(table @ weights, start, goal)
    assert irl.normalized_reward(reward, path, start, goal) > 0.99
