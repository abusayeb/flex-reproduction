import numpy as np

from flex import irl
from flex.editing import Context, clean_name, flex
from flex.flexv import flex_v
from flex.llm import Oracle
from flex.scenarios import all_scenarios, personal_scenarios
from flex.world import goal_index, start_index


def test_clean_name():
    assert clean_name('**Distance to the stove**') == 'distance_to_the_stove'
    assert clean_name('1. distance_to_lava (0.9)') == 'distance_to_lava'


def test_oracle_flex_makes_the_expected_edit():
    scenario = [s for s in all_scenarios(['kitchen']) if s.name == 'All Three'][0]
    features = flex(scenario.initial_features(), Context(scenario.env(0), Oracle(), scenario))
    assert [f.name for f in features] == ['distance_to_goal', 'distance_to_heat_source']


def test_flex_v_undoes_a_wrong_generalisation():
    scenario = personal_scenarios()[0]
    env = scenario.env(0)
    start, goal = start_index(env), goal_index(env)
    demos = irl.sample_demos(scenario.true_reward(env), start, goal, 10, np.random.default_rng(0))
    llm = Oracle(overgeneralize=True)
    plain = flex(scenario.initial_features(), Context(env, llm, scenario, demos))
    checked = flex_v(scenario.initial_features(), Context(env, llm, scenario, demos))
    assert 'distance_to_heat_source' in [f.name for f in plain]
    assert [f.name for f in checked] == ['distance_to_goal', 'distance_to_kettle']
