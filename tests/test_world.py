from flex.world import env_description, get_objects_with_property, make_env


def test_layout_is_reproducible():
    assert make_env('kitchen', 4) == make_env('kitchen', 4)
    assert make_env('kitchen', 4) != make_env('kitchen', 5)


def test_objects_lie_in_the_central_block():
    for seed in range(20):
        env = make_env('lavaland', seed)
        for slot in range(1, 4):
            x, y = env[f'objects{slot}_pos'][0]
            assert 2 <= x <= 6 and 2 <= y <= 6


def test_property_lookup():
    env = make_env('kitchen', 0)
    assert len(get_objects_with_property('heat_source', env)) == 2
    assert get_objects_with_property('goal', env) == env['goals_pos']
    assert env_description(env).startswith('env:\n  type: gridrobot')
