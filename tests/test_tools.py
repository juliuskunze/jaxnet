from jaxnet.tools import map_nested, enumerate_nested, zip_nested, get_nested_element, \
    set_nested_element, nested_any, flatten_nested


def test_map_nested():
    assert (1, {'a': 2, 'b': [3]}) == map_nested(lambda x: x + 1, (0, {'a': 1, 'b': [2]}))


def test_map_nested_tuples_to_lists():
    r = map_nested(lambda x: x + 1, (0, {'a': 1, 'b': [2]}), tuples_to_lists=True)
    assert [1, {'a': 2, 'b': [3]}] == r


def test_enumerate_nested():
    assert enumerate_nested((0, {'a': 1, 'b': [2]})) == (
        ((0,), 0), {'a': ((1, 'a'), 1), 'b': [((1, 'b', 0), 2)]})


def test_flatten_nested():
    assert (0, 1, 2) == tuple(flatten_nested((0, {'a': 1, 'b': [2]})))


def test_zip_nested():
    a = (0, {'a': 1, 'b': [2]})
    b = (1, {'a': 2, 'b': [3]})
    assert ((0, 1), {'a': (1, 2), 'b': [(2, 3)]}) == zip_nested(a, b)


def test_get_nested_element():
    nested = (0, {'a': 1, 'b': [2]})
    assert 2 == get_nested_element(nested, (1, 'b', 0))
    assert nested == get_nested_element(nested, ())


def test_set_nested_element():
    nested = (0, {'a': 1, 'b': [2]})
    path = (1, 'b', 0)
    set_nested_element(nested, path, 3)
    assert get_nested_element(nested, path) == 3


def test_set_nested_element_in_tuple():
    nested = (0, {'a': 1, 'b': (2,)})
    path = (1, 'b', 0)
    set_nested_element(nested, path, 3)
    assert (0, {'a': 1, 'b': [3]}) == nested


def test_nested_any():
    nested = (False, {'a': False, 'b': [True]})
    assert nested_any(nested)

    nested = (False, {'a': False, 'b': [False]})
    assert not nested_any(nested)
