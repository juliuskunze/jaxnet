from jaxnet.tools import nested_map, nested_enumerate, nested_zip, get_nested_element, \
    set_nested_element, nested_any


def test_nested_map():
    assert nested_map(lambda x: x + 1, (0, {'a': 1, 'b': [2]})) == (1, {'a': 2, 'b': [3]})


def test_nested_map_tuples_to_lists():
    r = nested_map(lambda x: x + 1, (0, {'a': 1, 'b': [2]}), tuples_to_lists=True)
    assert r == [1, {'a': 2, 'b': [3]}]


def test_nested_enumerate():
    assert nested_enumerate((0, {'a': 1, 'b': [2]})) == (
        ((0,), 0), {'a': ((1, 'a'), 1), 'b': [((1, 'b', 0), 2)]})


def test_nested_zip():
    a = (0, {'a': 1, 'b': [2]})
    b = (1, {'a': 2, 'b': [3]})
    assert nested_zip(a, b) == ((0, 1), {'a': (1, 2), 'b': [(2, 3)]})


def test_get_nested_element():
    nested = (0, {'a': 1, 'b': [2]})
    assert get_nested_element(nested, (1, 'b', 0)) == 2
    assert get_nested_element(nested, ()) == nested


def test_set_nested_element():
    nested = (0, {'a': 1, 'b': [2]})
    path = (1, 'b', 0)
    set_nested_element(nested, path, 3)
    assert get_nested_element(nested, path) == 3


def test_nested_any():
    nested = (False, {'a': False, 'b': [True]})
    assert nested_any(nested)

    nested = (False, {'a': False, 'b': [False]})
    assert not nested_any(nested)
