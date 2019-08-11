from collections import namedtuple


def nested_map(transform, nested, element_types=(), tuples_to_lists=False):
    if any(isinstance(nested, t) for t in element_types):
        return transform(nested)

    if isinstance(nested, list) or isinstance(nested, tuple):
        result = (nested_map(transform, v, element_types) for v in nested)

        if isinstance(nested, list) or tuples_to_lists:
            return list(result)

        return tuple(result)

    if isinstance(nested, dict):
        nested = {k: nested_map(transform, v, element_types) for k, v in nested.items()}

        return nested

    return transform(nested)


def zip_dicts(dict1, dict2):
    return {k: (v1, dict2[k]) for k, v1 in dict1.items()}


ZippedValue = namedtuple('ZippedValue', ('value1', 'value2'))
IndexedValue = namedtuple('IndexedValue', ('index', 'value'))


def nested_zip(nested1, nested2, element_types=()):
    if any(isinstance(nested1, t) or isinstance(nested2, t) for t in element_types):
        return ZippedValue(nested1, nested2)

    if isinstance(nested1, list) or isinstance(nested1, tuple):
        return type(nested1)(map(lambda cs: nested_zip(*cs, element_types), zip(nested1, nested2)))

    if isinstance(nested1, dict):
        return type(nested1)(
            {k: nested_zip(v1, v2, element_types) for k, (v1, v2) in
             zip_dicts(nested1, nested2).items()})

    return ZippedValue(nested1, nested2)


def nested_enumerate(nested, prefix=(), element_types=()):
    if any(isinstance(nested, t) for t in element_types):
        return IndexedValue(prefix, nested)

    if isinstance(nested, list) or isinstance(nested, tuple):
        return type(nested)(
            nested_enumerate(value, prefix=prefix + (index,), element_types=element_types)
            for index, value in enumerate(nested))

    if isinstance(nested, dict):
        return type(nested)(
            {index: nested_enumerate(value, prefix=prefix + (index,), element_types=element_types)
             for index, value in nested.items()})

    return IndexedValue(prefix, nested)


def get_nested_element(nested, index_path):
    for index in index_path:
        nested = nested[index]

    return nested


def set_nested_element(nested, index_path, value):
    for index in index_path[:-1]:
        nested = nested[index]

    nested[index_path[-1]] = value


def nested_any(nested):
    if isinstance(nested, list) or isinstance(nested, tuple):
        return any([nested_any(v) for v in nested])

    if isinstance(nested, dict):
        return any([nested_any(v) for v in nested.values()])

    return nested
