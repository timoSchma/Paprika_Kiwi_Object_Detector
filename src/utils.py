from enum import Enum
from collections import namedtuple


class Corner(Enum):
    """The four corners of a box as enum."""
    TOP_LEFT = 1
    TOP_RIGHT = 2
    BOTTOM_LEFT = 3
    BOTTOM_RIGHT = 4


Point = namedtuple('Point', ['x', 'y'])

Dimension = namedtuple('Dimension', ['width', 'height'])


def split_list(list_, index_lists):
    """
    Splits a list into multiple lists according to a given iterable of index lists.

    Parameters
    ----------
    list_ : List[Any]
        The list to split.
    index_lists : Iterable[List[int]]
        The iterable of index lists.

    Returns
    -------
    iterable of lists :
        The split lists.
    """
    return ([list_[i] for i in indices] for indices in index_lists)


def split_dict(dict_, index_lists=None, key_lists=None):
    """
    Splits a dict into multiple dicts according to a given iterable of key lists or index lists if no key lists are
    given. At least one of the two must be given, otherwise the original dict is returned.

    Parameters
    ----------
    dict_ : Dict
        The dict to split.
    index_lists : Iterable[List[int]]
        The iterable of index lists.
    key_lists : Iterable[List[Any]]
        The iterable of key lists.

    Returns
    -------
    iterable of dicts
        The split dicts.
    """
    if key_lists is None:
        if index_lists is None:
            return dict_
        else:
            key_lists = split_list(list(dict_.keys()), index_lists)

    return ({key: dict_[key] for key in keys} for keys in key_lists)
