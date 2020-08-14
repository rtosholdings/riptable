import unittest
import pytest
from enum import IntEnum
from contextlib import contextmanager
from typing import Union, List
from IPython import get_ipython
from IPython.core.completer import provisionalcompleter
from IPython.terminal import ptutils
from riptable import FastArray, Categorical, Dataset, Struct, Multiset, arange
from riptable.Utils.ipython_utils import (
    enable_custom_attribute_completion,
    _get_key_names,
)
from riptable.rt_misc import autocomplete
from riptable.tests.test_categorical import decision_dict
from riptable.tests.utils import LikertDecision


_DEBUG = False


class CompletionType(IntEnum):
    UNDEFINED = 0
    KEY = 1
    ATTRIBUTE = 2


class GreedyJediConfigType(IntEnum):
    NEITHER = 0
    GREEDY = 1
    JEDI = 2
    GREEDY_JEDI = 3


_COMPLETION_TYPE_TABLE = [CompletionType.KEY, CompletionType.ATTRIBUTE]


_GREEDY_JEDI_CONFIG_TABLE = [
    GreedyJediConfigType.NEITHER,
    GreedyJediConfigType.GREEDY,
    GreedyJediConfigType.JEDI,
    GreedyJediConfigType.GREEDY_JEDI,
]


CODES = [1, 44, 44, 133, 75]


# Todo - put this in a function generator that can get data based on data type
# Add data entries to _RT_DATA_TABLE in an append only way since tests depend on ordering.
# Tests should not depend on ordering of _RT_DATA_TABLE, see above work item.
_RT_DATA_TABLE = [
    Categorical(
        FastArray(['a', 'b', 'c', 'c', 'd', 'a', 'b']),
        ordered=True,
        base_index=1,
        filter=None,
    ),
    Categorical(CODES, LikertDecision),
    Categorical(CODES, decision_dict),
    Categorical(['b', 'a', 'a', 'c', 'a', 'b'], ['b', 'a', 'c', 'e'], sort_gb=True),
    Dataset(
        {
            _k: list(range(_i * 10, (_i + 1) * 10))
            for _i, _k in enumerate(
                [
                    "alpha",
                    "beta",
                    "gamma",
                    "delta",
                    "epsilon",
                    "zeta",
                    "eta",
                    "theta",
                    "iota",
                    "kappa",
                    "lambada",
                    "mu",
                    "nu",
                    "xi",
                    "omnicron",
                    "pi",
                ]
            )
        }
    ),
    Multiset(
        {
            "ds_alpha": Dataset(
                {
                    k: list(range(i * 10, (i + 1) * 10))
                    for i, k in enumerate(
                        ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
                    )
                }
            ),
            "ds_beta": Dataset(
                {
                    k: list(range(i * 10, (i + 1) * 10))
                    for i, k in enumerate(
                        ["eta", "theta", "iota", "kappa", "lambada", "mu"]
                    )
                }
            ),
        }
    ),
    Struct(
        {
            "alpha": 1,
            "beta": [2, 3],
            "gamma": ['2', '3'],
            "delta": arange(10),
            "epsilon": Struct({"theta": Struct({"kappa": 3, "zeta": 4,}), "iota": 2,}),
        }
    ),
]


@contextmanager
def greedy_completion():
    ip = get_ipython()
    greedy_original = ip.Completer.greedy
    try:
        ip.Completer.greedy = True
        yield
    finally:
        ip.Completer.greedy = greedy_original


@contextmanager
def jedi_completion():
    ip = get_ipython()
    use_jedi_original = ip.Completer.use_jedi
    try:
        ip.Completer.use_jedi = True
        yield
    finally:
        ip.Completer.use_jedi = use_jedi_original


def get_completion_text(completion_type) -> str:
    text: str = ""
    if completion_type == CompletionType.KEY:
        text = "d['"
    elif completion_type == CompletionType.ATTRIBUTE:
        text = "d."
    else:
        raise ValueError(
            "get_completion_text: could not handled {} of type {}".format(
                completion_type, type(completion_type)
            )
        )
    return text


@pytest.mark.parametrize('greedy_jedi_config_type', _GREEDY_JEDI_CONFIG_TABLE)
@pytest.mark.parametrize('completion_type', _COMPLETION_TYPE_TABLE)
@pytest.mark.parametrize('rt_data', _RT_DATA_TABLE)
def test_simple_completion(
    greedy_jedi_config_type: GreedyJediConfigType,
    completion_type: CompletionType,
    rt_data: Union[FastArray, Dataset, Categorical, Struct],
):
    ip = get_ipython()
    ip.user_ns["d"] = rt_data

    completions: list
    text = get_completion_text(completion_type)
    with provisionalcompleter():
        if greedy_jedi_config_type == GreedyJediConfigType.GREEDY_JEDI:
            with greedy_completion():
                with jedi_completion():
                    completions = list(ip.Completer.completions(text, len(text)))
        elif greedy_jedi_config_type == GreedyJediConfigType.GREEDY:
            with greedy_completion():
                completions = list(ip.Completer.completions(text, len(text)))
        elif greedy_jedi_config_type == GreedyJediConfigType.JEDI:
            with jedi_completion():
                completions = list(ip.Completer.completions(text, len(text)))
        elif greedy_jedi_config_type == GreedyJediConfigType.NEITHER:
            completions = list(ip.Completer.completions(text, len(text)))
        else:
            raise ValueError(
                "test_simple_completion: could not handle greedy_jedi_config_type {} of type {}".format(
                    greedy_jedi_config_type, type(greedy_jedi_config_type)
                )
            )

    matches = [completion.text for completion in completions]
    if isinstance(rt_data, Categorical):
        rt_data = rt_data.categories()
    if _get_key_names(rt_data):
        for key in rt_data.keys():
            assert True, key in matches


# add struct to categorical, dataset, fastarray, and struct completion tests


class TestIPCompleterIntegration(unittest.TestCase):
    def setUp(self):
        """
        Silence all PendingDeprecationWarning when testing the completer integration.
        PendingDeprecationWarning: `Completer.complete` is pending deprecation since IPython 6.0 and will be replaced by `Completer.completions`.
        """
        self._assertwarns = self.assertWarns(PendingDeprecationWarning)
        self._assertwarns.__enter__()

    def tearDown(self):
        try:
            self._assertwarns.__exit__(None, None, None)
        except AssertionError:
            pass

    # region Categorical
    def test_categorical_string_array_key_completion(self):
        ip = get_ipython()
        complete = ip.Completer.complete
        lst = ['a', 'b', 'c', 'c', 'd', 'a', 'b']  # type: List[str]
        ip.user_ns["cat"] = Categorical(
            FastArray(lst), ordered=True, base_index=1, filter=None
        )
        _, matches = complete(line_buffer="cat['")
        for s in lst:
            self.assertIn(s, matches)

    def test_categorical_int_enum_key_completion(self):
        ip = get_ipython()
        complete = ip.Completer.complete

        ip.user_ns["cat"] = Categorical(CODES, LikertDecision)
        _, matches = complete(line_buffer="cat['")
        for k in decision_dict.keys():
            self.assertIn(k, matches)

    def test_categorical_dict_key_completion(self):
        ip = get_ipython()
        complete = ip.Completer.complete

        ip.user_ns["cat"] = Categorical(CODES, decision_dict)
        _, matches = complete(line_buffer="cat['")
        for k in decision_dict.keys():
            self.assertIn(k, matches)

    def test_categorical_numeric_array_key_completion(self):
        ip = get_ipython()
        complete = ip.Completer.complete

        lst = [1, 44, 44, 133, 75]  # type: List[int]
        ip.user_ns["cat"] = Categorical(FastArray(lst))
        _, matches = complete(line_buffer="cat['")
        expected = [str(i) for i in lst]
        for c in expected:
            self.assertIn(c, matches)

    # todo alz 20191202 - add variant test cases for types of various combinations of input keys
    def test_categorical_multi_key_completion(self):
        ip = get_ipython()
        complete = ip.Completer.complete

        # note - 'e' is not in first list
        lst1 = ['b', 'a', 'a', 'c', 'a', 'b']  # type: List[str]
        lst2 = ['b', 'a', 'c', 'e']  # type: List[str]
        ip.user_ns["cat"] = Categorical(lst1, lst2, sort_gb=True)
        _, matches = complete(line_buffer="cat['")
        for c in lst1:
            self.assertIn(c, matches)
        for c in lst2:
            self.assertIn(c, matches)

    # todo alz 20191202 - add test cases for column selection cat[[string1, string2]]
    # endregion


# region rt_misc autocomplete
@pytest.mark.parametrize(
    'rt_data, text',
    [
        # Simple attribute completion tests
        (_RT_DATA_TABLE[4], "d."),
        (_RT_DATA_TABLE[5], "d."),
        (_RT_DATA_TABLE[6], "d."),
        # Nested attribute completion tests
        # Multiset
        (_RT_DATA_TABLE[5], "m."),
        (_RT_DATA_TABLE[5], "m.ds_alpha."),
        # Struct
        (_RT_DATA_TABLE[6], "s."),
        (_RT_DATA_TABLE[6], "s.epsilon."),
        (_RT_DATA_TABLE[6], "s.epsilon.theta."),
    ],
)
def test_riptable_autocomplete_keys_ordering(
    rt_data: Union[Dataset, Multiset, Struct], text: str
):
    # Keys are expected to be in sorted order.
    autocomplete()
    with provisionalcompleter(), jedi_completion():
        ip = get_ipython()

        text_split = text.split(".")
        ns_key: str = text_split[0]
        ip.user_ns[ns_key] = rt_data

        completions = list(
            ptutils._deduplicate_completions(
                text, ip.Completer.completions(text, len(text))
            )
        )
        matches = [completion.text for completion in completions]

        expected_matches: List[str] = []
        if len(text_split) > 2:  # support arbitrary object depth expectations
            locals()[ns_key] = rt_data
            rt_data = eval(text[:-1])

        if _get_key_names(rt_data):  # support object key expectations
            expected_matches.extend(sorted(rt_data.keys()))

        assert len(expected_matches), f"type {type(rt_data)} has no keys"

        for i, k in enumerate(expected_matches):
            # Match will be key, attribute, or method name.
            # E.g., given namespace name `d` and key `alpha` completion matches will contain the value `alpha`.
            assert i == matches.index(
                k
            ), f"expected key {k} in completion result position {i}\nwanted keys {expected_matches}\ngot matches {matches}"


@pytest.mark.parametrize(
    'rt_data, text',
    [
        # Simple attribute completion tests
        (_RT_DATA_TABLE[4], "d."),
        (_RT_DATA_TABLE[5], "d."),
        (_RT_DATA_TABLE[6], "d."),
        # Nested attribute completion tests
        # Multiset
        (_RT_DATA_TABLE[5], "m."),
        (_RT_DATA_TABLE[5], "m.ds_alpha."),
        # Struct
        (_RT_DATA_TABLE[6], "s."),
        (_RT_DATA_TABLE[6], "s.epsilon."),
        (_RT_DATA_TABLE[6], "s.epsilon.theta."),
    ],
)
def test_riptable_autocomplete_equivalent_to_ipython_without_ordering(
    rt_data: Union[Dataset, Multiset, Struct], text: str
):
    # Keys are expected to be in sorted order, but are not. This is tracked in RIP-323.
    # Ensure riptable autocomplete contains sane completion results as IPython's auto completion.
    with provisionalcompleter(), jedi_completion():
        ip = get_ipython()

        ns_key: str = text.split(".")[0]
        ip.user_ns[ns_key] = rt_data

        expected_completions = list(
            ptutils._deduplicate_completions(
                text, ip.Completer.completions(text, len(text))
            )
        )
        expected_matches = set([completion.text for completion in expected_completions])

        autocomplete()

        completions = list(
            ptutils._deduplicate_completions(
                text, ip.Completer.completions(text, len(text))
            )
        )
        matches = set([completion.text for completion in completions])

        # IPython's set of completions should be the same as Riptable's set of completions.
        assert not expected_matches.symmetric_difference(
            matches
        ), f'expected matches {expected_matches}\ngot matches {matches}'


# endregion rt_misc autocomplete


# region ipython_utils enable_custom_attribute_completion
# Please leave these tests at the end since they activate the custom completion machinery.
# As of 20191218, if ``CUSTOM_COMPLETION`` is toggled on it results in a one-time registration of custom
# but does not support deregistration.
@pytest.mark.parametrize(
    'rt_data, text',
    [
        # Simple attribute completion tests
        (_RT_DATA_TABLE[4], "d."),
        (_RT_DATA_TABLE[5], "d."),
        (_RT_DATA_TABLE[6], "d."),
        # Nested attribute completion tests
        # Multiset
        (_RT_DATA_TABLE[5], "m."),
        (_RT_DATA_TABLE[5], "m.ds_alpha."),
        # Struct
        (_RT_DATA_TABLE[6], "s."),
        (_RT_DATA_TABLE[6], "s.epsilon."),
        (_RT_DATA_TABLE[6], "s.epsilon.theta."),
    ],
)
def test_monkey_patch_complete_match_ordering(
    rt_data: Union[Dataset, Multiset, Struct], text: str
):
    # 1) Keys are expected to be in sorted order.
    enable_custom_attribute_completion()

    ip = get_ipython()
    complete = ip.Completer.complete

    text_split = text.split(".")
    ns_key: str = text_split[0]
    ip.user_ns[ns_key] = rt_data

    _, matches = complete(line_buffer=text)

    expected_matches: List[str] = []
    if len(text_split) > 2:  # support arbitrary object depth expectations
        locals()[ns_key] = rt_data
        rt_data = eval(text[:-1])

    if _get_key_names(rt_data):  # support object key expectations
        expected_matches.extend(sorted(rt_data.keys()))

    for expected_index, expected_match in enumerate(expected_matches):
        # Match will be namespace name dot key, attribute, or method name.
        # E.g., given namespace name `d` and key `alpha` completion matches will contain the value `d.alpha`.
        expected_match: str = text + expected_match
        assert expected_index == matches.index(
            expected_match
        ), f"expected match {expected_match} in completion result position {expected_index}\nwanted keys {expected_matches}\ngot matches {matches}"


# endregion ipython_utils enable_custom_attribute_completion


if __name__ == '__main__':
    tester = unittest.main()
