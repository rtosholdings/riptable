# End-to-end integration tests that bring up a ipykernel instance and tests riptable completions.
import pytest
from riptable import (
    Dataset,
    Multiset,
    Struct,
    arange,
)  # not unused; needed for evaluated code below
from typing import List, Set, Tuple
from textwrap import dedent
from ipykernel.tests.utils import kernel, TIMEOUT, execute, wait_for_idle
from riptable.Utils.teamcity_helper import is_running_in_teamcity


def get_ignore_matches():
    """Return the list of match results to ignore."""
    return ["---"]


def get_custom_cells() -> List[str]:
    return [
        dedent(
            '''\
        from riptable.rt_misc import autocomplete
        autocomplete()
        '''
        ),
        dedent(
            '''\
        from riptable.Utils.ipython_utils import enable_custom_attribute_completion
        enable_custom_attribute_completion()
        '''
        ),
    ]


def get_rt_object_to_complete_texts() -> List[Tuple[str, str]]:
    """Returns a list of tuples of riptable code object text with associated completion text."""
    return [
        (
            dedent(
                '''Dataset({_k: list(range(_i * 10, (_i + 1) * 10)) for _i, _k in enumerate(
            ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambada", "mu",
             "nu", "xi", "omnicron", "pi"])})'''
            ),
            "dataset.",
        ),
        (
            dedent(
                '''Struct({"alpha": 1, "beta": [2, 3], "gamma": ['2', '3'], "delta": arange(10),
            "epsilon": Struct({
                "theta": Struct({
                    "kappa": 3,
                    "zeta": 4,
                    }),
                "iota": 2,
                })
            })'''
            ),
            "struct.",
        ),
        (
            dedent(
                '''Multiset(
    {"ds_alpha": Dataset({k: list(range(i * 10, (i + 1) * 10)) for i, k in enumerate(
                ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"])}),
     "ds_beta": Dataset({k: list(range(i * 10, (i + 1) * 10)) for i, k in enumerate(
                ["eta", "theta", "iota", "kappa", "lambada", "mu"])}),
                })'''
            ),
            "multiset.",
        ),
    ]


# When running on version below ipykernel=5.1.3 and jupyter_console=6.0.0, the kernel would
# deterministically die on the wait_for_ready call with the exception message
# “RuntimeError: Kernel died before replying to kernel_info”.
# Beware, there is an open Github issue about the same failure happening in a nondeterministic fashion.
# https://github.com/jupyter/jupyter_client/issues/154
def get_matches(code_text: str, complete_text: str) -> List[str]:
    """Run ipykernel session with code_text and return associated matches of the completion_text."""
    matches: List[str]
    with kernel() as kc:
        _, reply = execute(code_text, kc=kc)
        assert reply['status'] == 'ok'
        wait_for_idle(kc)
        kc.complete(complete_text)
        reply = kc.get_shell_msg(block=True, timeout=TIMEOUT)
        matches = reply['content']['matches']
    return matches


@pytest.mark.parametrize(
    'greedy, jedi', [(True, True), (True, False), (False, True), (False, False),]
)
@pytest.mark.parametrize(
    'rt_obj_text, complete_text', get_rt_object_to_complete_texts()
)
@pytest.mark.parametrize('custom_cell', get_custom_cells()[:1])
def test_completion_equivalence(
    rt_obj_text: str, complete_text: str, custom_cell: str, greedy: bool, jedi: bool
):
    text_split = complete_text.split(".")
    identifier: str = text_split[0]

    import_cell = dedent(
        '''\
        from riptable import Dataset, Multiset, Struct, arange
        from IPython import get_ipython
        '''
    )
    config_cell = '\n'.join(
        [
            "ip = get_ipython()",
            f"ip.Completer.greedy = {greedy}",
            f"ip.Completer.use_jedi = {jedi}",
        ]
    )
    code_cell = f"{identifier}={rt_obj_text}"

    expected_cell = u'\n'.join([import_cell, config_cell, code_cell,])
    actual_cell = u'\n'.join([import_cell, config_cell, custom_cell, code_cell,])

    expected_matches: Set[str] = set(get_matches(expected_cell, complete_text))
    actual_matches: Set[str] = set(get_matches(actual_cell, complete_text))

    for m in get_ignore_matches():
        if m in expected_matches:
            expected_matches.remove(m)
        if m in actual_matches:
            actual_matches.remove(m)

    assert not expected_matches.symmetric_difference(
        actual_matches
    ), f'expected matches {expected_matches}\ngot matches {actual_matches}'


@pytest.mark.parametrize(
    'greedy, jedi', [(True, True), (True, False), (False, True), (False, False),]
)
@pytest.mark.parametrize(
    'rt_obj_text, complete_text',
    get_rt_object_to_complete_texts()[:2],  # todo - add support for multiset
)
@pytest.mark.parametrize('custom_cell', get_custom_cells()[:1])
def test_match_key_order(
    rt_obj_text: str, complete_text: str, custom_cell: str, greedy: bool, jedi: bool
):
    text_split = complete_text.split(".")
    identifier: str = text_split[0]

    import_cell = dedent(
        '''\
        from riptable import Dataset, Multiset, Struct, arange
        from IPython import get_ipython
        '''
    )
    config_cell = '\n'.join(
        [
            "ip = get_ipython()",
            f"ip.Completer.greedy = {greedy}",
            f"ip.Completer.use_jedi = {jedi}",
        ]
    )
    code_cell = f"{identifier}={rt_obj_text}"
    cell = u'\n'.join([import_cell, config_cell, custom_cell, code_cell,])

    matches: List[str] = get_matches(cell, complete_text)

    # evaluate object text to inspect key set
    rt_obj = eval(rt_obj_text)
    expected_keys: List[str] = []
    if isinstance(rt_obj, (Struct, Dataset)):
        expected_keys = sorted(rt_obj.keys())

    assert len(expected_keys), f"type {type(rt_obj)} has no keys"

    for i, k in enumerate(expected_keys):
        if not jedi:  # without jedi - completion results show with completion text
            k = complete_text + k
        assert (
            k == matches[i]
        ), f"expected key {k} in completion result position {i}\nwanted keys {expected_keys}\ngot matches {matches}"


@pytest.mark.xfail(
    reason="RIP-323: enable_custom_attribute_completion fails; complete text shows in results"
)
@pytest.mark.skipif(
    is_running_in_teamcity(), reason="Please remove alongside xfail removal."
)
@pytest.mark.parametrize(
    'greedy, jedi', [(True, True), (True, False), (False, True), (False, False),]
)
@pytest.mark.parametrize(
    'rt_obj_text, complete_text', get_rt_object_to_complete_texts()
)
@pytest.mark.parametrize('custom_cell', get_custom_cells()[0:])
def test_completion_equivalence_fail(
    rt_obj_text: str, complete_text: str, custom_cell: str, greedy: bool, jedi: bool
):
    text_split = complete_text.split(".")
    identifier: str = text_split[0]

    import_cell = dedent(
        '''\
        from riptable import Dataset, Multiset, Struct, arange
        from IPython import get_ipython
        '''
    )
    config_cell = '\n'.join(
        [
            "ip = get_ipython()",
            f"ip.Completer.greedy = {greedy}",
            f"ip.Completer.use_jedi = {jedi}",
        ]
    )
    code_cell = f"{identifier}={rt_obj_text}"

    expected_cell = u'\n'.join([import_cell, config_cell, code_cell,])
    actual_cell = u'\n'.join([import_cell, config_cell, custom_cell, code_cell,])

    expected_matches: Set[str] = set(get_matches(expected_cell, complete_text))
    actual_matches: Set[str] = set(get_matches(actual_cell, complete_text))

    for m in get_ignore_matches():
        if m in expected_matches:
            expected_matches.remove(m)
        if m in actual_matches:
            actual_matches.remove(m)

    assert not expected_matches.symmetric_difference(
        actual_matches
    ), f'expected matches {expected_matches}\ngot matches {actual_matches}'


@pytest.mark.parametrize(
    'greedy, jedi', [(True, True), (True, False), (False, True), (False, False),]
)
@pytest.mark.parametrize(
    'rt_obj_text, complete_text',
    get_rt_object_to_complete_texts()[:2],  # todo - add support for multiset
)
@pytest.mark.parametrize('custom_cell', get_custom_cells()[:1])
def test_match_order_runtime_key_addition(
    rt_obj_text: str, complete_text: str, custom_cell: str, greedy: bool, jedi: bool
):
    text_split = complete_text.split(".")
    identifier: str = text_split[0]

    import_cell = dedent(
        '''\
        from riptable import Dataset, Multiset, Struct, arange
        from IPython import get_ipython
        '''
    )
    config_cell = '\n'.join(
        [
            "ip = get_ipython()",
            f"ip.Completer.greedy = {greedy}",
            f"ip.Completer.use_jedi = {jedi}",
        ]
    )

    first_key, last_key = 'AAAA', 'zzzz'
    code_cell = '\n'.join(
        [
            f"{identifier}={rt_obj_text}",
            f"{identifier}['{first_key}'] = '{first_key}'",
            f"{identifier}['{last_key}'] = '{last_key}'",
        ]
    )
    cell = u'\n'.join([import_cell, config_cell, custom_cell, code_cell,])

    matches: List[str] = get_matches(cell, complete_text)

    # evaluate object text to inspect key set
    rt_obj = eval(rt_obj_text)
    rt_obj[first_key] = first_key
    rt_obj[last_key] = last_key

    expected_keys: List[str] = []
    if hasattr(rt_obj, "keys"):
        expected_keys = sorted(rt_obj.keys())

    if not jedi:  # without jedi - completion results show with completion text
        first_key = complete_text + first_key
        last_key = complete_text + last_key

    assert (
        matches[0] == first_key
    ), f"expected key {first_key} to be the first match result\nwanted keys {expected_keys}\ngot matches {matches}"

    last_key_index = len(expected_keys) - 1
    assert (
        matches[last_key_index] == last_key
    ), f"expected key {last_key} to be at match result index {last_key_index}\nwanted keys {expected_keys}\ngot matches {matches}"
