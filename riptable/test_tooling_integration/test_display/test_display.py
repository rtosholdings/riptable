import os

import pytest

from .vnu_checker import VNUChecker
from riptable import Dataset, Struct, Multiset

@pytest.mark.xfail(reason="Requires the VNU.jar HTML validator")
@pytest.mark.parametrize(
    "rt_obj",
    [
        # Add complex examples that use composition and recursion of other riptable data structures.
        Dataset({"one": 1}),
        Multiset({"ds": Dataset({"one": 1})}),
        Struct({"ds": Dataset({"one": 1})}),
    ],
)
def test_simple_html_repr(rt_obj, tmp_path):
    # dump HTML representation to files
    if not os.path.isdir(tmp_path):
        tmp_path.mkdir()
    checker = VNUChecker(base_path=str(tmp_path), errors_only=True, ascii_quotes=True)
    fn = os.path.join(checker.base_path, rt_obj.__class__.__name__ + ".html")
    with open(fn, "w") as f:
        # Consider using builtin library lxml or third-party BeautifulSoup for better HTML dumps:
        # lxml:
        # doc = html.fromstring(rt_obj._repr_html_())
        # f.write(etree.tostring(doc, encoding="unicode", pretty_print=True))
        # BeautifulSoup:
        # f.write(BeautifulSoup(rt_obj._repr_html_(), features="lxml").prettify())
        f.write(rt_obj._repr_html_())
    if hasattr(rt_obj, "_T"):  # dump and validate transpose representation
        with open(fn, "w") as f:
            f.write(rt_obj._repr_html_())

    # validate the dumped HTML files are valid HTML
    errors = checker.validate()

    assert not len(errors), (
        f"Checker failed.\nChecker command: {checker}\nGot the following errors:\n"
        + "\n".join(errors)
    )
