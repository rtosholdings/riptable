# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from datetime import datetime
import os
import sys
import typing

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT_PATH = "../../"
sys.path.insert(0, os.path.abspath(ROOT_PATH))

# -- Helper functions ---------------------------------------------------------


def parse_commented_strings(iter: typing.Iterable[str]) -> list[str]:
    stripped = [line.split("#")[0].strip() for line in iter]
    return [line for line in stripped if line]


def parse_filters(filterpath: str) -> typing.Tuple[typing.Optional[list[str]], typing.Optional[list[str]]]:
    includes = []
    excludes = []
    filters = parse_commented_strings(open(filterpath))
    for filter in filters:
        if len(filter) > 1:
            if filter[0] == "+":
                includes.append(filter[1:])
                continue
            if filter[0] == "-":
                excludes.append(filter[1:])
                continue
        raise ValueError("Unexpected filter: " + filter)
    return includes if includes else None, excludes if excludes else None


# Load the full config from pyproject.toml, if found.
PYPROJ_CONFIG = {}

PYPROJ_TOML_PATH = os.path.join(ROOT_PATH, "pyproject.toml")
with open(PYPROJ_TOML_PATH, "rb") as f:
    PYPROJ_CONFIG = tomllib.load(f)

# Extract common sub-configs.
PYPROJ_PROJECT = PYPROJ_CONFIG.get("project", {})
PYPROJ_VALIDATE_DOCSTRINGS = PYPROJ_CONFIG.get("tool", {}).get("validate_docstrings", {})

# -- Project information -----------------------------------------------------

project = PYPROJ_PROJECT["name"]
author = PYPROJ_PROJECT["authors"][0]["name"]
copyright = f"{datetime.now().year}, {author}"
# Leave the release version unspecified, for now. Tricky to obtain dynamically.
# release = "1.x"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "sphinx_design",
]

# Put the list of valid time zones into docstrings using rst_prolog.
from riptable import TimeZone

tz_list = 'Supported timezones: "' + '", "'.join(TimeZone.valid_timezones) + '"'
current_tzs_note = f"{tz_list}. To see supported timezones, use ``rt.TimeZone.valid_timezones``."


rst_prolog = f"""
.. |rtosholdings_docs| replace:: {os.getenv("RTOSHOLDINGS_DOCS", "rtosholdings-docs@sig.com")}
.. |To see supported timezones, use ``rt.TimeZone.valid_timezones``.| replace:: {current_tzs_note}
"""


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes. There's also https://sphinx-themes.org/.
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# CSS file -- copied from NumPy; makes the landing page look better.
html_css_files = ["riptable.css"]

# TJD -- needed or sphinx fails near final stages
master_doc = "index"


# -- Napoleon configuration --------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None


# -- AutoAPI configuration --------------------------------------------------

autoapi_dirs = [os.path.join(ROOT_PATH, "riptable")]
autoapi_ignore = [
    "*/riptable/benchmarks/*",
    "*/riptable/hypothesis_tests/*",
    "*/riptable/test_tooling_integration/*",
    "*/riptable/testing/*",
    "*/riptable/tests/*",
    "*/riptable/Utils/rt_test*.py",
]

# Sphinx template dir. Sphinx needs this setting, and the AutoAPI extension needs the one below.
templates_path = ["_autoapi_templates"]

# AutoAPI-specific template dir. Any modified AutoApI templates go here.
# The default templates live in /site-packages/autoapi/templates/python.
# https://sphinx-autoapi.readthedocs.io/en/latest/how_to.html#customise-templates
autoapi_template_dir = "_autoapi_templates"

# When autoapi_add_toctree_entry=True, an index page is automatically generated that
# can't have a TOC added to the sidebar. To get around that, set it to False. A different
# index page is then created that uses the AutoAPI templates, which can be modified to put
# a TOC in the sidebar. See:
# https://sphinx-autoapi.readthedocs.io/en/latest/how_to.html#how-to-remove-the-index-page
# https://sphinx-autoapi.readthedocs.io/en/latest/how_to.html#customise-templates
autoapi_add_toctree_entry = False

# Order members by their type then alphabetically.
# https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html#confval-autoapi_member_order
autoapi_member_order = "groupwise"

# Remove typehints from html.
autodoc_typehints = "none"

# Suppress some specific Sphinx build warnings.
suppress_warnings = ["autodoc"]

# Default role -- defines how to link text surrounded by single backticks:
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-default_role
# py:obj references Python objects:
# https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#python-roles
default_role = "py:obj"

# Skip private functions except for those on an "include" list.
# https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html#event-autoapi-skip-member


def is_default_excluded(fullname: str) -> bool:
    """Indicates whether the fullname is excluded from documentation by default."""
    for name in fullname.split("."):
        # Exclude any private names by default.
        if name.startswith("_"):
            return True
    # Exclude any names that are defined in local/global context (confuses numpydoc name parsing)
    if ".<locals>" in fullname or ".<globals>" in fullname:
        return True
    return False


DOCSTRING_FILTERS_TXT = os.path.join(ROOT_PATH, PYPROJ_VALIDATE_DOCSTRINGS.get("filters", "?filters?"))
INCLUDES, EXCLUDES = parse_filters(DOCSTRING_FILTERS_TXT)


def is_included(fullname: str) -> typing.Tuple(bool, bool):
    """Indicates whether the name should be included in documentation."""

    def matches(name, namelist):
        for n in namelist:
            if name.startswith(n):
                if len(name) == len(n) or name[len(n)] == ".":
                    return True
        return False

    do_default_include = not is_default_excluded(fullname)

    do_include = do_default_include
    if INCLUDES and matches(fullname, INCLUDES):
        do_include = True
    if EXCLUDES and matches(fullname, EXCLUDES):
        do_include = False

    return do_include, do_default_include


def name_filter(app, what, name, obj, skip, options):
    included, default_included = is_included(name)

    do_log = included != default_included  # only announce changes from the default
    if do_log:
        names_logged = globals().setdefault("_name_filter_NAMES_LOGGED", set())
        if name not in names_logged:
            print(f"name_filter: {name} is " + ("" if included else "not ") + "included")
            names_logged.add(name)

    return None if included else True


def setup(sphinx):
    sphinx.connect("autoapi-skip-member", name_filter)


# --Intersphinx configuration--------------------------------------------------

# Turn off SSL verification so Intersphinx can reach the endpoints below ------
tls_verify = False

intersphinx_mapping = {
    "pyarrow": ("https://arrow.apache.org/docs", None),
    "numba": ("https://numba.pydata.org/numba-doc/latest", None),
    "neps": ("https://numpy.org/neps", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "imageio": ("https://imageio.readthedocs.io/en/stable", None),
    "skimage": ("https://scikit-image.org/docs/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "scipy-lecture-notes": ("https://scipy-lectures.org", None),
    "pytest": ("https://docs.pytest.org/en/stable", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "numpy-tutorials": ("https://numpy.org/numpy-tutorials", None),
    "numpydoc": ("https://numpydoc.readthedocs.io/en/latest", None),
    "dlpack": ("https://dmlc.github.io/dlpack/latest", None),
}
