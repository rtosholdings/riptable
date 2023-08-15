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
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))


# -- Project information -----------------------------------------------------

project = "riptable"
copyright = "2022, rtosholdings"
author = "rtosholdings"

# The full version, including alpha/beta/rc tags
release = "1.3"


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
]

# Napoleon settings
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


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# TJD -- needed or sphinx fails near final stages
master_doc = "index"

# Autoapi settings

autoapi_dirs = ["../../riptable"]
autoapi_ignore = ["*test*", "*benchmarks*"]

# When using AutoAPI, use autoapi_template_dir (see below).
# templates_path = ["_templates"]

# https://sphinx-autoapi.readthedocs.io/en/latest/how_to.html#customise-templates
autoapi_template_dir = "_autoapi_templates"

# Setting autoapi_add_toctree_entry = False prevents a table of contents entry for the
# API Reference (/riptable/riptable/autoapi/index.html) from being automatically created.
# See https://sphinx-autoapi.readthedocs.io/en/latest/how_to.html#how-to-remove-the-index-page.
# That API Reference page doesn't seem to allow any sidebar TOC entries.
# Instead, manually add a link to /autoapi/riptable/index in /docs/source/index.rst as
# suggested at the sphinx-autoapi link above. This link goes to /autoapi/riptable/index.html
# -- a page that is generated using an AutoAPI template, which can be customized (see above).
autoapi_add_toctree_entry = False

# Order members by their type then alphabetically.
# https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html#confval-autoapi_member_order
autoapi_member_order = "groupwise"

# Remove typehints from html
autodoc_typehints = "none"

# Suppress Sphinx build warnings

suppress_warnings = ["autodoc"]

# Default role -- defines what to do with text surrounded by single backticks:
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-default_role
# py:obj references Python objects:
# https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#python-roles

default_role = "py:obj"

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
