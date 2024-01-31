#!/usr/bin/env python3
"""
Analyze docstrings to detect errors.

If no argument is provided, it does a quick check of docstrings and returns
a csv with all API functions and results of basic checks.

If a function or method is provided in the form "riptable.function",
"riptable.module.class.method", etc. a list of all errors in the docstring for
the specified function or method.

Usage::
    $ ./validate_docstrings.py
    $ ./validate_docstrings.py riptable.FastArray.get_name
"""

# Adapted from https://github.com/pandas-dev/pandas/blob/main/scripts/validate_docstrings.py
# Module walking inspired by https://github.com/pyvista/numpydoc-validation/blob/main/numpydoc_validation/_validate.py

from __future__ import annotations

import argparse
import doctest
import enum
import functools
import importlib
import inspect
import io
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import typing
import warnings

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import matplotlib
import matplotlib.pyplot as plt
import numpy
from numpydoc.docscrape import get_doc_object
from numpydoc.validate import (
    error as npd_error,
    Validator,
    validate,
)

import riptable
from riptable.Utils.common import cached_weakref_property

# With template backend, matplotlib plots nothing
matplotlib.use("template")

# Apply riptable docstring configuration for examples.
from _docstring_config import *


ERROR_MSGS = {
    "GL99": "Error parsing docstring: {doc_parse_error}",
    "GL98": "Private classes ({mentioned_private_classes}) should not be " "mentioned in public docstrings",
    "GL97": "Use 'array-like' rather than 'array_like' in docstrings.",
    "GL96": "Warning validating docstring: {doc_validation_warning}",
    "GL95": "Expected fail docstring not failed; remove from xfails list",
    "SA99": "{reference_name} in `See Also` section does not need `riptable` " "prefix, use {right_reference} instead.",
    "EX99": "Examples do not pass tests:\n{doctest_log}",
    "EX98": "flake8 error: line {line_number}, col {col_number}: {error_code} {error_message}",
    "EX97": "Do not import {imported_library}, as it is imported automatically for the examples",
    "EX96": "flake8 warning: line {line_number}, col {col_number}: {error_code} {error_message}",
    "EX95": "black format error:\n{error_message}",
}

OUT_FORMAT_OPTS = "default", "json", "actions"

NAMES_FROM_OPTS = "module", "rst"

IGNORE_VALIDATION = {
    # Styler methods are Jinja2 objects who's docstrings we don't own.
    # "Styler.env",
    # "Styler.template_html",
    # "Styler.template_html_style",
    # "Styler.template_html_table",
    # "Styler.template_latex",
    # "Styler.template_string",
    # "Styler.loader",
}

PRIVATE_CLASSES = [
    # "NDFrame",
    # "IndexOpsMixin",
]

IMPORT_CONTEXT = {
    "np": numpy,
    "rt": riptable,
}


def riptable_error(code, **kwargs):
    """
    Copy of the numpydoc error function, since ERROR_MSGS can't be updated
    with our custom errors yet.
    """
    return (code, ERROR_MSGS[code].format(**kwargs)) if code in ERROR_MSGS else npd_error(code, **kwargs)


def get_api_items(api_doc_fd):
    """
    Yield information about all public API items.

    Parse api.rst file from the documentation, and extract all the functions,
    methods, classes, attributes... This should include all riptable public API.

    Parameters
    ----------
    api_doc_fd : file descriptor
        A file descriptor of the API documentation page, containing the table
        of contents with all the public API.

    Yields
    ------
    name : str
        The name of the object (e.g. 'riptable.FastArray.get_name).
    func : function
        The object itself. In most cases this will be a function or method,
        but it can also be classes, properties, cython objects...
    section : str
        The name of the section in the API page where the object item is
        located.
    subsection : str
        The name of the subsection in the API page where the object item is
        located.
    """
    current_module = "riptable"
    previous_line = current_section = current_subsection = ""
    position = None
    for line in api_doc_fd:
        line_stripped = line.strip()
        if len(line_stripped) == len(previous_line):
            if set(line_stripped) == set("-"):
                current_section = previous_line
                continue
            if set(line_stripped) == set("~"):
                current_subsection = previous_line
                continue

        if line_stripped.startswith(".. currentmodule::"):
            current_module = line_stripped.replace(".. currentmodule::", "").strip()
            continue

        if line_stripped == ".. autosummary::":
            position = "autosummary"
            continue

        if position == "autosummary":
            if line_stripped == "":
                position = "items"
                continue

        if position == "items":
            if line_stripped == "":
                position = None
                continue
            if line_stripped in IGNORE_VALIDATION:
                continue
            func = importlib.import_module(current_module)
            for part in line_stripped.split("."):
                func = getattr(func, part)

            yield (
                f"{current_module}.{line_stripped}",
                func,
                current_section,
                current_subsection,
            )

        previous_line = line_stripped


class RiptableDocstring(Validator):
    def __init__(self, func_name: str, doc_obj=None, verbose: bool = False) -> None:
        self.func_name = func_name
        self.verbose = verbose
        if doc_obj is None:
            doc_obj = get_doc_object(Validator._load_obj(func_name))
        super().__init__(doc_obj)

    @property
    def name(self):
        return self.func_name

    @property
    def mentioned_private_classes(self):
        return [klass for klass in PRIVATE_CLASSES if klass in self.raw_doc]

    @property
    def examples_errors(self):
        flags = doctest.NORMALIZE_WHITESPACE | doctest.IGNORE_EXCEPTION_DETAIL
        finder = doctest.DocTestFinder()
        runner = doctest.DocTestRunner(
            optionflags=flags,
            verbose=self.verbose,  # set this explicitly to avoid default inspection of sys.argv
        )
        error_msgs = ""
        current_dir = set(os.listdir())
        tempdir = pathlib.Path("tempdir")  # special reserved directory for temporary files; will be deleted per test
        for test in finder.find(self.raw_doc, self.name, globs=IMPORT_CONTEXT):
            tempdir.mkdir()
            f = io.StringIO()
            failed_examples, total_examples = runner.run(test, out=f.write)
            if failed_examples:
                error_msgs += f.getvalue()
            shutil.rmtree(tempdir)
        leftovers = set(os.listdir()).difference(current_dir)
        if leftovers:
            for leftover in leftovers:
                path = pathlib.Path(leftover).resolve()
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.is_file():
                    path.unlink(missing_ok=True)
            error_msgs += (
                f"The following files/dirs were leftover from the doctest: " f"{leftovers}. Please use # doctest: +SKIP"
            )
        return error_msgs

    @property
    def examples_source_code(self):
        lines = doctest.DocTestParser().get_examples(self.raw_doc)
        return [line.source for line in lines]

    def validate_pep8(self):
        if not self.examples:
            return

        # F401 is needed to not generate flake8 errors in examples
        # that do not use the imported context
        content = ""
        for k, v in IMPORT_CONTEXT.items():
            content += f"import {v.__name__} as {k}  # noqa: F401\n"
        content += "".join((*self.examples_source_code,))

        error_messages = []

        file = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
        try:
            file.write(content)
            file.flush()
            cmd = [
                "python",
                "-m",
                "flake8",
                "--format=%(row)d\t%(col)d\t%(code)s\t%(text)s",
                file.name,
            ]
            response = subprocess.run(cmd, capture_output=True, text=True)
            if response.stderr:
                stderr = response.stderr.strip("\n")
                error_messages.append(f"0\t1\tERROR\t{stderr}")
            stdout = response.stdout
            stdout = stdout.replace(file.name, "")
            messages = stdout.strip("\n").splitlines()
            if messages:
                error_messages.extend(messages)
        finally:
            file.close()
            os.remove(file.name)

        for error_message in error_messages:
            line_number, col_number, error_code, message = error_message.split("\t", maxsplit=3)
            # Subtract the added context lines from line counter
            yield error_code, message, int(line_number) - len(IMPORT_CONTEXT), int(col_number)

    def validate_format(self):
        if not self.examples:
            return

        content = "".join((*self.examples_source_code,))

        error_messages = []

        file = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
        try:
            file.write(content)
            file.flush()
            cmd = [
                "python",
                "-m",
                "black",
                "--quiet",
                "--diff",
                file.name,
            ]
            response = subprocess.run(cmd, capture_output=True, text=True)
            if response.stderr:
                stderr = response.stderr.strip("\n")
                error_messages.extend(stderr)
            stdout = response.stdout
            stdout = stdout.replace(file.name, "<example>")
            messages = stdout.strip("\n")
            if messages:
                error_messages.append(messages)
        finally:
            file.close()
            os.remove(file.name)

        for error_message in error_messages:
            yield error_message

    def non_hyphenated_array_like(self):
        return "array_like" in self.raw_doc


def riptable_validate(
    func_name: str,
    errors: typing.Optional(list[str]) = None,
    not_errors: typing.Optional(list[str]) = None,
    flake8_errors: typing.Optional(list[str]) = None,
    flake8_not_errors: typing.Optional(list[str]) = None,
    xfails: typing.Optional(list[str]) = None,
    verbose: bool = False,
):
    """
    Call the numpydoc validation, and add the errors specific to riptable.

    Parameters
    ----------
    func_name : str
        Name of the object of the docstring to validate.

    Returns
    -------
    dict
        Information about the docstring and the errors found.
    """
    func_obj = Validator._load_obj(func_name)
    doc_parse_error = None

    with warnings.catch_warnings(record=True) as doc_warnings:
        try:
            doc_obj = get_doc_object(func_obj, doc=func_obj.__doc__)
        except ValueError as ex:
            doc_parse_error = str(ex)
            doc_obj = get_doc_object(func_obj, doc="")
        doc = RiptableDocstring(func_name, doc_obj, verbose=verbose)
        result = validate(doc_obj)
        if doc_parse_error:
            result["errors"].insert(0, riptable_error("GL99", doc_parse_error=doc_parse_error))

        # Remove incorrect failures for undocumented enum class parameters
        if inspect.isclass(func_obj) and issubclass(func_obj, enum.Enum):
            result["errors"] = [item for item in result["errors"] if item not in doc.parameter_mismatches]

        mentioned_errs = doc.mentioned_private_classes
        if mentioned_errs:
            result["errors"].append(riptable_error("GL98", mentioned_private_classes=", ".join(mentioned_errs)))

        # If documented function/method docstring exists...
        if doc.raw_doc and doc.is_function_or_method:
            # If there exist any sections (it appears to be a public API docstring)...
            if not doc.returns and "RT01" not in set(x[0] for x in result["errors"]):
                result["errors"].append(riptable_error("RT01"))

        if doc.see_also:
            for rel_name in doc.see_also:
                if rel_name.startswith("riptable."):
                    result["errors"].append(
                        riptable_error(
                            "SA99",
                            reference_name=rel_name,
                            right_reference=rel_name[len("riptable.") :],
                        )
                    )

        def matches(test: str, matches: list[str]):
            for match in matches:
                if test.startswith(match):
                    return True
            return False

        result["examples_errs"] = ""
        if doc.examples:
            result["examples_errs"] = doc.examples_errors
            if result["examples_errs"]:
                result["errors"].append(riptable_error("EX99", doctest_log=result["examples_errs"]))

            for error_code, error_message, line_number, col_number in doc.validate_pep8():
                result["errors"].append(
                    riptable_error(
                        "EX98"
                        if error_code == "ERROR"
                        or (
                            (not flake8_errors or (flake8_errors and matches(error_code, flake8_errors)))
                            and (not flake8_not_errors or not matches(error_code, flake8_not_errors))
                        )
                        else "EX96",
                        error_code=error_code,
                        error_message=error_message,
                        line_number=line_number,
                        col_number=col_number,
                    )
                )
            examples_source_code = "".join(doc.examples_source_code)
            for wrong_import in [v.__name__ for v in IMPORT_CONTEXT.values()]:
                if re.search(f"import {wrong_import}\W+", examples_source_code):
                    result["errors"].append(riptable_error("EX97", imported_library=wrong_import))

            for error_message in doc.validate_format():
                result["errors"].append(
                    riptable_error(
                        "EX95",
                        error_message=error_message,
                    )
                )

        if doc.non_hyphenated_array_like():
            result["errors"].append(riptable_error("GL97"))

        plt.close("all")

    if doc_warnings:
        for doc_warning in doc_warnings:
            doc_validation_warning = warnings.formatwarning(
                doc_warning.message, doc_warning.category, doc_warning.filename, doc_warning.lineno
            ).strip()
            result["errors"].append(riptable_error("GL96", doc_validation_warning=doc_validation_warning))

    if errors or not_errors:
        filtered_errors = []
        for err_code, err_desc in result["errors"]:
            if not (errors and not matches(err_code, errors) or not_errors and matches(err_code, not_errors)):
                filtered_errors.append((err_code, err_desc))
        result["errors"] = filtered_errors

    result["xerrors"] = []
    if xfails and func_name in xfails:
        if result["errors"]:
            result["xerrors"] = result["errors"]
            result["errors"] = []
        elif not result["errors"]:
            result["errors"].append(riptable_error("GL95"))

    return result


def resolve_obj(obj: object) -> object:
    """
    Resolves obj by unwrapping any known wrappers to the most innermost object.
    """
    while True:
        obj = inspect.unwrap(obj)
        if isinstance(obj, property):
            obj = obj.fget
        elif isinstance(obj, functools.cached_property) or isinstance(obj, cached_weakref_property):
            obj = obj.func
        else:
            return obj


def get_all_objects(root: object, modulename: str) -> typing.Iterable[(str, object)]:
    """
    Retrieves a sequence of resolved unique names and objects that are part of modulename found within root.
    """
    found = {}

    def populate_all_objects(base: object):
        for name in dir(base):
            # Resolve to the innermost object.
            obj = resolve_obj(getattr(base, name))

            # Ignore any system dunder names
            if name.startswith("__"):
                continue

            # Collect objects that are part of this module and have a qualified name.
            modname = getattr(obj, "__module__", None)
            qualname = getattr(obj, "__qualname__", None)
            if modname and qualname and modname.startswith(modulename):
                fullname = modname + "." + qualname
                if not fullname in found:
                    found[fullname] = obj
                    if inspect.isclass(obj):
                        populate_all_objects(obj)

    populate_all_objects(root)

    return found.items()


def get_module_items(modulename: str) -> list[str]:
    module = importlib.import_module(modulename)
    items = [(fullname, obj, None, None) for fullname, obj in get_all_objects(module, modulename)]
    return items


def is_default_excluded(fullname: str) -> bool:
    """Indicates whether the fullname is excluded from validation by default."""
    for name in fullname.split("."):
        # Exclude any private names by default.
        if name.startswith("_"):
            return True
    # Exclude any names that are defined in local/global context (confuses numpydoc name parsing)
    if ".<locals>" in fullname or ".<globals>" in fullname:
        return True
    return False


def is_included(
    fullname: str,
    includes: typing.Optional(list[str]) = None,
    excludes: typing.Optional(list[str]) = None,
) -> bool:
    """Indicates whether the name should be included in validation."""

    def matches(name, namelist):
        for n in namelist:
            if name.startswith(n):
                if len(name) == len(n) or name[len(n)] == ".":
                    return True
        return False

    do_include = not is_default_excluded(fullname)
    if includes and matches(fullname, includes):
        do_include = True
    if excludes and matches(fullname, excludes):
        do_include = False
    return do_include


def validate_all(
    match: str,
    not_match: str = None,
    names_from: str = NAMES_FROM_OPTS[0],
    errors: typing.Optional(list[str]) = None,
    not_errors: typing.Optional(list[str]) = None,
    flake8_errors: typing.Optional(list[str]) = None,
    flake8_not_errors: typing.Optional(list[str]) = None,
    ignore_deprecated: bool = False,
    includes: typing.Optional(list[str]) = None,
    excludes: typing.Optional(list[str]) = None,
    xfails: typing.Optional(list[str]) = None,
    verbose: int = 0,
) -> dict:
    """
    Execute the validation of all docstrings, and return a dict with the
    results.

    Parameters
    ----------
    prefix : str or None
        If provided, only the docstrings that start with this pattern will be
        validated. If None, all docstrings will be validated.
    ignore_deprecated: bool, default False
        If True, deprecated objects are ignored when validating docstrings.

    Returns
    -------
    dict
        A dictionary with an item for every function/method... containing
        all the validation information.
    """
    result = {}
    seen = {}

    api_items = []

    if names_from == "rst":
        base_path = pathlib.Path(__file__).parent.parent
        api_doc_fnames = pathlib.Path(base_path, "doc", "source", "reference")
        for api_doc_fname in api_doc_fnames.glob("*.rst"):
            with open(api_doc_fname) as f:
                api_items += list(get_api_items(f))
    else:
        api_items.extend(get_module_items("riptable"))

    api_items = [item for item in api_items if is_included(item[0], includes=includes, excludes=excludes)]

    api_items.sort()

    match_re = re.compile(match) if match else None
    not_match_re = re.compile(not_match) if not_match else None

    processed = set()

    for func_name, _, section, subsection in api_items:
        if func_name in processed:
            continue
        processed.add(func_name)

        if match_re and not match_re.search(func_name) or not_match_re and not_match_re.search(func_name):
            if verbose > 1:
                print(f"Ignoring {func_name} not matching prefix {match}")
            continue
        if verbose:
            print(f"Validating {func_name}... ", end="", flush=True)
        doc_info = riptable_validate(
            func_name,
            errors=errors,
            not_errors=not_errors,
            flake8_errors=flake8_errors,
            flake8_not_errors=flake8_not_errors,
            xfails=xfails,
            verbose=False,  # don't generate verbose output for each test
        )
        if ignore_deprecated and doc_info["deprecated"]:
            if verbose > 1:
                print(f"Ignoring deprecated {func_name}")
            continue

        if verbose:

            def get_errlist(errs):
                return ",".join([c for c, _ in errs])

            status = (
                f"FAILED ({get_errlist(doc_info['errors'])})"
                if doc_info["errors"]
                else f"XFAILED ({get_errlist(doc_info['xerrors'])})"
                if doc_info["xerrors"]
                else "OK"
            )
            print(status)
            for err_code, err_desc in doc_info["errors"]:
                print(f'{doc_info["file"]}:{doc_info["file_line"]}: {func_name}: {err_code}: {err_desc}')
        result[func_name] = doc_info

        shared_code_key = doc_info["file"], doc_info["file_line"]
        shared_code = seen.get(shared_code_key, "")
        result[func_name].update(
            {
                "in_api": True,
                "section": section,
                "subsection": subsection,
                "shared_code_with": shared_code,
            }
        )

        seen[shared_code_key] = func_name

    return result


def print_validate_all_results(
    match: str,
    not_match: str = None,
    names_from: str = NAMES_FROM_OPTS[0],
    errors: typing.Optional(list[str]) = None,
    not_errors: typing.Optional(list[str]) = None,
    flake8_errors: typing.Optional(list[str]) = None,
    flake8_not_errors: typing.Optional(list[str]) = None,
    out_format: str = OUT_FORMAT_OPTS[0],
    ignore_deprecated: bool = False,
    includes: typing.Optional(list[str]) = None,
    excludes: typing.Optional(list[str]) = None,
    xfails: typing.Optional(list[str]) = None,
    outfile: typing.IO = sys.stdout,
    outfailsfile: typing.Optional(typing.IO) = None,
    verbose: int = 0,
):
    if out_format not in OUT_FORMAT_OPTS:
        raise ValueError(f'Unknown output_format "{out_format}"')

    result = validate_all(
        match,
        not_match=not_match,
        names_from=names_from,
        errors=errors,
        not_errors=not_errors,
        flake8_errors=flake8_errors,
        flake8_not_errors=flake8_not_errors,
        ignore_deprecated=ignore_deprecated,
        includes=includes,
        excludes=excludes,
        xfails=xfails,
        verbose=verbose,
    )

    if verbose:
        print("Results:")

    if out_format == "json":
        json.dump(result, outfile, indent=2)
    else:
        prefix = "##[error]" if out_format == "actions" else ""
        for name, res in result.items():
            for err_code, err_desc in res["errors"] + res["xerrors"]:
                outfile.write(f'{prefix}{res["file"]}:{res["file_line"]}: {name}: {err_code}: {err_desc}\n')

    errors_count = 0
    xerrors_count = 0

    for name, res in result.items():
        error_count = 1 if res["errors"] else 0
        xerror_count = 1 if res["xerrors"] else 0
        if error_count or xerror_count:
            print(name, file=outfailsfile)
        errors_count += error_count
        xerrors_count += xerror_count

    if verbose:
        print(f"{len(result)} validated: {errors_count} failed, {xerrors_count} expected failed")
        print("Validation " + ("FAILED" if errors_count else ("XFAILED!" if xerrors_count else "OK!")))

    exit_status = 1 if errors_count else 0
    return exit_status


def print_validate_one_results(
    func_name: str,
    errors: typing.Optional(list[str]) = None,
    not_errors: typing.Optional(list[str]) = None,
    flake8_errors: typing.Optional(list[str]) = None,
    flake8_not_errors: typing.Optional(list[str]) = None,
    xfails: typing.Optional(list[str]) = None,
    outfile: typing.IO = sys.stdout,
    verbose: int = 0,
):
    def header(title, width=80, char="#"):
        full_line = char * width
        side_len = (width - len(title) - 2) // 2
        adj = "" if len(title) % 2 == 0 else " "
        title_line = f"{char * side_len} {title}{adj} {char * side_len}"

        return f"\n{full_line}\n{title_line}\n{full_line}\n\n"

    if verbose:
        print(f"Validating {func_name}... ", end="", flush=True)
    result = riptable_validate(
        func_name,
        errors=errors,
        not_errors=not_errors,
        flake8_errors=flake8_errors,
        flake8_not_errors=flake8_not_errors,
        xfails=xfails,
        verbose=True,  # generate verbose output for each test
    )
    exit_status = 1 if result["errors"] else 0
    if verbose:
        print(("XFAILED!" if result["xerrors"] else "OK!") if exit_status == 0 else "FAILED!")

    outfile.write(header(f"Docstring ({func_name})"))
    outfile.write(f"{result['docstring']}\n")

    outfile.write(header("Validation"))
    errors = result["errors"] if result["errors"] else result["xerrors"]
    if errors:
        outfile.write(f"{len(errors)} Errors found:\n")
        for err_code, err_desc in errors:
            desc = "Examples do not pass tests (see details below)" if err_code == "EX99" else err_desc
            outfile.write(f'\t{result["file"]}:{result["file_line"]}: {func_name}: {err_code}: {desc}\n')
    else:
        outfile.write(f'Docstring for "{func_name}" is OK.\n')

    if result["examples_errs"]:
        outfile.write(header("Doctests"))
        outfile.write(result["examples_errs"])

    return exit_status


def find_parent_dir_containing(filename: str) -> typing.Optional[str]:
    cur_dir = os.getcwd()
    while not os.path.exists(os.path.join(cur_dir, filename)):
        cur_dir = os.path.dirname(cur_dir)
        if not cur_dir:
            return None
    return cur_dir


def find_pyproject_toml() -> typing.Optional[str]:
    pyproj_toml_filename = "pyproject.toml"
    root_dir = find_parent_dir_containing(pyproj_toml_filename)
    return os.path.join(root_dir, pyproj_toml_filename) if root_dir else None


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


def main():
    """
    Main entry point. Call the validation for one or for all docstrings.
    """
    argparser = argparse.ArgumentParser(description="Validate riptable docstrings")
    argparser.add_argument(
        "function",
        nargs="?",
        default=None,
        help="Function or method to validate (e.g. riptable.FastArray.get_name) "
        "if not provided, all docstrings are validated and returned "
        "as JSON.",
    )
    argparser.add_argument(
        "--names-from",
        default=NAMES_FROM_OPTS[0],
        choices=NAMES_FROM_OPTS,
        help=f"Source of names when searching for all docstrings. It can be one of {str(NAMES_FROM_OPTS)[1:-1]} (default: '%(default)s').",
    )
    argparser.add_argument(
        "--format",
        default=OUT_FORMAT_OPTS[0],
        choices=OUT_FORMAT_OPTS,
        help="format of the output when validating "
        "multiple docstrings (ignored when validating one). "
        f"It can be one of {str(OUT_FORMAT_OPTS)[1:-1]} (default: '%(default)s').",
    )
    argparser.add_argument(
        "--match",
        default=None,
        help="Regex pattern for matching "
        "docstring names, in order to decide which ones "
        'will be validated. The match "FastArray" '
        "will make the script validate all the docstrings "
        'of methods containing "FastArray". It is '
        "ignored if function option is provided.",
    )
    argparser.add_argument(
        "--not-match",
        default=None,
        help="Regex pattern for not matching "
        "docstring names, in order to decide which ones "
        'will be validated. The not-match "mapping" '
        "will make the script validate all the docstrings "
        'of methods not containing "mapping". '
        "Any matches are performed first, then any not-matches are excluded. "
        "It is ignored if function option is provided.",
    )
    argparser.add_argument(
        "--errors",
        default=None,
        help="Comma separated "
        "list of error codes to validate. By default it "
        "validates all errors. Ignored when validating "
        "a single docstring.",
    )
    argparser.add_argument(
        "--not-errors",
        default=None,
        help="Comma separated "
        "list of error codes not to validate. Empty by default. "
        "Ignored when validating a single docstring.",
    )
    argparser.add_argument(
        "--ignore-deprecated",
        default=False,
        action="store_true",
        help="If this flag is set, deprecated objects are ignored when validating all docstrings.",
    )
    argparser.add_argument(
        "--flake8-errors",
        default=None,
        help="Comma separated list of flake8 error codes to treat as errors. Others are treated as warnings.",
    )
    argparser.add_argument(
        "--flake8-not-errors",
        default=None,
        help="Comma separated list of flake8 error codes not to validate. Empty by default",
    )
    argparser.add_argument(
        "--example-prelude",
        default=None,
        help="Optional code to execute before the examples.",
    )
    argparser.add_argument(
        "--xfails",
        default=None,
        help="Path to file of known expected failures.",
    )
    argparser.add_argument(
        "--filters",
        default=None,
        help="Path to file of include/exclude filters.",
    )
    argparser.add_argument(
        "--out",
        "-o",
        default=None,
        type=str,
        help="Output file path, else use stdout.",
    )
    argparser.add_argument(
        "--out-fails",
        default=None,
        type=str,
        help="Output path listing failures.",
    )
    argparser.add_argument(
        "--verbose",
        "-v",
        default=0,
        action="count",
        help="Emit verbose progress output. Specify multiple times for more verbosity.",
    )

    args = argparse.Namespace()

    pyproj_toml_path = find_pyproject_toml()
    if pyproj_toml_path:
        with open(pyproj_toml_path, "rb") as f:
            pyproj_toml = tomllib.load(f)
        config = pyproj_toml.get("tool", {}).get("validate_docstrings", {})
        for k, v in config.items():
            setattr(args, k, v)

    argparser.parse_args(namespace=args)

    xfails = parse_commented_strings(open(args.xfails)) if args.xfails else None

    includes, excludes = parse_filters(args.filters) if args.filters else (None, None)

    bad = [n for n in xfails if n in excludes]
    if bad:
        raise ValueError(f"Excluded names also found in xfails list, resolve duplicate: {bad}")

    outfailsfile = open(args.out_fails, "w", encoding="utf-8", errors="backslashreplace") if args.out_fails else None

    with open(args.out, "w", encoding="utf-8", errors="backslashreplace") if args.out else open(
        sys.stdout.fileno(), "w", closefd=False
    ) as outfile:
        errors = args.errors.split(",") if args.errors is not None else None
        not_errors = args.not_errors.split(",") if args.not_errors is not None else None
        flake8_errors = args.flake8_errors.split(",") if args.flake8_errors is not None else None
        flake8_not_errors = args.flake8_not_errors.split(",") if args.flake8_not_errors is not None else None

        if args.function is None:
            return print_validate_all_results(
                match=args.match,
                not_match=args.not_match,
                names_from=args.names_from,
                errors=errors,
                not_errors=not_errors,
                flake8_errors=flake8_errors,
                flake8_not_errors=flake8_not_errors,
                out_format=args.format,
                ignore_deprecated=args.ignore_deprecated,
                includes=includes,
                excludes=excludes,
                xfails=xfails,
                outfile=outfile,
                outfailsfile=outfailsfile,
                verbose=args.verbose,
            )
        else:
            return print_validate_one_results(
                args.function,
                errors=errors,
                not_errors=not_errors,
                flake8_errors=flake8_errors,
                flake8_not_errors=flake8_not_errors,
                xfails=xfails,
                outfile=outfile,
                verbose=args.verbose,
            )


if __name__ == "__main__":
    sys.exit(main())
