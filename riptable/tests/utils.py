#! /usr/bin/env python

import sys, os
from contextlib import contextmanager
from enum import IntEnum


__all__ = [
    'LikertDecision',
    'redirectStderr2StdoutCtx',
    "redirectStdoutCtx",
    "redirectStdoutAndErrorCtx",
    'chdirCtx',
    'new_array_function',
]


# TODO refactor LikerDecision to ISO3166CountryCodes
# Instead of Likert scale, which can be confusing when considering categorical sort order,
# use countries from ISO-3166 country codes as a replacement
class LikertDecision(IntEnum):
    """A Likert scale with the typical five-level Likert item format."""

    StronglyAgree = 44
    Agree = 133
    Disagree = 75
    StronglyDisagree = 1
    NeitherAgreeNorDisagree = 144


@contextmanager
def redirectStderr2StdoutCtx():
    origStderr = sys.stderr
    sys.stderr = sys.stdout
    try:
        yield
    except:
        sys.stderr = origStderr
        sys.stdout.flush()
        raise
    else:
        sys.stderr = origStderr
        sys.stdout.flush()


@contextmanager
def redirectStdoutCtx(filenameOrStream, mode='w'):
    if filenameOrStream is None:
        yield sys.stdout
        return
    if hasattr(filenameOrStream, "write"):
        f = filenameOrStream
    else:
        f = open(filenameOrStream, mode)
    origStdout = sys.stdout
    sys.stdout = f
    try:
        yield f
    except:
        sys.stdout = origStdout
        if f != filenameOrStream:
            f.close()
        raise
    else:
        sys.stdout = origStdout
        if f != filenameOrStream:
            f.close()


@contextmanager
def redirectStdoutAndErrorCtx(filenameOrStream, mode='w'):
    if filenameOrStream is None:
        yield sys.stdout
        return
    if hasattr(filenameOrStream, "write"):
        f = filenameOrStream
    else:
        f = open(filenameOrStream, mode)
    origStdout = sys.stdout
    origStderr = sys.stderr
    sys.stdout = f
    sys.stderr = f
    try:
        yield f
    except:
        sys.stderr = origStderr
        sys.stdout = origStdout
        if f != filenameOrStream:
            f.close()
        raise
    else:
        sys.stderr = origStderr
        sys.stdout = origStdout
        if f != filenameOrStream:
            f.close()


@contextmanager
def chdirCtx(newdir):
    """
    print os.getcwd()
    with chdirCtx(newdir) as (od, nd):
      print od, nd, os.getcwd()
    print os.getcwd()
    """
    if not os.path.isdir(newdir):
        raise IOError("chdirCtx(%r): not a directory" % newdir)
    olddir = os.getcwd()
    try:
        os.chdir(newdir)
        yield (olddir, newdir)
    except:
        os.chdir(olddir)
        raise
    else:
        os.chdir(olddir)


@contextmanager
def new_array_function(kls: type):
    """A context manager that allows using the new array function code path given a class `kls`
    that has a feature flag ``NEW_ARRAY_FUNCTION_ENABLED`` to enable the NEP 18 array function
    machinery."""
    try:
        kls.NEW_ARRAY_FUNCTION_ENABLED = True
        yield
    finally:
        kls.NEW_ARRAY_FUNCTION_ENABLED = False
