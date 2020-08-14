# -*- coding: utf-8 -*-
from .bench_numpy import gen_farange
from .runner import benchmark as bench

from ..rt_categorical import Categorical


def gen_singlekey_cat():
    for farr in gen_farange():
        yield Categorical(farr)


def gen_multikey_cat():
    for farr in gen_farange():
        yield Categorical([farr, farr])


# TODO add benchmarks around parital overlap and no overlap
# ----------------------------------------------------
@bench(benchmark_params={"c": gen_singlekey_cat()})
def bench_singlekey_cat_isin(c):
    c.isin(c)


# ----------------------------------------------------
@bench(benchmark_params={"c": gen_multikey_cat()})
def bench_multikey_cat_isin(c):
    c.isin(c)
