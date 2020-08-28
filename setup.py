from setuptools import setup, find_packages, Extension, Command
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
import os
import sys
import pathlib
import datetime
import logging
import re

package_name='riptable'
# N.B. Need to read the file and use regex to get the version number because I can't import riptable
#      here before pip has chance to know the dependencies and install them.
with open('riptable/_version.py', 'r') as f:
    version = re.search('\d+\.\d+\.\d+([ab]\d+|)', f.readline()).group()

setup(
    name = package_name,
    packages = [package_name],
    package_dir = {package_name: 'riptable'},
    version = version,
    description = 'Python Package for riptable studies framework',
    author = 'RTOS Holdings',
    author_email = 'thomasdimitri@gmail.com',
    url="https://github.com/rtosholdings/riptable",
    install_requires=['numpy','riptide_cpp','ansi2html','numba','python-dateutil'],
    include_package_data=True,
    package_data={package_name: ['tests/*','tests/test_files/*', 'benchmarks/*', 'hypothesis_tests/*','Utils/*', 'test_tooling_integration/*']},
    classifiers=[
         "Development Status :: 4 - Beta",
         "Programming Language :: Python :: 3",
         "Programming Language :: Python :: 3.6",
         "Programming Language :: Python :: 3.7",
         "Programming Language :: Python :: 3.8",
         "License :: OSI Approved :: BSD License",
         "Operating System :: OS Independent",
    ]
)
