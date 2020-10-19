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

setup(
    name = package_name,
    packages = [package_name],
    use_scm_version = {
        'version_scheme': 'post-release',
        'write_to': 'riptable/_version.py',
        'write_to_template': '__version__ = "{version}"',
    },
    setup_requires=['setuptools_scm'],
    package_dir = {package_name: 'riptable'},
    description = 'Python Package for riptable studies framework',
    author = 'RTOS Holdings',
    author_email = 'thomasdimitri@gmail.com',
    url="https://github.com/rtosholdings/riptable",
    install_requires=['numpy','riptide_cpp>=1.6.13','ansi2html','numba','python-dateutil'],
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
