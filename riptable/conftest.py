import pytest
import numpy
import pandas
import riptable as rt


def get_doctest_dataset_data():
    return {
        'ds_simple_1': rt.Dataset({'A': [0, 1, 6, 7], 'B': [1.2, 3.1, 9.6, 21]}),
        'ds_simple_2': rt.Dataset({'X': [0, 1, 6, 9], 'C': [2.4, 6.2, 19.2, 53]}),
        'ds_complex_1': rt.Dataset(
            {'A': [0, 6, 9, 11], 'B': ['Q', 'R', 'S', 'T'], 'C': [2.4, 6.2, 19.2, 25.9]}
        ),
        'ds_complex_2': rt.Dataset(
            {
                'A': [0, 1, 6, 10],
                'B': ['Q', 'R', 'R', 'T'],
                'E': [1.5, 3.75, 11.2, 13.1],
            }
        ),
    }


@pytest.fixture(autouse=True)
def docstring_imports(doctest_namespace):
    doctest_namespace['np'] = doctest_namespace['numpy'] = numpy
    doctest_namespace['pd'] = doctest_namespace['pandas'] = pandas
    doctest_namespace['rt'] = doctest_namespace['riptable'] = rt


@pytest.fixture(autouse=True)
def docstring_merge_datasets(doctest_namespace):
    doctest_namespace.update(get_doctest_dataset_data())
