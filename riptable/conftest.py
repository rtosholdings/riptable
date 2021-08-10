import pytest
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
    import numpy

    doctest_namespace['np'] = doctest_namespace['numpy'] = numpy
    doctest_namespace['rt'] = doctest_namespace['riptable'] = rt

    # Optional dependencies.
    import pandas
    doctest_namespace['pd'] = doctest_namespace['pandas'] = pandas


@pytest.fixture(autouse=True)
def docstring_merge_datasets(doctest_namespace):
    doctest_namespace.update(get_doctest_dataset_data())


@pytest.fixture(scope='session', autouse=True)
def register_null_log_handler():
    """
    Session-level fixture that installs a top-level log handler (or formatter) at the DEBUG level (or anything higher than NOTSET).
    It formats the messages in the typical way then just throws away the result. The idea is to just
    force all logging code to run, even if guarded with something like ``if logger.isEnabledFor(logging.DEBUG):``
    so we exercise that code within the tests. This helps guard against bad logging code that's only
    discovered when debug-level logging is enabled.
    """

    import logging

    class RenderingNullHandler(logging.Handler):
        """
        A ``logging.Handler`` which forces messages to be formatted using the default formatter
        to verify formatting strings are correct, then discards the formatted messages.
        """

        def __init__(self):
            """
            Initialize the instance.
            """
            logging.Handler.__init__(self)

        def emit(self, record):
            """
            Emit a record. Just forces the message to be formatted using the default
            formatter, then discards the result.
            """
            _ = self.format(record)

    # Create an instance of the RenderingNullHandler then register it
    # at the top level for all levels higher than logging.NOTSET.
    rendering_handler = RenderingNullHandler()
    root_logger = logging.getLogger()
    root_logger.addHandler(rendering_handler)
    root_logger.setLevel(logging.NOTSET + 1)
