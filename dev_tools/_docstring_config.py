import riptable
import contextlib


def _setup_display_config():
    """Initialize display config settings.
    Any options that can be modified should be set here, even if set to default values.
    """
    riptable.Display.FORCE_REPR = True  # Don't auto-detect console dimensions, just use CONSOLE_X/Y
    riptable.Display.options.CONSOLE_X = 150
    riptable.Display.options.COL_MAX = 1_000_000  # display all Dataset columns (COL_ALL is incomplete)
    riptable.Display.options.E_MAX = 100_000_000  # render up to 100MM before using scientific notation
    riptable.Display.options.P_THRESHOLD = 0  # truncate small decimals, rather than scientific notation
    riptable.Display.options.NUMBER_SEPARATOR = True  # put commas in numbers
    riptable.Display.options.HEAD_ROWS = 3
    riptable.Display.options.TAIL_ROWS = 3
    riptable.Display.options.ROW_ALL = False
    riptable.Display.options.COL_ALL = False
    riptable.Display.options.MAX_STRING_WIDTH = 15


def setup_init_config():
    """Initialize all config settings. Typically only done once."""
    _setup_display_config()


class ScopedExampleSetup(contextlib.AbstractContextManager):
    """Context manager to clean up after any changes made during example setup."""

    _CLEANUP_CALLBACKS = []

    @staticmethod
    def add_cleanup_callback(fn):
        ScopedExampleSetup._CLEANUP_CALLBACKS.append(fn)

    def __enter__(self) -> None:
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback) -> bool | None:
        callbacks = ScopedExampleSetup._CLEANUP_CALLBACKS
        ScopedExampleSetup._CLEANUP_CALLBACKS = []
        for callback in callbacks:
            callback()
        return super().__exit__(exc_type, exc_value, traceback)


def setup_for_examples(*configs: tuple[str]):
    """Applies specified config setups for an example.
    Configs are applied in order.
    Any modifications done here need to be undone by registering a cleanup task with ScopedExampleSetup.
    """

    for config in configs:
        if config == "struct-display":
            riptable.Display.options.CONSOLE_X = 120
            riptable.Display.options.HEAD_ROWS = 15
            riptable.Display.options.TAIL_ROWS = 15
            ScopedExampleSetup.add_cleanup_callback(_setup_display_config)  # reset all display configs.

        else:
            raise NotImplementedError(f"Unknown config, {config}")


# Initialize all config globally.
setup_init_config()
