import riptable

# Standardized riptable configuration settings applied when doing docstring validation.

# Standardize on these display settings when executing examples
riptable.Display.FORCE_REPR = True  # Don't auto-detect console dimensions, just use CONSOLE_X/Y
riptable.Display.options.COL_MAX = 1_000_000  # display all Dataset columns (COL_ALL is incomplete)
riptable.Display.options.E_MAX = 100_000_000  # render up to 100MM before using scientific notation
riptable.Display.options.P_THRESHOLD = 0  # truncate small decimals, rather than scientific notation
riptable.Display.options.NUMBER_SEPARATOR = True  # put commas in numbers
riptable.Display.options.HEAD_ROWS = 3
riptable.Display.options.TAIL_ROWS = 3
