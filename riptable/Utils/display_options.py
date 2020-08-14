"""Display options for formatting and displaying numeric values, datasets, and multisets."""
import os
from json import load, dump
from typing import Optional, Union
from .appdirs import user_config_dir
from ..rt_enum import TypeRegister, DisplayNumberSeparator

__all__ = ['DisplayOptions',]


class DisplayOptions(object):
    # Todo alz 20191125 - fix disappearing attributes from numpy docs
    # Following the numpy class docstring guide (https://numpy.org/devdocs/docs/howto_document.html#class-docstring),
    # the following attributes and methods do not appear in the rendered HTML output.
    # Investigate sphinx.ext.napoleon interaction with class variables.
    # Property attributes are known to render correctly.
    """Provides display options for:
      1) console screen for customizing width, height, and character buffers
      2) row, columns, and styles when displaying datasets and multisets
      3) formatting headers, footers, and general string widths
      4) formatting numeric types using scientific notation or specifying precision
      5) other miscellaneous display options such as prefixing group by column names

    Attributes
    ----------
    CONSOLE_X_BUFFER : int
        Number of characters for buffer width on console display (default 30).
    CONSOLE_X_HTML : int
        Number of characters for buffer width on HTML display (default 340).
    CONSOLE_X : int
        Number of characters for console display width (default 150).
    CONSOLE_Y : int
        Number of characters for console display height (default 25).
    HTML_DISPLAY : bool
        Toggle HTML display mode (default True).
    X_PADDING : int
        Number of characters for column buffer in console display (default 4).
    Y_PADDING : int
        Number of characters for row buffer in console display (default 3).
    ROW_ALL : bool
        Toggle display of all rows for Dataset, Multiset, and Struct objects (default False).
    COL_ALL : bool
        Toggle display of all columns for Dataset, Multiset, and Struct objects (default False).
    COL_MIN : int
        Minimum columns to display for Dataset, Multiset, and Struct objects (default 1).
    COL_MAX : int
        Maximum columns to display for Dataset, Multiset, and Struct objects (default 50).
    COL_T : int
        Number of transposed rows to display, which appear as columns for Dataset, Multiset, and Struct objects (default 8).
    HEAD_ROWS : int
        Number of rows to display when calling head on a Dataset, Multiset, or Struct object (default 15).
    TAIL_ROWS : int
        Number of rows to display when calling tail on a Dataset, Multiset, or Struct object (default 15).
    MAX_ROWS : int
        Maximum number of rows to display for Dataset, Multiset, and Struct objects (default 30).
    NO_STYLES : bool
        Toggle for colors in IPython console (default False).
        Note, may be difficult to see with light background.
    COLOR_MODE : DisplayColorMode or None, optional
        Color mode for display (default None).
    CUSTOM_COMPLETION = bool
        Toggle on for attribute completion results that show in alphanumeric key, attribute, then method ordering
        for ``Dataset``, ``Multiset``, ``Struct`` (default False).

        This will override the default IPython ``Completer._complete`` to a custom variant that allows custom completer
        dispatching using the ``IPython.utils.generics.complete_object`` hook while preserving the custom ordering.

        Caution, below are the side effects when toggling this on:
        - IPython ``use_jedi`` is set to False since this approach is currently incompatible with Jedi completion
        because the code is actually evaluated on TAB.
        - IPython ``Completer._complete`` is monkey patched to change use the custom completion that is backwards compatible
        with ``Completer._complete``, but allows preserving the order.
        - As of 20191218, if ``CUSTOM_COMPLETION`` is toggled on it results in a one-time registration of custom
        attribute completion per IPython session as opposed to supporting deregistration.
    MAX_HEADER_WIDTH : int
        Maximum number of characters for header strings in a Dataset, Multiset, or Struct object (default 15).
    MAX_FOOTER_WIDTH : int
        Maximum number of characters for footer strings in Dataset, Multiset, or Struct object  (default 15).
    MAX_STRING_WIDTH : int
        Maximum number of characters for all strings (default 15).
    PRECISION :
        Number of digits to display to the right of the decimal (default 2).
    E_PRECISION :
        Number of digits to display to the right of the decimal in scientific notation (default 3).
    E_THRESHOLD :
        Power of 10 at which the float flips to scientific notation 10**+/- (default 6).
    E_MIN : int or None, optional
        lower limit before going to scientific notation (default None).
    E_MAX : int or None, optional
        Upper limit before going to scientific notation (default None).
    P_THRESHOLD : int or None, optional
        Precision threshold for area in between - so small values don't display as zero (default None).
    NUMBER_SEPARATOR : bool
        Flag for separating thousands in floats and ints (default False).
    NUMBER_SEPARATOR_CHAR : DisplayNumberSeparator
        Character for separating `,`, `.`, or `_` (default DisplayNumberSeparator.Comma).
    GB_PREFIX : str
         Prefix for column names to indicate that they are groupby keys (default "*").

    Methods
    -------
    save_config
    load_config
    reset_config
    e_min
    e_max
    p_threshold
    no_colors

    See also
    --------
    DisplayTable : builds result tables with display options.

    DisplayColorMode : enumerates supported color modes.

    get_terminal_size : calculates console height and width.

    Examples
    --------
    `CONSOLE_X_HTML` sets the number of characters for buffer width on the HTML display.
    Truncated characters are replaced by ellipsis.

    >>> from riptable.Utils.display_options import DisplayOptions
    >>> ds = rt.Dataset({'A':[0,6,9], 'B' : [1.2,3.1,9.6], 'C': [-1.6,2.7,4.6], 'D' : [2.4,6.2,19.2]})
    >>> ds
    #   A      B       C       D
    -   -   ----   -----   -----
    0   0   1.20   -1.60    2.40
    1   6   3.10    2.70    6.20
    2   9   9.60    4.60   19.20
    >>> DisplayOptions.CONSOLE_X_HTML = 25
    >>> ds
    #	A	B	...	D
    0	0	1.20	...	2.40
    1	6	3.10	...	6.20
    2	9	9.60	...	19.20
    """
    # class related options
    _CONFIG_LOADED = False  # default config file was found and loaded
    _AUTO_SAVE = False      # if true, config file will be saved to default path each time an option changes
    _USERNAME = None        # to distinguish between windows/linux environments
    _RESET_OPTIONS = False  # a flag so the next session will replace custom user options with the default options

    # screen/environment
    # Todo alz 20191125 - revisit the implementation, couldn't find usages and doesn't behave with console buffer
    CONSOLE_X_BUFFER = 30   # overall x buffer for console display
    CONSOLE_X_HTML = 340    # default "console width" for html display
    CONSOLE_X = 150         # default console width (also calculated by terminalsize.py) TODO: remove
    CONSOLE_Y = 25          # default console height (also calculated by terminalsize.py) TODO: remove
    HTML_DISPLAY = True     # force html display (TODO: remove)
    X_PADDING = 4           # character buffer for each column in console
    Y_PADDING = 3           # character buffer for each row in console

    # dataset/multiset
    ROW_ALL = False     # force all rows to display
    COL_ALL = False     # force all columns to display
    COL_MIN = 1         # min columns to display
    COL_MAX = 50        # max columns to display
    COL_T = 8           # number of transposed rows to display (which appear as columns)
    HEAD_ROWS = 15      # for dataset head
    TAIL_ROWS = 15      # for dataset tail
    MAX_ROWS = 30       # max rows to display
    NO_STYLES = False   # toggle for colors in the ipython console (sometimes hard to see with light background)
    COLOR_MODE = None   # set a color mode
    #NROWS_TRANSPOSE = 0 # 
    #NCOLS_TRANSPOSE = 0 # if > 0, a specific number of
    #BORDER     = True # add a border beneath header labels
    # toggle so completion results show in alphanumeric key, attribute, then method ordering for Dataset, Multiset,
    # and Struct at any nested level
    CUSTOM_COMPLETION: bool = False

    # formatting for datasets/mutilsets/etc
    MAX_HEADER_WIDTH = 15 # maximum for header strings in dataset/multiset
    MAX_FOOTER_WIDTH = 15 # maximum for footer strings in dataset/multiset
    MAX_STRING_WIDTH = 15 # maximum for ALL strings

    # formatting for floating point and integer
    PRECISION = 2            # number of digits to the right of the decimal
    E_PRECISION = 3          # number of digits to display to the right of the decimal (sci notation)
    E_THRESHOLD = 6          # power of 10 at which the float flips to scientific notation 10**+/-
    
    E_MIN = None             # lower limit before going to scientific notation
    E_MAX = None             # upper limit before going to scientific notation
    P_THRESHOLD = None       # precision threshold for area in between - so small values don't display as zero

    NUMBER_SEPARATOR = False # flag for separating thousands in floats and ints
    NUMBER_SEPARATOR_CHAR = DisplayNumberSeparator.Comma # character for separating , . or _

    # misc
    GB_PREFIX = "*" # prefix for column names to indicate that they are groupby keys

    # TODO: split the json config loader to separate files so that new display formatting
    # can be added more easily for future data types

    # min/max values for each display option
    # TODO: clean up and test DisplayOptions.__setitem__ to make sure these are followed
    _BOUNDS = {
                "HEAD_ROWS" : (4,500),
                "TAIL_ROWS" : (4,500),
                "MAX_ROWS"  : (4,500),
                "COL_MIN" : (1,1000),
                "COL_MAX" : (1,51),
                "COL_T" : (1,200),
                "MAX_HEADER_WIDTH" : (0,100),
                "MAX_FOOTER_WIDTH" : (0,100),
                "MAX_STRING_WIDTH" : (0,100),
                "PRECISION" : (1,14),
                "E_PRECISION" : (.00000000001,6),
                "E_THRESHOLD" : (1,20),
                "X_PADDING" : (1,25),
                "Y_PADDING" : (1,15),
                "CONSOLE_X" : (80, 500),
                "CONSOLE_Y" : (25, 150)
    }

    # test flags for display styling
    # see rt_display
    _PAINT_SIGNS = False
    _PAINT_ZEROS = False
    _PAINT_MIN = False
    _PAINT_MAX = False
    _BAR_GRAPH = False
    _TRANSPOSE = False
    _HEAT_MAP = False

    # test flag for including a footer in dataset/multiset display
    # TODO: more testing when a new Table class is created
    # header/footer/left/right
    _TEST_FOOTERS = False

    # test flag for reducing table printing to one pass over the data
    _TEST_ONE_PASS = False
    
    def __new__(cls):
        if cls._CONFIG_LOADED is False:
            DisplayOptions._get_username()
            was_loaded = DisplayOptions.load_config()
            if was_loaded is False:
                print("No display options found. Creating new display option config file...")
                try:
                    # may fail in windows
                    DisplayOptions.save_config()
                except Exception:
                    pass
            # reset code
            elif was_loaded == -1:
                DisplayOptions.save_config(force_overwrite=True)
        return cls

    def __init__(self):
        if self._CONFIG_LOADED is False:
            DisplayOptions._get_username()
            was_loaded = DisplayOptions.load_config()
            if was_loaded is False:
                print("No display options found. Creating new display option config file...")
                DisplayOptions.save_config()
            # reset code
            elif was_loaded == -1:
                DisplayOptions.save_config(force_overwrite=True)

    # when a new item is replaced or added ---------------------------
    def __setattr__(self, name, value):
        if hasattr(self,name):
            if name[0] == '_':
                setattr(self,name,value)

            else:
                bmin, bmax = self._BOUNDS[name]
                # old, for restricting property values
                #if value > bmax:
                #    pass
                #elif value < bmin:
                #    pass
                #else:
                #    pass

                # reset precision properties
                if name in ['PRECISION', 'E_PRECISION', 'E_THRESHOLD']:
                    self.E_MIN = None
                    self.E_MAX = None
                    self.P_THRESHOLD = None

                setattr(self, name, value)

            if self._AUTO_SAVE:
                self.save_config()
        else:
            raise NameError(f"DisplayOptions has no attribute '{name}'.")

    @staticmethod
    def _get_username():
        if "USERNAME" in os.environ:
            DisplayOptions._USERNAME = os.environ["USERNAME"]
        elif "LOGNAME" in os.environ:
                DisplayOptions._USERNAME = os.environ["LOGNAME"]
        else:
            DisplayOptions._USERNAME = "riptable"

    @staticmethod
    def _get_default_path():
        # maybe store this to a class global?
        app_name = "riptable"
        #app_author = DisplayOptions._USERNAME
        config_dir = user_config_dir(app_name)

        return config_dir

    @staticmethod
    def save_config(path: Optional[str] = None, name: Optional[str] = None, force_overwrite: bool = False) -> bool:
        """
        Save display options at the default config file path if path and name are not supplied.
        Otherwise save display options using path and name.
        If `force_overwrite` is True, then silently overwrite any previous configs at that file path, otherwise prompt
        for user input before overwrite.

        Parameters
        ----------
        path : str, optional
            Path to display config
        name : str, optional
            Name of display config
        force_overwrite : bool
            True to overwrite if file already exists, otherwise prompt user whether to overwrite the file.

        Returns
        -------
        result : bool
            True if config was saved, otherwise False.
        """
        # get default save location, or accept a custom one
        # use this for now
        if path is None:
            path = DisplayOptions._get_default_path()
            if not os.path.exists(path):
                print("Default directory",path,"doesn't exist.")
                print("Creating directory...")
                # make an riptable directory if it doesn't exist
                try:
                    os.makedirs(path)
                except:
                    print("Unable to create directory",path)
                    return False
                else:
                    print("Success.")
            # add condition for generating default path for linux users
        if name is None:
            # fixes bug with autoreload
            if DisplayOptions._USERNAME is None:
                DisplayOptions._get_username()
            name = DisplayOptions._USERNAME+"_display_options.json"
        full_path = path+os.path.sep+name

        if os.path.exists(full_path) is True:
            if DisplayOptions._AUTO_SAVE is False:
                # if auto save is turned off, double check with user to make sure overwrite is okay
                if force_overwrite is False:
                    overwrite_prompt = "File exists at " +full_path +". Overwrite? (y/n) "
                    overwrite = input(overwrite_prompt)
                    if overwrite != 'y' and overwrite != 'Y':
                        return False

        save_dict = {}
        for var in dir(DisplayOptions):
            # don't store functions or python default variables
            if not(callable(getattr(DisplayOptions, var))) and not var.startswith("__"):
                save_dict[var] = DisplayOptions.__getattribute__(DisplayOptions, var)
        try:
            with open(full_path, 'w') as save_file:
                dump(save_dict, save_file)
        except:
            print("Error saving display options to",full_path)
            return False
        else:
            if force_overwrite is False:
                print("Saved display options as",name)
            return True

    @staticmethod
    def load_config(path: Optional[str] = None, name: Optional[str] = None) -> Union[bool, int]:
        """
        Load display config file from the default location if path and name are not supplied.
        Otherwise load display config settings using path and name.
        Return bool if applied correctly, otherwise return -1 if resetting display options.

        Parameters
        ----------
        path : str, optional
            Path to display config file
        name : str, optional
            Name of display config file

        Returns
        -------
        result : bool or int
            True if config was loaded correctly, otherwise False.
            -1 if a new default config will be saved after a reset
        """
        if path is None:
            try:
                # fails on some windows configs
                path = DisplayOptions._get_default_path()
            except Exception:
                return False
        if name is None:
            DisplayOptions._get_username()
            name = DisplayOptions._USERNAME+"_display_options.json"
        full_path = path+os.path.sep+name

        if not os.path.exists(full_path):
            print("No configuration file found at",full_path)
            return False

        try:
            with open(full_path,'r') as config_file:
                data = load(config_file)
        except:
            print("Error loading display options from",full_path)
            return False
        else:
            # a new default config file will be saved after a reset
            reset = data.get("_RESET_OPTIONS",False)
            # Todo alz 20191125 - use tuple rather than returning a value of different types
            # PEP 20 - Explicit is better than implicit and readability counts.
            if reset:
                print("Resetting display options.")
                return -1
            # print("Loaded display options.")
            for var, value in data.items():
                setattr(DisplayOptions,var,value)
            DisplayOptions._CONFIG_LOADED = True
            return True

    @staticmethod
    def reset_config(path: Optional[str] = None, name: Optional[str] = None) -> None:
        """
        Reapply display config file from default location if path and name are not supplied.
        Otherwise override display config settings using path and name.

        Parameters
        ----------
        path : str, optional
            Path to display config file
        name : str, optional
            Name of display config file

        Returns
        -------
        None
        """
        overwrite_prompt = "Are you sure you want to reset your display options? (y/n)"
        overwrite = input(overwrite_prompt)
        if overwrite != 'y' and overwrite != 'Y':
            return
        print("Options marked for reset. Start new session to complete reset.")
        DisplayOptions._RESET_OPTIONS = True
        # Todo alz 20191125 - implement use of path and name parameters
        DisplayOptions.save_config(force_overwrite=True)

    @classmethod
    def e_min(cls) -> int:
        """
        Returns the lower limit integer before displaying in scientific notation.

        Returns
        -------
        e_min: int
            lower limit before going to scientific notation.
        """
        if cls.E_MIN is None:
            cls.E_MIN = 10 ** (-1 * TypeRegister.DisplayOptions.E_THRESHOLD)
        return cls.E_MIN

    @classmethod
    def e_max(cls) -> int:
        """
        Returns the upper limit integer before displaying in scientific notation.

        Returns
        -------
        e_max : int
            Upper limit before going to scientific notation
        """
        if cls.E_MAX is None:
            cls.E_MAX = 10 ** TypeRegister.DisplayOptions.E_THRESHOLD
        return cls.E_MAX

    @classmethod
    def p_threshold(cls) -> float:
        """
        Returns DisplayOption.P_THRESHOLD.
        Defaults to 10 ** (-1 * DisplayOptions.PRECISION) - 1e-5.

        Returns
        -------
        p_threshold : float
             The precision threshold for area in between - so small values don't display as zero
        """
        if cls.P_THRESHOLD is None:
            cls.P_THRESHOLD = 10 ** (-1 * TypeRegister.DisplayOptions.PRECISION) - 1e-5
        return cls.P_THRESHOLD

    @staticmethod
    def no_colors() -> None:
        """
        Turn off all non-default table styles.

        Returns
        -------
        str, optional
        """
        DisplayOptions._PAINT_SIGNS = False
        DisplayOptions._PAINT_ZEROS = False
        DisplayOptions._PAINT_MIN = False
        DisplayOptions._PAINT_MAX = False
        DisplayOptions._BAR_GRAPH = False


TypeRegister.DisplayOptions = DisplayOptions
