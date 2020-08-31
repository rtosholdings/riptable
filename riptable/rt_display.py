from typing import Optional

__all__ = ['DisplayCell', 'DisplayColumn', 'DisplayText',
           'DisplayDetect', 'DisplayString', 'DisplayTable']

import os
import numpy as np
import re # used for new console display, remove when moved
import warnings
try:
    from IPython import get_ipython
except:
    pass
from .Utils.display_options import DisplayOptions
from .Utils.terminalsize import get_terminal_size
from .Utils.rt_display_properties import ItemFormat, DisplayConvert, default_item_formats, get_array_formatter
from .rt_enum import  DisplayDetectModes, DisplayArrayTypes, DisplayLength, DisplayColumnColors, DisplayJustification, DisplayColorMode, DisplayTextDecoration, NumpyCharTypes, ColHeader, INVALID_DICT, TypeRegister, INVALID_SHORT_NAME, INVALID_LONG_NAME, ColumnStyle
from .rt_misc import build_header_tuples, parse_header_tuples
from .rt_datetime import DateTimeBase
from .rt_numpy import arange, hstack, bool_to_fancy, ismember
from .rt_timers import GetTSC


class DisplayAttributes(object):
    MARGIN_COLUMNS = "MarginColumns"
    NUMBER_OF_FOOTER_ROWS = "NumberOfFooterRows"


class DisplayDetect(object):
    # Detects which environment the data is being displayed in.
    # Dataset class flips global mode to DisplayDetectModes.HTML when the first
    # _repr_html_ is called.
    Mode =0
    ForceRepr = False
    ColorMode = DisplayColorMode.Dark

    @staticmethod
    def get_display_mode():
        if (DisplayDetect.Mode ==0):
            try:
                ip = get_ipython()
                configdict = ip.config
                lenconfig = len(configdict)
                # spyder has InteractiveShell
                if (lenconfig > 0 and not configdict.has_key('InteractiveShell')):
                    #notebook or spyder
                    DisplayDetect.Mode =DisplayDetectModes.Jupyter
                else:
                    #ipython
                    DisplayDetect.Mode =DisplayDetectModes.Ipython

                # set a color mode for lightbg, darkbg, or no colors
                if DisplayOptions.COLOR_MODE is not None:
                    DisplayDetect.ColorMode = DisplayOptions.COLOR_MODE
                else:
                    color_detected = ip.colors
                    if color_detected == 'Linux' or color_detected == 'LightBG':
                        DisplayDetect.ColorMode = DisplayColorMode.Light
                    elif color_detected == 'Neutral' or (
                            'PYCHARM_HOSTED' in os.environ and
                            os.environ['PYCHARM_HOSTED'] == '1'):
                        DisplayDetect.ColorMode = DisplayColorMode.Dark
                    else:
                        DisplayDetect.ColorMode = DisplayColorMode.NoColors

            except:
                DisplayDetect.Mode =DisplayDetectModes.Console

        return DisplayDetect.Mode


class DisplayString(object):
    # wrapper for display operations that do not return a dataset or multiset
    # ex. transpose: Dataset._T

    def __init__(self, string):
        self.data = string

    def __repr__(self):
        TypeRegister.Struct._lastrepr =GetTSC()
        return self.data

    def _repr_html_(self):
        TypeRegister.Struct._lastreprhtml =GetTSC()
        if DisplayDetect.Mode == DisplayDetectModes.HTML:
            return self.data
        else:
            return None

    def __str__(self):
        if DisplayDetect.Mode == DisplayDetectModes.Console:
            return self.data
        else:
            if DisplayDetect.Mode == DisplayDetectModes.HTML:
                return self._repr_html_()
            return self.__repr__()


class DisplayText(object):
    '''
    Only uses two colors: green and purple   OR  cyan and blue
    For HTML

    ds = rt.Dataset({'test': rt.arange(10)})
    schema = {'Description': 'This is a structure', 'Steward': 'Nick'}
    ds.apply_schema(schema)
    ds.info()
    '''
    ESC = '\x1b['
    RESET = '\x1b[00m'
    TITLE_DARK = '1;32m'   # green
    TITLE_LIGHT = '1;35m'  # purple
    HEADER_DARK = '1;36m'  # cyan
    HEADER_LIGHT = '1;34m' # blue

    def __init__(self, text):
        '''
        Wrapper for display of possibly formatted text (e.g., Dataset.info()

        :param text:
        '''
        self.data = text

    @staticmethod
    def _as_if_dark():
        return DisplayDetect.ColorMode == DisplayColorMode.Dark and\
            DisplayDetect.Mode != DisplayDetectModes.Jupyter

    @staticmethod
    def _title_color():
        if DisplayText._as_if_dark():
            return DisplayText.TITLE_DARK
        else:
            return DisplayText.TITLE_LIGHT

    @staticmethod
    def _header_color():
        if DisplayText._as_if_dark():
            return DisplayText.HEADER_DARK
        else:
            return DisplayText.HEADER_LIGHT

    @staticmethod
    def _format(txt, fmt):
        return DisplayText.ESC + fmt + txt + DisplayText.RESET

    @staticmethod
    def title_format(txt):
        return DisplayText._format(txt, DisplayText._title_color())

    @staticmethod
    def header_format(txt):
        return DisplayText._format(txt, DisplayText._header_color())

    def __str__(self):
        return self.data

    def __repr__(self):
        return self.data

    def _repr_html_(self):
        # creates a dependency on ansi2html
        from ansi2html import Ansi2HTMLConverter
        #preamble='<html>\n<head>\n<meta http-equiv="Content-Type" content="text/html; charset=utf-8">\n<title></title>\n<style type="text/css">\n.ansi2html-content { display: inline; white-space: pre-wrap; word-wrap: break-word; }\n.body_foreground { color: #AAAAAA; }\n.body_background { background-color: #000000; }\n.body_foreground > .bold,.bold > .body_foreground, body.body_foreground > pre > .bold { color: #FFFFFF; font-weight: normal; }\n.inv_foreground { color: #000000; }\n.inv_background { background-color: #AAAAAA; }\n</style>\n</head>\n<body class="body_foreground body_background" style="font-size: normal;" >\n<pre class="ansi2html-content">\n'
        #postamble = '</pre>\n</body>\n\n</html>\n'
        #return preamble + self.data + postamble
        return Ansi2HTMLConverter().convert(self.data)


class DisplayTable(object):
    TestFooter = False

    DebugMode = False

    INVALID_DATA = np.nan
    options = DisplayOptions()
    console_x_offset = 3
    FORCE_REPR = False

    @classmethod
    def console_detect_settings(cls):
        '''
        For debugging console display.
        '''
        display_mode = DisplayDetect.Mode
        display_color_mode = DisplayDetect.ColorMode
        detected_x, detected_y = get_terminal_size()
        default_x = cls.options.CONSOLE_X
        default_y = cls.options.CONSOLE_Y

        settings_string = ["\n"]
        settings_string.append("      display mode:"+str(display_mode))
        settings_string.append("        color mode:"+str(display_color_mode))
        settings_string.append("detected console x:"+str(detected_x))
        settings_string.append("detected console y:"+str(detected_y))
        settings_string.append(" default console x:"+str(default_x))
        settings_string.append(" default console y:"+str(default_y))

        print("\n".join(settings_string))

    def __init__(self, attribs: Optional[dict] = None):
        if attribs is None:
            attribs = dict()
        self._console_x = self.options.CONSOLE_X
        self._console_y = self.options.CONSOLE_Y
        if DisplayTable.FORCE_REPR is False:
            self._console_x, self._console_y = get_terminal_size()
            if self._console_x is None:
                if DisplayTable.DebugMode: print("could not detect console size. using defaults.")
                DisplayTable.FORCE_REPR = True
                self._console_x = self.options.CONSOLE_X
                self._console_y = self.options.CONSOLE_Y

        # certain machines will not fail to detect the console width, but set it to zero
        # default to the minimum console x bound
        if self._console_x < self.options._BOUNDS['CONSOLE_X'][0]:
            self._console_x = self.options._BOUNDS['CONSOLE_X'][0]

        self._display_mode = DisplayDetect.Mode
        # check for html
        if self._display_mode == DisplayDetectModes.HTML or self._display_mode == DisplayDetectModes.Jupyter:
            self._console_x = self.options.CONSOLE_X_HTML

        # dict for any display attributes passed by the initializer
        self._attribs = attribs
    
    #---------------------------------------------------------------------------
    def build_result_table(
        self, header_tups, main_data, nrows:int,
       footer_tups=None, keys:dict=None, sortkeys=None, 
       from_str=False, sorted_row_idx=None, transpose_on=False, 
       row_numbers=None, right_cols=None, 
       badrows=None,
       badcols=None,
       styles=None,
       callback=None):

        """
        Step 1: save all parameters into self namespace, as build_result_table is broken down into several functions.
        Step 2: if set_view has been called, only display the specified columns. if sort_values has been called, move those columns to the front.
        Step 3: build a row mask. if the table is too large to display, pull the first and last rows for display. if a sorted index is present, apply it.
        Step 4: measure the table. 
        groupby key columns will always be included. fit as many columns as possible into the console. if the display is for html, defaults have been set to a hard-coded console width. other console width is detected upon each display. if there are too many columns to display, a column break will be set.
        Step 5: build the table.
        the result table is broken down into three parts: headers, left side, and main table.
        the headers are column names combined with left headers, or numbers if the table is transposed.
        the left side is row numbers, row labels, or groupby keys.
        the main table is first and last columns that would fit in the display.
        use the DisplayColumn class to organize the data for future styling. If the table is abbreviated, include a row break in each column.
        Step 6: style the table. html_on will let DisplayColumn and DisplayCell know how to “paint” the individual cells.
        Step 7: if the header has multiple lines and/or needs to be transposed, fix it up now.
        Step 8: transpose the table for final display. we build the table by column, but it needs to be displayed by row. if the table should be transposed, don’t rotate it - clean up the headers.
        Step 9: pass the table string to our console or html routine for final output.

        **TODO: reduce the measuring and building to one pass over the data. currently rendering time is not an issue. ~15ms
        """
        return self.build_result_table_new(
        header_tups, 
        main_data, 
        nrows, 

        keys=keys, 
        sortkeys=sortkeys,
        from_str=from_str,
        sorted_row_idx=sorted_row_idx,
        transpose_on=transpose_on,

        row_numbers=row_numbers,
        right_cols=right_cols,
        footer_tups=footer_tups,
        badrows=badrows,
        badcols=badcols,
        styles=styles,
        callback=callback)

    # -------------------------------------------------------------------------------------
    def build_row_mask(self, head, tail, total):
        # r_mask is the reqd data indices in one array (wether or not a row/column gets split)
        # if necessary, break will be inserted in the final DisplayColumn

        # if number requested >= size of original data, all will be shown
        if (head+tail >= total) or self.options.ROW_ALL is True:
            r_mask = arange(total, dtype=np.int64)
            rowbreak = None

        else:
            #split mask [0 to head]  row_break   [end-tail to end]
            h = arange(head, dtype=np.int64)
            t = arange(total-tail, total, dtype=np.int64)
            r_mask = hstack((h,t))
            # row break is the row number at which to insert a break
            rowbreak = head

        # save the row mask, will check for sort later
        return r_mask, rowbreak


    #---------------------------------------------------------------------------
    def get_sort_col_idx(self, col_names):
        # returns a list of indicies of columns to sort by
        # used to move sort key columns to the front, or mask the columns from a col_set_view() call
        sorted_col_mask = []
        for name in col_names:
            # extract the index from np where. the header names array has unique values.
            current_idx_tup = np.where(self._header_names == name)
            current_idx = current_idx_tup[0]
            if len(current_idx) < 1:
                self._missing_sort_cols = True
                # print("Sort column",name,"missing from sorted dataset view. Use ds.col_set_view('*') to reset column view or ds.unsort() to remove sort columns.")
            else:
                current_idx = current_idx[0]
                sorted_col_mask.append(current_idx)
        return sorted_col_mask

    #---------------------------------------------------------------------------
    def get_bad_color(self):
        '''
        put in the bad_col dictionary
        '''
        return ColumnStyle(color=DisplayColumnColors.GrayItalic)

    #---------------------------------------------------------------------------
    def build_result_table_new(self,
        header_tups, 
        main_data, 
        nrows:int, 

        keys:dict=None, 
        sortkeys=None,
        from_str=False,
        sorted_row_idx=None,
        transpose_on=False,

        row_numbers=None,
        right_cols=None,
        footer_tups=None,
        badcols=None,
        badrows=None,
        styles=None,
        callback=None):
        '''

        callback: func, default None
           callback to signature
        '''

        #---------------------------
        def trim_data(trimslice, main=True):
            # as left, right, center are built, parts of headers need to be trimmed off
            # main data, bottom headers need to be trimmed for left and right side
            if main:
                self._main_data = self._main_data[trimslice]
            self._header_tups[-1] = self._header_tups[-1][trimslice]
            self._header_names = self._header_names[trimslice]
            #if self._footer_tups is not None:
            #    for i, frow in enumerate(self._footer_tups):
            #        self._footer_tups[i] = frow[trimslice]

        #---------------------------
        # MAIN TABLE INFO
        self._header_tups = header_tups     # list of lists of header tuples (see ColHeader in rt_enum)
        self._main_data = main_data         # list of all column arrays
        self._nrows = nrows                 # number of rows
        self._ncols = len(self._main_data)  # number of columns

        self._gbkeys = keys                 # dictionary of groupby keys:data (left hand columns)
                                            # *** changing to list of names, only need to know number of label cols
        self._right_cols = right_cols       # dictionary of right hand columns -> data
                                            # *** changing to list of names, only need to know number of right cols

        self._gb_prefix = TypeRegister.DisplayOptions.GB_PREFIX # add star before groupby names
        self._footer_name = ""
        self._footer_tups = footer_tups
        if self._footer_tups is not None:
            self._footer_name = self._footer_tups[0][0].col_name

        # FLAGS
        self._transpose_on = transpose_on   # boolean: will the table transposed
        self._from_str = from_str           # boolean: should simple text be forced (no HTML)

        # SORTS / MASKS
        self._col_sortlist = sortkeys         # list of names of columns which are sorted (for rearanging and styling)
        self._sorted_row_idx = sorted_row_idx # ndarray of sort indices
        self._missing_sort_cols = False       # sort column names were provided, but not found in the data

        # COLUMN STYLES
        self._row_numbers = row_numbers
        number_style = ColumnStyle(color=DisplayColumnColors.Rownum, align=DisplayJustification.Right)
        label_style = ColumnStyle(color=DisplayColumnColors.Rownum)
        gb_style = ColumnStyle(color=DisplayColumnColors.Groupby)
        sort_style = ColumnStyle(color=DisplayColumnColors.Sort)
        right_style = ColumnStyle(color=DisplayColumnColors.Groupby)
        red_style = ColumnStyle(color=DisplayColumnColors.Red)
        gray_style = ColumnStyle(color=DisplayColumnColors.GrayItalic)


        # initialize width lists for console display
        self._left_widths = []
        self._main_first_widths = []
        self._main_last_widths = []
        self._right_widths = []
        self._all_widths = []

        # used for users requesting all columns to be displayed in console
        self._column_sets = None
        self._set_widths = None

        # used to measure how many columns will fit
        self._total_width = 0

        # break flags, sorting masks
        self._has_row_break = 0
        self._has_col_break = 0
        self._row_break = None
        self._col_break = None
        self._c_mask = None
        self._r_mask = None

        self._has_rownums = False

        # extract names from last row in header tuples for a simple list of column names
        self._header_names = np.array([h.col_name for h in self._header_tups[-1]])
        self._num_header_lines = len(self._header_tups)


        # detect HTML display for width calculation and future styling
        self._html_on = False
        if self._display_mode==DisplayDetectModes.HTML:
            if from_str:
                # max console display width in browser is smaller for simple table
                self._console_x = 120
            else:
                self._html_on = True
        else:
            # option to remove colors from ipython display (hard to read in certain console windows)
            # HTML for jupyter lab/notebook will always be styled, unless the table is called with a print statement
            if self.options.NO_STYLES:
                from_str = True

        # -------------CALCULATE NUMBER OF ROWS TO BE DISPLAYED--------------------------------
        # -------------------------------------------------------------------------------------
        # head and tail functions will be taken into consideration here
        # transposed tables will change this number significantly
        totalrows = 0
    
        # transposed
        if self._transpose_on:
            if DisplayTable.DebugMode: print("*table transposed")
            # use the display option for number of columns to show in a transposed table
            # if there aren't enough rows, display all
            self._num_transpose_rows = min(self.options.COL_T,nrows)
            totalrows = self._num_transpose_rows
            head = self._num_transpose_rows
            tail = 0

        else:
            if DisplayTable.DebugMode: print("*table untransposed")
            totalrows=nrows
            # force all rows to be shown
            if self.options.ROW_ALL is True:
                if DisplayTable.DebugMode is True: print("*forcing all rows")
                head = nrows
                tail = 0
            # enforce number of rows to show based on display option
            else:
                head = self.options.HEAD_ROWS
                tail = self.options.TAIL_ROWS
            # if the table is empty, exit routine here
            if totalrows is None:
                #return "Table is empty (has no rows)."
                totalrows = 0

        # -----------BUILD A ROW MASK ---------------------------------------------------------
        # -------------------------------------------------------------------------------------
        self._r_mask, self._row_break = self.build_row_mask(head, tail, totalrows)
        # possibly apply the mask (single arange or stacked arange) to the indirect sort
        if sorted_row_idx is not None:
            self._r_mask = self._sorted_row_idx[self._r_mask]
        self._has_row_break = self._row_break is not None

        # ---------------BUILD THE LEFT TABLE--------------------------------------------------
        # -------------------------------------------------------------------------------------
        # the left frame will ALWAYS be displayed
        self._left_table_data = []      # list of arrays
        left_header_names = []
        # transposed
        if self._transpose_on:
            # left table will be all groupby key names + all column names in a single column
            if self._gbkeys is not None:
                nKeys = len(self._gbkeys)
                header_column = self._header_names
                trimslice = slice(nKeys,None,None)
                trim_data(trimslice)
            else:
                # trim off 1 for row numbers, or leave as-is to prepare for row numbers callback
                nKeys = 1 if self._row_numbers is None else 0
                trimslice = slice(nKeys, None, None)
                trim_data(trimslice, main=False)
                header_column = self._header_names

            self._left_table_data.append(header_column)
            left_header_names.append("Fields:")
            
        # untransposed
        else:
            # groupby keys take priority
            if self._gbkeys is not None:
                nKeys = len(self._gbkeys)
                for i in range(nKeys):
                    # only last header line is handled here
                    left_header_names.append(self._gb_prefix+str( self._header_tups[-1][i][0] ))
                    self._left_table_data.append( self._main_data[i] )
                # after adding, trim off left columns
                trimslice = slice(nKeys,None,None)
                trim_data(trimslice)
            else:
                self._has_rownums=True
                # these will be used on the left and/or sent to row_label/row_number callback
                row_numbers = self._r_mask

                # regular row numbers
                if self._row_numbers is None:
                    left_header_names.append(self._header_tups[-1][0][0])
                    trimslice = slice(1,None,None)
                    trim_data(trimslice, main=False)
                    self._left_table_data.append(row_numbers)

                # custom row numbers
                else:
                    # the entire left side of the table will be redefined by _row_numbers callback
                    # returns a single column, same size as row index sent in
                    func = self._row_numbers
                    name, numbers, number_style = func(row_numbers, number_style)
                    left_header_names.append(name)
                    self._left_table_data.append(numbers)

        # display-generated row numbers and/or class-defined row_numbers function
        if self._has_rownums or self._transpose_on:
            color = DisplayColumnColors.Rownum
            masked = True
            style = number_style

        # untransposed groupby columns
        else:
            color = None
            masked = False
            style = gb_style

        left_footers = None
        # TODO: move this to generic trim_data() util
        if self._footer_tups is not None:
            left_footers = [ frow[:len(left_header_names)] for frow in self._footer_tups ]

        # build DisplayColumns and measure left hand side
        # _left_table_columns - a list of DisplayColumn objects
        # _left_widths - a list of ints
        self._left_table_columns, self._left_widths = self.add_required_columns(left_header_names,
                                                                                self._left_table_data,
                                                                                left_footers,
                                                                                gbkeys=self._gbkeys,
                                                                                color=color,
                                                                                masked=masked,
                                                                                transpose=self._transpose_on,
                                                                                style=style)

        self._total_width += sum(self._left_widths)

        # ---------------BUILD THE RIGHT TABLE-------------------------------------------------
        # -------------------------------------------------------------------------------------
        # if it exists, right frame will ALWAYS be displayed
        self._right_table_columns = []
        self._right_table_data = []      # raw data
        right_header_names = []
        if self._right_cols is not None:
            right_footers = None
            nKeys = len(self._right_cols)
            right_header_names = [ t[0] for t in self._header_tups[-1][-nKeys:] ]
            self._right_table_data = self._main_data[-nKeys:]
            # trim off right columns from main data
            # also trim footers
            trimslice = slice(None,-nKeys,None)
            trim_data(trimslice)
            if self._footer_tups is not None:
                right_footers = [ frow[-nKeys:] for frow in self._footer_tups ]

            # build DisplayColumns and measure right hand side
            # _right_table_columns - a list of DisplayColumn objects
            # _right_widths - a list of ints
            self._right_table_columns, self._right_widths = self.add_required_columns(right_header_names,
                                                                                      self._right_table_data,
                                                                                      right_footers)

            self._total_width += sum(self._right_widths)

        # ---------------BUILD THE MAIN TABLE--------------------------------------------------
        if self._transpose_on:
            self._main_table_columns = self.build_transposed_columns(self._main_data)
        else:
            # all columns in console
            if self.options.COL_ALL and ((self._html_on is False) or self._from_str) and self._num_header_lines == 1:
                self._column_sets, self._set_widths =  self.all_columns_console(
                    self._console_x, self._total_width, self._header_names, self._main_data)
            else:
                if self._footer_tups is not None:
                    # start at beginning of main columns
                    left_offset = len(self._left_table_columns)
                    frows = [ f[left_offset:] for f in self._footer_tups ]
                    footer_arr = []
                    for i in range(len(frows[0])):
                        # list of value for each line, for each column
                        # e.g. if column had sum 6, mean 2.00, its list would be ['6','2.00']
                        footer_arr.append( [f[i][0] for f in frows])
                else:
                    footer_arr = None

                self._main_table_columns, self._main_first_widths, self._main_last_widths = self.fit_max_columns(
                    self._header_names, self._main_data, self._total_width, self._console_x, footer_arr)

        # -------------------------STYLE THE TABLE--------------------------------------------
        # ------------------------------------------------------------------------------------

        # recolor badrows
        if badrows is not None:
            # if rmask is set, it is a lookup to the actual row being used
            # fixup badrows

            rmask=self._r_mask
            rbreak=self._row_break

            # use the head + tail to calculate how many relevant rows
            ldata = head + tail

            # build a new badrows with the correct line 
            newbadrows={}

            for k,v in badrows.items():
                if rmask is not None:
                    # look find our number in the mask
                    #print("rmask", rmask, "k",k)
                    bmask= rmask==k
                    if isinstance(bmask, np.ndarray):
                        btf = bool_to_fancy(bmask)
                        if len(btf) > 0:
                            # we found the number, now see if its before or after break
                            loc = btf[0]
                            if rbreak is not None:
                                if loc < rbreak:
                                    newbadrows[loc]= v
                                else:
                                    newbadrows[loc + 1]= v
                            else:
                                newbadrows[loc] =v
                elif k < ldata:
                    # as long as there is no mask, just check to see if we are in range
                    # if rmask does not exist, then neither does rbreak
                    newbadrows[k]= v

            if len(newbadrows) > 0:
                badrows = newbadrows
            else:
                badrows = None

        # groupby
        if self._gbkeys is not None and self._transpose_on is False:
            for c in self._left_table_columns:
                c.paint_column(gb_style, badrows=badrows)

        # right margin
        if self._right_cols is not None and self._transpose_on is False:
            for c in self._right_table_columns:
                c.paint_column(right_style, badrows=badrows)

        # sort
        if self._col_sortlist is not None:
            if self._column_sets is None:
                if not self._missing_sort_cols:
                    for i, c in enumerate(self._col_sortlist):
                        self._main_table_columns[i].paint_column(sort_style, badrows=badrows)
            else:
                for i, c in enumerate(self._col_sortlist):
                    self._column_sets[0][i].paint_column(sort_style, badrows=badrows)

        # custom left columns
        if self._row_numbers is not None:
            self._left_table_columns[-1].style_column(number_style)

        # general purpose styles
        # move to main build / measure column routine
        if styles is not None:
            # put in a callback for now, maybe pass the headers being displayed?
            # styles = styles()
            for i, col in enumerate(self._main_table_columns):
                s = styles.get(col.header,None)
                if s is not None:
                    self._main_table_columns[i].style_column(s)

        # color entire column
        if badcols is not None or badrows is not None:
            for i, col in enumerate(self._main_table_columns):
                if badcols is not None:
                    color_style = badcols.get(col.header,None)
                else:
                    color_style=None
                if color_style is not None or badrows is not None:
                    self._main_table_columns[i].style_column(color_style, badrows=badrows)

        # Attribute-based styling
        # -----------------------
        # Color margin columns and footer rows
        if len(self._attribs) > 0 and self._column_sets is None:
            main_header_names = [c.header for c in self._main_table_columns]
            num_footer_rows = self._attribs.get(
                DisplayAttributes.NUMBER_OF_FOOTER_ROWS, 0)
            for i, name in enumerate(main_header_names):
                is_margin_column = name in self._attribs.get(
                    DisplayAttributes.MARGIN_COLUMNS, [])
                if is_margin_column:
                    self._main_table_columns[i].paint_column(DisplayColumnColors.Groupby)
                if num_footer_rows:
                    self._main_table_columns[i].paint_column(DisplayColumnColors.Groupby,
                                                             slice(-num_footer_rows, None, None))

            # Right justify row label(s) of footer rows
            for i, name in enumerate(left_header_names):
                if num_footer_rows:
                    self._left_table_columns[i].align_column(DisplayJustification.Right,
                                                             slice(-num_footer_rows, None, None))

        
        # -----------------------FIX / TRANSLATE TABLE HEADERS---------------------------------
        # -------------------------------------------------------------------------------------
        if self._num_header_lines > 1 and self._transpose_on is False and self._col_break is not None:
            self.fix_multiline_headers()
            final_footers = self.fix_multiline_footers(plain=from_str, badcols=badcols, badrows=badrows)

        if self._column_sets is None:
            self._all_widths = self._left_widths + self._main_first_widths + self._main_last_widths + self._right_widths

        # all columns requested in console
        else:
            # left / right columns will always be included, so need to prepend/append their widths to the final widths
            for idx, main_widths in enumerate(self._set_widths):
                self._set_widths[idx] = self._left_widths + main_widths + self._right_widths

        final_headers, final_footers = self.build_final_ends(plain=from_str, badcols=badcols, badrows=badrows)

        # -------------------------FINAL STYLING----------------------------------------------
        # ------------------------------------------------------------------------------------
        # this is a good place to style for operations that span the whole table
        if self._num_header_lines > 1:
            if self._column_sets is None:
                for i, cell in enumerate(self._header_tups[-1]):
                    color = DisplayColumnColors.Multiset_col_a + (cell.color_group % 2)
                    self._main_table_columns[i].paint_column(color)
            else:
                for set_index, set in enumerate(self._header_sets):
                    for i, cell in enumerate(set[-1]):
                        color = DisplayColumnColors.Multiset_col_a + (cell.color_group % 2)
                        self._column_sets[set_index][i].paint_column(color)
        
        # -------------SUB FUNCTION WITH CALLBACK FOR CELL STYLING----------------------------
        # ------------------------------------------------------------------------------------
        def style_cells(listcols, stylefunc, rows=True, callback=None, location=None):
            # listcols is a list of DisplayColumn objects
            # stylefunc is a string of style function name: plain_string_list or styled_string_list
            # turns columns into formatted strings, if rows is true, each row in its own list
            # returns the formatted strings

            # give user a chance to modify styling
            # if the user returns a list of strings, we do not bother to style ourselves
            if callback:
                # in this callback, the style will indicate plain or not
                # the location is either left, right, or main
                # if rows is False, the data is rotated 90 (transposed)
                # if html is True then expecting html styling vs console styling
                result = callback(listcols, style=stylefunc, location=location, rows=rows, html=self._html_on)

                if isinstance(result, list):
                    # the user can take over and return a list of string with html or console color styling
                    return result

            table_strings = [ getattr(c, stylefunc)() for c in listcols ]
            if rows:
                table_strings = [ list(row) for row in zip(*table_strings) ]

            # return a list of string with html or console color styling
            return table_strings

        # -------------------------BUILD AND ROTATE THE TABLE---------------------------------
        # ------------------------------------------------------------------------------------
        # plain
        if from_str:
            stylefunc = 'plain_string_list'
        # ipython / html
        else:
            stylefunc = 'styled_string_list'

        # call style function to build final strings from stored styles
        # left / right will always be rotated
        right_table_strings = []

        # allow callback for labels and margins
        left_table_strings = style_cells(self._left_table_columns, stylefunc, callback=callback, location='left')
        if self._right_cols is not None:
            right_table_strings = style_cells(self._right_table_columns, stylefunc, callback=callback, location='right')

        # handle single, multiple sets the same way
        if self._column_sets is None:
            final_column_sets = [self._main_table_columns]
        else:
            final_column_sets = self._column_sets
        
        # call style function to build final strings from stored styles
        main_table_strings = []
        as_rows = not self._transpose_on
        for col_set in final_column_sets:
            col_set = style_cells(col_set, stylefunc, as_rows,  callback=callback, location='main')
            main_table_strings.append(col_set)
                
        # single table
        if self._column_sets is None:
            main_table_strings = main_table_strings[0]

            if self._footer_tups is None:
                final_footers = []
            if len(final_footers) == 0:
                final_footers = None

            if self._html_on:
                result_table = DisplayHtmlTable(final_headers, left_table_strings, main_table_strings, right_columns=right_table_strings, footers=final_footers)
            else:
                result_table = DisplayConsoleTable(self._all_widths, final_headers, left_table_strings, main_table_strings, right_table_strings, final_footers)

            return result_table.build_table()

        # print a table for each row
        else:
            result_tables = []
            for idx, main_strings in enumerate(main_table_strings):
                result_tables.append( DisplayConsoleTable(self._set_widths[idx], final_headers[idx], left_table_strings, main_strings, right_table_strings, None).build_table() )
            return "\n\n".join(result_tables)

    #-----------------------------------------------------------------------
    def fix_repeated_keys(self, columns, repeat_string='.'):
        '''
        Display a different string when the first column of a multikey groupby is repeated.
        TODO: add support for the same behavior with repeated keys in multiple columns.
        '''

        column_arrays = [c.data for c in columns]

        for idx, keylist in enumerate(column_arrays):
            if idx == 0:
                pkey = column_arrays[0]
                lenk = len(pkey)
                if lenk > 1:
                    lastkey = str(pkey[0])  # string inside of DisplayCell object
                    for i in range(1, lenk):
                        item1 = str(pkey[i])
                        if item1 == lastkey:
                            pkey[i].string=repeat_string
                        else:
                            lastkey=item1

    # ------------------------------------------------------------------------------------
    def build_final_headers_html(self, plain=False):
        '''
        Translates the tables header tuples into HTML tags.
        Note: this routine is very similar to build_final_headers_console.
        Keeping them separate for readability.
        '''
        final_headers = []
        span = len(self._left_table_columns)
        pad_string = "<td colspan='"+str(span)+"' class='lg'></td>"
        for i, line in enumerate(self._header_tups):
            styled_line = []
            # add padding to every row except the last
            if i != len(self._header_tups)-1:
                styled_line.append(pad_string)
            for j, cell in enumerate(line):
                # color correction for multiline
                color = DisplayColumnColors.Multiset_head_a + (cell.color_group % 2)
                new_cell = DisplayCell(cell.col_name, color=color, html=self._html_on, colspan=cell.cell_span)
                new_cell.paint_cell()

                # match the alignment to main table for the last row
                align = DisplayJustification.Center
                if i == len(self._header_tups)-1:
                    pass
                    # changed default - multiline header cells will always be centered
                    #align = self._main_table_columns[j]._align
                new_cell.prefix += " "+DisplayColumn.align_html(align)

                # BUG: justification doesn't stick
                # build styled string
                new_cell = new_cell.display()
                styled_line.append(new_cell)
            final_headers.append(styled_line)

        # add left table headers to last row
        for c in reversed(self._left_table_columns):
            final_headers[-1].insert(0,c.build_header())

        return final_headers

    # ------------------------------------------------------------------------------------
    def build_final_headers_console(self, plain=False):
        '''
        **specifically for multi-line
        Translates the tables header tuples into console strings with spaces for padding
        Note: this routine is very similar to build_final_headers_html.
        Keeping them separate for readability.
        '''
        
        final_headers = []
        span = len(self._left_table_columns)
        pad_cell = ColHeader("",1,0)
        column_margin = 3

        for i, line in enumerate(self._header_tups[:-1]):
            styled_line = []
            width_index = 0
            # add padding to every row except the last
            # number of pad cells needs to be the number of left hand columns
            if i != len(self._header_tups)-1:
                for j in range(span):
                    self._header_tups[i].insert(0,pad_cell)

            for j,cell in enumerate(line):
                # fix multiline cell colors
                color = DisplayColumnColors.Multiset_head_a + (cell.color_group % 2)
                new_cell = DisplayCell(cell.col_name, color=color, html=self._html_on)

                # get the width of bottom cells in same group to fix alignment
                combined_width = sum(self._all_widths[width_index:width_index+cell.cell_span])
                margin_width = ((cell.cell_span-1) * column_margin)
                combined_width += margin_width
                new_cell.string = DisplayColumn.align_console_string(new_cell.string, combined_width, align=DisplayJustification.Center)
                width_index += cell.cell_span

                # apply final styling
                new_cell.paint_cell()
                new_cell = new_cell.display(plain=plain)
                styled_line.append(new_cell)
            final_headers.append(styled_line)

        
        # use DisplayColumns to style bottom row
        # bug in alignment if sent through the same loop as other headers
        bottom_headers = []
        bottom_colors = [DisplayColumnColors.Multiset_head_a + (cell.color_group % 2) for cell in self._header_tups[-1]]
        for c in self._left_table_columns:
            bottom_headers.append(c.build_header(plain=plain))
        for i,c in enumerate(self._main_table_columns):
            bottom_headers.append(c.build_header(bottom_colors[i], plain=plain, align=DisplayJustification.Center))
        final_headers.append(bottom_headers)

        return final_headers

    # ------------------------------------------------------------------------------------
    def build_final_ends(self, plain=False, badcols=None, badrows=None):
        '''
        '''

        final_headers = []
        final_footers = []
        gray_style = DisplayColumnColors.GrayItalic

        # transposed headers
        if self._transpose_on:
            transposed_headers = []
            for idx, width in enumerate(self._all_widths[len(self._left_table_columns):]):
                new_cell = DisplayCell(str(idx), color=DisplayColumnColors.Rownum, html=self._html_on, colspan=1)
                new_cell.paint_cell()
                if self._html_on: # and from_str is False
                    pass
                else:
                    new_cell.string = DisplayColumn.align_console_string(new_cell.string, width, align=DisplayJustification.Right)
                transposed_headers.append(new_cell.display(plain=plain))
            
            for c in reversed(self._left_table_columns):
                if self._from_str:
                    transposed_headers.insert(0, c.build_header(plain=True))
                else:
                    transposed_headers.insert(0, c.build_header(DisplayColumnColors.Rownum))
            final_headers.append(transposed_headers)

        # untransposed headers
        else:
            # fix multiline
            if self._num_header_lines > 1:
                if self._html_on: #and from_str is False:
                    final_headers = self.build_final_headers_html(plain=plain)
                else:
                    final_headers = self.build_final_headers_console(plain=plain)

                final_footers = self.fix_multiline_footers(plain=plain)

            # default to the headers constructed in DisplayColumns
            else:
                if self._column_sets is None:
                    bottom_headers = []
                    bottom_footers = []
                    for c in self._left_table_columns + self._main_table_columns + self._right_table_columns:
                        # NOTE: you can choose a different final color with the final_color keyword here
                        # see rt_enum.DisplayColumnColors
                        bottom_headers.append(c.build_header(plain=plain))
                        if badcols is not None and badcols.get(c.header, None) is not None:
                            bottom_footers.append(c.build_footer(final_color = gray_style, plain=plain))
                        else:
                            # footers for each column will be in a list
                            bottom_footers.append(c.build_footer(plain=plain))

                    final_headers.append(bottom_headers)
                    final_footers = [ list(frow) for frow in zip(*bottom_footers) ]

                # all columns requested in console
                # TODO: colapse these into one function
                else:
                    for set in self._column_sets:
                        current_row = []
                        final_headers.append([])
                        for c in self._left_table_columns:
                            current_row.append(c.build_header(plain=plain))
                        for c in set:
                            current_row.append(c.build_header(plain=plain))
                        for c in self._right_table_columns:
                            current_row.append(c.build_header(plain=plain))
                        final_headers[-1].append(current_row)

        return final_headers, final_footers

    def footers_to_string(self, footer_row):
        '''
        Takes row of footer tuples and turns into string list.
        For adding/styling multiline footers.
        '''
        pass

    # ------------------------------------------------------------------------------------
    def fix_multiline_headers(self):
        '''
        Fixes multi-line headers if a column break was present.
        cell_spans in ColHeader might need to be changed.
        Need use cases for more than two lines, but the same loop should work.
        '''

        top_header_row = self._header_tups[0]
        break_tup = ColHeader("",1,0)

        # left side
        col_idx = 0
        num_left_cols = len(self._main_first_widths)
        for i, top_header in enumerate(top_header_row):
            current_span = top_header.cell_span
            # keep walking through
            if (col_idx + current_span) < num_left_cols-1:
                col_idx += current_span
            else:
                new_left_head = top_header_row[:i]

                # last span fit
                if (col_idx + current_span) == num_left_cols-1:
                    #print("last left tuple fit")
                    last_tup = top_header
                # trim last span
                else:
                    #print("fixing left span")
                    new_span = num_left_cols - col_idx - 1
                    last_tup = ColHeader(top_header.col_name, new_span, top_header.color_group)

                new_left_head.append(last_tup)
                new_left_head.append(break_tup)
                break

        # right side
        col_idx = 0
        num_right_cols = len(self._main_last_widths)
        for i, top_header in enumerate(reversed(top_header_row)):
            current_span = top_header.cell_span
            # keep walking through
            if (col_idx + current_span) < num_right_cols:
                col_idx += current_span
            else:
                new_right_head = []
                if i != 0:
                    new_right_head = top_header_row[-i:]
                # last span needs to get changed, build a new tuple
                if (col_idx + current_span) > num_right_cols:
                    #print("fixing right span")
                    new_span = num_right_cols - col_idx
                    last_tup = ColHeader(top_header.col_name, new_span, top_header.color_group)
                # last span fits
                else:
                    #print("last right tuple fit")
                    last_tup = top_header

                new_right_head.insert(0,last_tup)
                break
        
        self._header_tups[0] = new_left_head + new_right_head

        # final row (no span changes)
        bottom_left = self._header_tups[-1][:num_left_cols-1]
        bottom_left.append(break_tup)
        bottom_right = self._header_tups[-1][-num_right_cols:]
        self._header_tups[-1] = bottom_left + bottom_right

    # -------------------------------------------------------------------------------------
    def fix_multiline_footers(self, plain=False, badcols=None, badrows=None):
        '''
        '''
        final_footers = []
        gray_style = DisplayColumnColors.GrayItalic

        # transposed
        if self._transpose_on:
            raise NotImplementedError

        # untransposed
        else:
            if self._column_sets is None:
                bottom_footers = []
                for c in self._left_table_columns:
                    bottom_footers.append(c.build_footer(plain=plain))
                for c in self._main_table_columns:
                    if badcols is not None and badcols.get(c.header, None) is not None:
                        bottom_footers.append(c.build_footer(final_color = gray_style , plain=plain))
                    else:
                        bottom_footers.append(c.build_footer(plain=plain))
                final_footers.append(bottom_footers)
            else:
                raise NotImplementedError

        return final_footers

    # -------------------------------------------------------------------------------------
    def build_column(self, header, column, masked=False, footer=None, style=None):
        '''
        All DisplayColumns built for final display will funnel through this function.
        Any row breaks will be added here if necessary.
        '''

        # ask the data how it would like to be displayed
        display_format, func = get_array_formatter(column)

        # merge custom styles or callback styles with default array styles
        if style is not None:
            if style.width is not None:
                display_format.maxwidth = style.width

        # format the footer with the same function as the column data
        # UPDATE: footers should always be string now, might be different type than column above
        if footer is not None:
            if isinstance(footer, list):
                pass
            elif not isinstance(footer, str):
                footer = func(footer, display_format)
                #print('footer was not none')
            
        # transposed data will have the same alignment
        if self._transpose_on:
            #align = DisplayJustification.Right
            display_format.justification = DisplayJustification.Right

        # prevents us from creating a massive array of break strings
        mask = self._r_mask
        if masked is True:
            mask = arange(len(column))
            
        # mask is now an array of numbers
        # when looping over a column

        cell_list = []
        # add the row break character to each column
        if self._row_break is not None:
            # get the head portion of the column
            cell_list = [DisplayCell(func(item, display_format), item, html=self._html_on) for item in column[mask[:self._row_break]]]
            
            # put in the break
            cell_list += [DisplayCell("...", "...", html=self._html_on)]

            # get the tail portion
            cell_list += [DisplayCell(func(item, display_format), item, html=self._html_on) for item in column[mask[self._row_break:]]]
        else:
            cell_list = [DisplayCell(func(item, display_format), item, html=self._html_on) for item in column[mask]]

        # TJD NOTE this is a natural location to color code the inner grid
        #cell_list[1].string="<td bgcolor=#00FFF>" + cell_list[1].string # + "</td>"
        #cell_list[1].color = DisplayColumnColors.GrayItalic
        
        # consider adding the break character here (one full, one for splits)
        new_column = DisplayColumn(cell_list,                   # build a display cell for every value in the masked column
                                        row_break = self._row_break, # a break character will be added for final table if necessary. hang on to index for now.
                                        color = None,
                                        header = header,
                                        #align = display_format.justification,
                                        html = self._html_on,
                                        itemformat = display_format,
                                        footer=footer)
        return new_column, footer

    # -------------------------------------------------------------------------------------
    def add_required_columns(self, header_names, table_data, footers, masked=False, gbkeys=None, 
                             transpose=False, color=None, style=None):
        '''
        header_names  : list of string header names (not tuples)
        table_data    : list of arrays
        footers       : list of footer rows (lists of ColHeader tuples)
        masked        : flag to indicate that the column has already been trimmed (build column does not need to apply a row mask)
        gbkeys        : dictionary of groupby keys - columns need to be painted differently
        '''
        table_columns = []
        widths = []
        footerval = None
        
        # TODO: support multikey here
        #if footers is not None:
        #    footers = footers[0]

        for i, column in enumerate(table_data):
            header = header_names[i]
            if footers is not None:
                footerval = [ f[i][0] for f in footers ]
                #footerval = footers[i][0]
            new_column, _ = self.build_column(header, column, masked=masked, style=style, footer=footerval)
            new_column._r_mask = self._r_mask
            
            # possibly paint entire column here
            if color is not None:
                new_column.paint_column(color)

            widths.append(new_column._max_width)
            table_columns.append(new_column)

            if gbkeys is not None and transpose:
                new_column.paint_column(DisplayColumnColors.Groupby, col_slice=slice(None, len(gbkeys)))

        # don't repeat labels in multikey groupby
        if gbkeys is not None and len(gbkeys)>1:
            self.fix_repeated_keys(table_columns, repeat_string='.')

        return table_columns, widths

    # -------------------------------------------------------------------------------------
    def fit_max_columns(self, headers, columns, total_width, console_width, footers=None):
        '''
        The display will attempt to fit as many columns as possible into the console.
        HTML display has been assigned a default value for self._console_x (see DisplayTable.__init__)
        
        If the user changes their self.options.COL_ALL to True, all columns will be displayed on the same line.
        *note: this will break console display for large tables and should only be used in jupyter lab now.

        *in progress
        If the user requested all columns to be shown - regardless of width, the display will split them up into
        separate views with the maximum columns per line.
        '''
        # ----------------------------------
        def build_break_column(nrows):
            #Builds a break column using the number of rows.
            #build_column() will add a row_break if necessary.
            breakstring = "..."
            col = np.full(nrows,breakstring).view(TypeRegister.FastArray)
            footer = [breakstring]
            if self._footer_tups is not None:
                footer = footer*len(self._footer_tups)
            return self.build_column(breakstring, col, masked=True, footer=footer)[0]
        # ----------------------------------


        force_all_columns = self.options.COL_ALL
        left_columns = []
        right_columns = []
        first_widths = []
        last_widths = []

        # check to see if forcing all columns to be displayed
        colbegin = 0
        colend = len(columns)-1

        # possibly build footers
        f_first = None
        f_last = None
        has_footers = False
        if footers is not None:
            has_footers = True
        
        #self._console_x -= 80
        while ((total_width <= console_width) or force_all_columns is True) and (colbegin <= colend):
            # pull from the front
            c = columns[colbegin]
            h_first = headers[colbegin]
            if has_footers: f_first = footers[colbegin]
            first_col, f_first = self.build_column(h_first, c, footer=f_first)
            d_first_width = first_col.display_width
            first_width = first_col._max_width

            # pull from the back
            c = columns[colend]
            h_last = headers[colend]
            if has_footers: f_last = footers[colend]
            last_col, f_last = self.build_column(h_last, c, footer=f_last)
            d_last_width = last_col.display_width
            last_width = last_col._max_width

            # if adding front column breaks console max
            if ((total_width + d_first_width) > console_width) and force_all_columns is False:
                self._col_break = colbegin
                break

            # front column fit
            else:
                #print("add to front",first_col._header)
                first_widths.append(first_width)
                left_columns.append(first_col)
                colbegin += 1
                total_width += d_first_width

                # break if the front just added the next back column
                # all columns were added
                if colbegin > colend:
                    break

                # break early if max has been reached
                if ((total_width + d_last_width) > console_width) and force_all_columns is False:
                    #print("max reached before last checked")
                    # if not all columns were added, set a column break
                    if (len(left_columns) + len(right_columns)) < len(columns):
                        self._col_break = colbegin
                    break

                # add column to the back list
                #print("add to back",last_col._header)
                last_widths.insert(0, last_width)
                right_columns.insert(0, last_col)
                colend-=1
                total_width += d_last_width

            if DisplayTable.DebugMode: print("total_width",total_width)
            if DisplayTable.DebugMode: print("console_x",console_width)
            if DisplayTable.DebugMode: print("colbegin",colbegin)
            if DisplayTable.DebugMode: print("colend",colend)
        
        
        # add the column break
        if self._col_break is not None:
            break_col = build_break_column(len(self._r_mask))
            first_widths.append(break_col._max_width)
            left_columns.append(break_col)

        # returns list of display columns (might have center gap), first widths, last widths
        # keep first and last widths separate to fix other headers spanning multiple columns
        return left_columns + right_columns, first_widths, last_widths

    # -------------------------------------------------------------------------------------
    def all_columns_console(self, console_width, left_offset, headers, columns):
        current_width = left_offset
        column_sets = [[]]
        column_widths = [[]]

        for col_index, c in enumerate(columns):
            h = headers[col_index]
            col, _ = self.build_column(h, c)
            d_width = col.display_width  # width with column padding (for measuring)
            width = col._max_width       # actual width of widest string

            # if too large for console, move to the next line
            if (current_width + d_width) > console_width:
                column_sets.append([])
                column_widths.append([])
                current_width = left_offset

            column_sets[-1].append(col)
            column_widths[-1].append(width)
            current_width += d_width

        return column_sets, column_widths

    # -------------------------------------------------------------------------------------
    def all_columns_console_multiline(self, console_width, left_offset, headers, columns):
        '''
        ** not implemented
        only supports two-line headers
        '''
        current_width = left_offset
        column_sets = [[]]
        column_widths = [[]]
        top_headers = [[]]
        bottom_headers = [[]]
        
        # keep each group together. for now this will only work for two line multi-column
        bottom_index = 0
        for header_tup in headers[0]:
            span = header_tup.cell_span
            current_cols = []
            current_headers = []
            current_d_widths = []
            current_widths = []

            # build column from each cell in bottom headers row
            for i in range(span):
                bottom_tup = headers[-1][bottom_index+i]
                c = columns[bottom_index+i]
                h = bottom_tup.col_name
                col, _ = self.build_column(h, c)
                current_cols.append(col)
                current_d_widths.append(col.display_width)
                current_widths.append(col._max_width)
                current_headers.append(bottom_tup)

            # if console width is broken, create a new column set
            if (current_width + sum(current_d_widths)) > console_width:
                column_sets.append([])
                column_widths.append([])
                top_headers.append([])
                bottom_headers.append([])
                current_width = left_offset

            # add info for all columns in current group
            for idx, col in enumerate(current_cols):
                column_sets[-1].append(col)
                column_widths[-1].append(current_widths[idx])
                bottom_headers[-1].append(current_headers[idx])
            top_headers[-1].append(header_tup)
            current_width += sum(current_d_widths)
            bottom_index += span

        header_sets = [ [top_headers[i], bottom_headers[i]] for i in range(len(top_headers)) ]
        return column_sets, column_widths, header_sets


    # -------------------------------------------------------------------------------------
    def build_transposed_columns(self, columns):
        '''
        Transposed column data needs to be constructed differently. Widths will be 
        calculated as a maximum items in multiple arrays.
        At the end of the table's construction, it will remain as a list of rows.
        '''
        #t_max = min(self.options.COL_T, self._nrows)
        t_max = min(15, self._nrows)
        # build column classes for EVERY column in the t table
        t_columns = []
        # groupby columns appear in the main table instead of the left table
        if self._gbkeys is not None:
            for gb in self._gbkeys.values():
                new_col = gb[:t_max]
                new_col, _ = self.build_column("", new_col, masked = True)
                new_col.paint_column(DisplayColumnColors.Groupby)
                t_columns.append(new_col)
        for column in columns:
            new_col = column[:t_max]
            new_col, _ = self.build_column("", new_col, masked = True)
            t_columns.append(new_col)


        # find the max width at each index (not very efficient)
        t_widths = []
        for i in range(t_max):
            max_width = len(max([c[i].string for c in t_columns], key=len))
            t_widths.append(max_width)

        # fit maximum number of columns in the console window
        t_display_widths = [w + DisplayConsoleTable.column_spacing for w in t_widths]
        total_width = self._total_width
        max_t_cols = 0
        for w in t_display_widths:
            total_width += w
            if total_width > self._console_x:
                break
            else:
                max_t_cols += 1

        self._main_first_widths = t_widths[:max_t_cols]
        
        # trim columns
        for i, col in enumerate(t_columns):
            t_columns[i]._data = col._data[:max_t_cols]

        self._nrows = self._ncols
        self._ncols = max_t_cols

        # set widths for individual cells in DisplayColumns
        for i,t in enumerate(t_columns):
            t_columns[i]._max_t_widths = self._main_first_widths

        return t_columns

    #---------------------------------------
    @staticmethod
    def display_detect():
        '''
        Call to redetect the display mode.
        This is useful when launching a qtconsole from jupyter lab.
        '''
        DisplayDetect.get_display_mode()

    @staticmethod
    def display_rows(rows=None):
        '''
        Parameters
        ----------
        rows: defaults to None.  How many top and bottom rows to display in a Dataset.
                  set to None to return the current rows.

        Contolled by Display.options.HEAD_ROWS/TAIL_ROWS

        See Also
        --------
        Display.options.TAIL_ROWS
        Display.options.HEAD_ROWS

        Examples
        --------
        rt.display_rows(20)
        '''
        if rows is None:
            return DisplayOptions.HEAD_ROWS, DisplayOptions.TAIL_ROWS
        DisplayOptions.TAIL_ROWS = rows
        DisplayOptions.HEAD_ROWS = rows

    @staticmethod
    def display_precision(precision=2):
        '''
        Parameters
        ----------
        precision: defaults to 2.  How many places after the decimal to display.
                  set to None to return the current precision.

        Examples
        --------
        rt.display_precision(4)
        '''

        if precision is None:
            return DisplayOptions.PRECISION
        DisplayOptions.PRECISION=precision
        DisplayOptions.P_THRESHOLD = None
        DisplayOptions.p_threshold()

    @staticmethod
    def display_threshold(threshold=6):
        '''
        Parameters
        ----------
        precision: defaults to 6.  How many powers of 10 before flipping to scientific notation.
                  set to None to return the current threshold.

        Notes
        -----
        E_THRESHOLD = 6          # power of 10 at which the float flips to scientific notation 10**+/-
        E_PRECISION = 3          # number of digits to display to the right of the decimal (sci notation)

        Examples
        --------
        rt.display_threshold(6)
        '''
        if threshold is None:
            return DisplayOptions.E_THRESHOLD

        DisplayOptions.E_THRESHOLD=threshold
        DisplayOptions.E_MIN = None
        DisplayOptions.E_MAX = None
        DisplayOptions.e_min()
        DisplayOptions.e_max()

    @staticmethod
    def display_html(html=None):
        '''
        Parameters
        ----------
        html: defaults to None.  Set to True to force html.
              set to None to return the current mode.
        '''
        if html is None:
            return DisplayOptions.HTML_DISPLAY
        DisplayOptions.HTML_DISPLAY=html

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
class DisplayHtmlTable:
    def __init__(self, headers, left_columns, main_columns, right_columns: Optional[list] = None, footers=None, axis_labels=False):
        if right_columns is None:
            right_columns = list()
        self.headers = headers
        self.left_columns = left_columns
        self.main_columns = main_columns
        self.right_columns = right_columns
        self.footers = footers
        
    def build_table(self):
        def join_row_section(rowstrings, idx=None):
            joined = ""
            numrows = len(rowstrings)
            if numrows > 0:
                if idx is None:
                    joined = "".join(rowstrings)
                elif i < numrows:
                    joined = "".join(rowstrings[i])
            return joined

        html_string_list = []
        display_id = str(np.random.randint(9999)) # unique id to fix bug in styling (jupyter lab likes to take over)
        css_prefix = "table.tbl-"+display_id

        # LEAVE THIS HERE
        # turn into fstring after editing

        #html_string_list.append("<html><head><style>")
        
        ## TODO: add support for axis labels
        ##html_string_list.append( css_prefix+" thead{border-bottom:none !important;}")
        ##html_string_list.append( css_prefix+" thead td:not(:first-child){font-weight:bold;border-bottom:var(--jp-border-width) solid var(--jp-border-color1);}")
        
        ## stops line break for cells in table body (headers will still break)
        #html_string_list.append( css_prefix+" tbody td{white-space: nowrap;}")

        ## main transparent colors
        #html_string_list.append( css_prefix+" .lc{font-weight:bold;background-color: var( --jp-rendermime-table-row-hover-background);}") # blue
        #html_string_list.append( css_prefix+" .lg{background-color: #66da9940;}")                                                         # green
        #html_string_list.append( css_prefix+" .lp{font-weight:bold;background-color: #ac66da40;}")                                        # purple
        #html_string_list.append( css_prefix+" .msc{font-weight:normal;background-color:#00000011;}")                                      # light shade
        
        ## text alignment
        #html_string_list.append( css_prefix+" .al{text-align:left;}")   # left
        #html_string_list.append( css_prefix+" .ar{text-align:right;}")  # right
        #html_string_list.append( css_prefix+" .ac{text-align:center;}") # center

        ## text style
        #html_string_list.append( css_prefix+" .bld{font-weight:bold;}")            # bold 
        #html_string_list.append( css_prefix+" .it{font-style:italic;}")            # italic
        #html_string_list.append( css_prefix+" .ul{text-decoration:underline;}")    # underline
        #html_string_list.append( css_prefix+" .st{text-decoration:line-through;}") # strikethrough

        #html_string_list.append("</style></head>")

        html_string_list.append(f"<html><head><style>{css_prefix} tbody td{{white-space: nowrap;}}{css_prefix} .lc{{font-weight:bold;background-color: var( --jp-rendermime-table-row-hover-background);}}{css_prefix} .lg{{background-color: #66da9940;}}{css_prefix} .lp{{font-weight:bold;background-color: #ac66da40;}}{css_prefix} .msc{{font-weight:normal;background-color:#00000011;}}{css_prefix} .al{{text-align:left;}}{css_prefix} .ar{{text-align:right;}}{css_prefix} .ac{{text-align:center;}}{css_prefix} .bld{{font-weight:bold;}}{css_prefix} .it{{font-style:italic;}}{css_prefix} .ul{{text-decoration:underline;}}{css_prefix} .st{{text-decoration:line-through;}}</style></head>")

        html_string_list.append(f"<body><table class='tbl-{display_id}'>")
        html_string_list.append("<thead>")
        for header_row in self.headers:
            html_string_list.append(f"<tr>{join_row_section(header_row)}</tr>")
        html_string_list.append("</thead><tbody>")

        # unlike starfish, in rt_display, left columns always present
        # all column groups contain same number of rows
        for i, left_cells in enumerate(self.left_columns):
            html_string_list.append(f"<tr>{join_row_section(left_cells)}{join_row_section(self.main_columns, i)}{join_row_section(self.right_columns, i)}</tr>")

        if self.footers is not None:
            for f in self.footers:
                html_string_list.append(f"<tr>{join_row_section(f)}</tr>")

        html_string_list.append("</tbody></table></body></html>")

        return "".join(html_string_list)

    def build_html_header(self):
        pass

    def build_html_table(self):
        pass


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
class DisplayConsoleTable:
    '''
    This class builds and returns a string with the final output for display in console.
    It is initialized with python lists of formatted strings.
    '''

    column_spacing = 3
    column_spacer = " " * column_spacing
    border_style = '-'
    def __init__(self, widths, headers, left_columns, main_columns, right_columns: Optional[list] = None, footers=None):
        if right_columns is None:
            right_columns = list()
        self.widths = widths
        self.headers = headers
        self.left_columns = left_columns
        self.main_columns = main_columns
        self.right_columns = right_columns
        self.footers = footers

        # need a use case when this block gets hit
        if len(main_columns) == 1:
            if len(main_columns[0]) == 0:
                self.main_columns = []
                #self.main_columns = None

    def build_table(self):
        # header
        table_str = self.build_end(is_header=True)
        # main
        for row in self.build_data():
            table_str.append(row)
        # table
        if self.footers is not None:
            for row in self.build_end(is_header=False):
                table_str.append(row)

        return "\n".join(table_str)

    def build_end(self, is_header=True):
        end_str = []
        
        if is_header is True: end_cells = self.headers
        else: end_cells = self.footers

        for end_row in end_cells:
            #print('end row',end_row)
            end_str.append(DisplayConsoleTable.column_spacer.join(end_row))

        # add border below header / above footer
        #if DisplayTable.options.BORDER:
        border_row = DisplayConsoleTable.column_spacer.join([DisplayConsoleTable.border_style*width for width in self.widths])
        if is_header is True: end_str.append(border_row)
        else: end_str.insert(0,border_row)

        return end_str
            
    def build_data(self):
        data_str = []
        
        datalist = []
        colgroups = [self.left_columns, self.main_columns, self.right_columns]
        for colgroup in colgroups:
            if len(colgroup) != 0:
                datalist.append(colgroup)

        column_data = zip(*datalist)

        for data in column_data:
            row = DisplayConsoleTable.column_spacer.join([i for sub in data for i in sub])
            data_str.append(row)

        return data_str


#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
class DisplayColumn:
    '''
        New display column class for holding final display data.

        Responsible for:
            -adding a row break character to each array at the appropriate spot
            -painting display cells
            -adding padding for console
            -replacing whitespace with &nbsp; for HTML
            -orchestrating math operations that span the whole column
    '''
    # names for individually adding styles to a <td> tag
    css_justification_classes = {
        DisplayJustification.Undefined      : "",
        DisplayJustification.Left           : "al",
        DisplayJustification.Right          : "ar",
        DisplayJustification.Center         : "ac"
    }
    css_decoration_classes = {
        DisplayTextDecoration.Undefined     : "",
        DisplayTextDecoration.Bold          : "bld",
        DisplayTextDecoration.Italic        : "it",
        DisplayTextDecoration.Underline     : "ul",
        DisplayTextDecoration.Strikethrough : "st"
    }
    # to be implemented
    css_color_classes = {}

    #---------------------------------------------------------------------------
    def __init__(self, data, row_break=None, color=None, header="", align=DisplayJustification.Right, html=False, itemformat=None, footer=None):
        
        # column data
        self._header = header
        self._data = data
        self._footer = footer

        # extra info
        self._row_break = row_break

        # style info
        self._color = color
        self._align = align
        self._html = html
        self._style = ColumnStyle(color=color, align=align)
        self._item_format = itemformat

        # also capture align, decoration (hoping to remove item format from this class)
        # need to fix transpose
        #if itemformat is not None:
        #    self._align = itemformat.justification


        # longest string repr of column data
        if len(data) > 0:
            self._max_width = len(max([cell.string for cell in data], key=len))
        else:
            self._max_width = len(header)

        if footer is None:
            self._max_width = max(self._max_width, len(header))

        # footers is a list of strings
        else:
            # measure all footer widths
            self._max_width = max(self._max_width, len(header), len(max(footer, key=len)))
            self._footer = footer

        # this will be used instead of _max_width for transposed columns (mixed types)
        self._max_t_widths = None

    #---------------------------------------------------------------------------
    def __getitem__(self, index):
        return self._data[index]

    #---------------------------------------------------------------------------
    def __setitem__(self, index, value):
        self._data[index] = value

    #---------------------------------------------------------------------------
    @property
    def display_width(self):
        '''
        When columns are being fit during display, need to account for padding between them, otherwise 
        overflow will occur and table will break.
        '''
        return self._max_width + TypeRegister.DisplayOptions.X_PADDING

    #---------------------------------------------------------------------------
    def build_summary(self):
        col_stats = []
        col_stats.append(f"    html: {self._html}")
        col_stats.append(f"   align: {self._align}")
        col_stats.append(f"rowbreak: {self._row_break}")
        col_stats.append(f"maxwidth: {self._max_width}")
        col_stats.append(f"   color: {self._color}")
        col_stats.append(f"  header: {self._header}")
        col_stats.append(f"    data: {self._data}")
        col_stats.append(f"  footer: {self._footer}")
        return "\n".join(col_stats)

    #---------------------------------------------------------------------------
    @property
    def header(self): return self._header

    @property
    def footer(self): return self._footer

    #---------------------------------------------------------------------------
    def add_footer(self, footer):
        # width needs to be updated if a footer is added
        self._footer = footer
        self._max_width = max(self._max_width, len(footer))

    #---------------------------------------------------------------------------
    def build_header(self, final_color=None, plain=False, align=None):
        if align is None:
            align = self._item_format.justification
            #align = self._style.align
        return self._build_end(self._header, final_color=final_color, plain=plain, justification=align)

    #---------------------------------------------------------------------------
    def build_footer(self, final_color=None, plain=False):
        if self._footer is None:
            return [None]
        if final_color is None:
            final_color = DisplayColumnColors.Groupby
        if isinstance(self._footer, list):
            return [self._build_end(f, final_color=final_color, plain=plain, justification=DisplayJustification.Right) for f in self._footer]
        return self._build_end(self._footer, final_color=final_color, plain=plain, justification=DisplayJustification.Right)

    #---------------------------------------------------------------------------
    def _build_end(self, end, final_color=None, plain=False, justification=None):
        if end is None:
            return

        if final_color is not None:
            color = final_color
        else:
            if self._color is None:
                color = DisplayColumnColors.Rownum
            else:
                color = self._color
        styled_end = DisplayCell(end, color=color, html=self._html)
        
        # for transposed tables
        if self._max_t_widths is not None:
            width = max(self._max_t_widths)
        else:
            width = self._max_width

        if self._html:
            styled_end.paint_cell()
            if end is self._header:
                styled_end.prefix += " " + self.text_decoration_html(DisplayTextDecoration.Bold) # headers will always be bold
            styled_end.prefix += " " + self.align_html(justification)
        styled_end.string = self.align_console_string(styled_end.string, width, align=justification)
        styled_end.color = color
        if plain is True:
            return styled_end.string
        return str(styled_end.display(html=self._html))

    #---------------------------------------------------------------------------
    def paint_column(self, color, col_slice=None, badrows=None):
        '''
        Called when a column needs to be colored
        For instance the left side may be row numbered
        '''
        if color is not None:
            if isinstance(color, ColumnStyle):
                color = color.color

            if col_slice is None:
                data = self._data
                self._color = color
            else:
                # will badrows works for a slice?
                data = self._data[col_slice]

            self._style.color = color
            for cell in data:
                # change all the colors in the cells for this column
                cell.color = color

        if badrows is not None:
            for k,v in badrows.items():
                self._data[k].color = v.color

    #---------------------------------------------------------------------------
    def align_column(self, align, col_slice=slice(None,None,None)):
        if col_slice == slice(None, None, None):
            self._align = align
        for cell in self._data[col_slice]:
            cell.align = align

    #---------------------------------------------------------------------------
    def style_column(self, style, col_slice=None, badrows=None):
        '''
        This will replace paint_column as a single styling call for all column properties.
        '''
        if style is None:
            # can be none when badrows exists but not badcols
            self.paint_column( None, col_slice=col_slice, badrows=badrows )
        else:
            self.paint_column( style.color, col_slice=col_slice, badrows=badrows )
            # testing - item format will be removed soon
            self._item_format.justification = style.align

    #---------------------------------------------------------------------------
    def paint_posneg(self):
        '''
        *** not implemented, maybe for future Dataset.style() call
        Will paint positive values green, negative values red for all numeric columns in IPython console.
        '''
        def apply_posneg(cell):
            if cell.value > 0:
                cell.color = DisplayColumnColors.Sort
            elif cell.value < 0:
                cell.color = DisplayColumnColors.Red
            return cell

        # bail early on non-computable
        if isinstance(self._data[0].value, (str,bytes)):
            return

        if self._row_break is None:
            for i, cell in enumerate(self._data):
                self._data[i] = apply_posneg(cell)
        else:
            for i, cell in enumerate(self._data[:self._row_break]):
                self._data[i] = apply_posneg(cell)
            for i, cell in enumerate(self._data[self._row_break+1:]):
                self._data[i] = apply_posneg(cell)

    #---------------------------------------------------------------------------
    def paint_highlightmax(self):
        '''
        *** not implemented, maybe for future Dataset.style() call
        Will paint max value of each numeric column gold.
        '''
        def get_argmax(dataslice):
            vals = []
            for i, cell in enumerate(dataslice):
                vals.append(cell.value)
            return np.nanargmax(vals)

        # bail early on non-computable
        if isinstance(self._data[0].value, (str, bytes)):
            return

        if self._row_break is None:
            idx = get_argmax(self._data)
            dataslice = self._data.copy()
            dataslice[idx].color = DisplayColumnColors.Groupby
        else:
            dataslicetop = self._data[:self._row_break]
            idxtop = get_argmax(dataslicetop)
            maxtop = dataslicetop[idxtop].value

            dataslicebtm = self._data[self._row_break+1:]
            idxbtm = get_argmax(dataslicebtm)
            maxbtm = dataslicebtm[idxbtm].value

            if maxtop > maxbtm:
                dataslicetop[idxtop].color = DisplayColumnColors.Groupby
            elif maxbtm > maxtop:
                dataslicebtm[idxbtm].color = DisplayColumnColors.Groupby
            else:
                dataslicetop[idxtop].color = DisplayColumnColors.Groupby
                dataslicebtm[idxbtm].color = DisplayColumnColors.Groupby

    #---------------------------------------------------------------------------
    def __repr__(self):
        return self.build_summary()

    #---------------------------------------------------------------------------
    def __str__(self):
        return self.build_summary()

    #---------------------------------------------------------------------------
    def styled_string_list(self):
        result = []
        # adjust the column data color for multiset comparisons (readability in final display)
        if self._color in (DisplayColumnColors.Multiset_head_a, DisplayColumnColors.Multiset_head_b) and len(self._data)!=1:
            self._color += 2

        if self._html is False:
            # ANSI Console pathway
            # pad the strings if not html
            for i, cell in enumerate(self._data):
                # transposed tables might use a different width for each cell
                max_width = self._max_width
                if self._max_t_widths is not None:
                    max_width = self._max_t_widths[i]
                # alignment may be set for an individual cell
                justification = self._item_format.justification
                if cell.align:
                    justification = cell.align
                cell.string = self.align_console_string(cell.string, max_width, align=justification)
                if cell.color is None:
                    cell.color = self._color
                # replace plain text with ansi escape text?
                self._data[i] = str(cell.display(html=False))
        else:
            # HTML pathway
            for i, cell in enumerate(self._data):
                if cell.color is None:
                    cell.color = self._color
                cell.paint_cell()
                item_format = self._item_format.copy()
                # Careful in ordering this condition since it depends on paint_cell and item_format initialization.
                if not cell.color:
                    # cell.paint_cell is responsible for adding the class attribute depending if there
                    # is a color, but we need to also consider item_format since that also adds class
                    # level styles.
                    if item_format.justification or item_format.decoration:
                        cell.prefix += " class='"
                # alignment may be set for an individual cell
                if cell.align:
                    item_format.justification = cell.align
                cell.prefix += self.style_classes_html(item_format)
                result.append(str(cell.display(html=True)))
            return result

        return self._data
        #return result

    #---------------------------------------------------------------------------
    def plain_string_list(self):
        if self._max_t_widths is not None:
            for i, cell in enumerate(self._data):
                self._data[i] = self.align_console_string(cell.string, self._max_t_widths[i], align=self._item_format.justification)
        else:
            for i, cell in enumerate(self._data):
                self._data[i] = self.align_console_string(cell.string, self._max_width, align=self._item_format.justification)
        #return [cell.string for cell in self._data]
        return self._data

    #---------------------------------------------------------------------------
    @property
    def data(self):
        '''
        Return string data array. Note: does not check for any formatting that has been applied.
        This routine is to assist in modifying repeated values in multikey groupby operations.
        '''
        return self._data

    #---------------------------------------------------------------------------
    @staticmethod
    def align_console_string(string, width, align=DisplayJustification.Right):
        """
        Pad string for correct alignment in console table columns.
        """
        whitespace = width - len(string)
        left_space = ""
        right_space = ""
        if align == DisplayJustification.Right:
            left_space = whitespace * " "
        elif align == DisplayJustification.Left:
            right_space = whitespace * " "
        # center multiline headers
        elif align == DisplayJustification.Center:
            left_space = int(np.floor(whitespace / 2)) * " "
            right_space = int(np.ceil(whitespace / 2)) * " "
        else:
            print("Invalid alignment. Returning unpadded string.")

        return left_space + string + right_space

    #---------------------------------------------------------------------------
    @staticmethod
    def style_classes_html(itemformat=None):
        '''
        Calls routines to add a CSS class for every styling option in the ItemFormat object
        '''
        class_list = [" "]
        if itemformat is not None:
            class_list.append(DisplayColumn.align_html(itemformat.justification))
            class_list.append(DisplayColumn.text_decoration_html(itemformat.decoration))
        return " ".join(class_list)

    #---------------------------------------------------------------------------
    @staticmethod
    def align_html(align=DisplayJustification.Right):
        return DisplayColumn.css_justification_classes.get(align,"")

    #---------------------------------------------------------------------------
    @staticmethod
    def text_decoration_html(style=None):
        return DisplayColumn.css_decoration_classes.get(style,"")


#---------------------------------------------------------------------------
class DisplayCell:
    '''
        Wrapper around a string for styled console or html display.
        Original value can also be stored for future column or table-wide math operations.
    '''
    # see http://github.com/jupyterlab/jupyterlab/blob/master/packages/notebook/style/base.css
    # and index.css
    # master/packages/theme-light-extension/style/varaibles.css

    # HTML (found in jupyter lab as of October 2019)
    #               .lc{font-weight:bold;
    #table.tbl-4111 .lg{background-color: #66da9940;}
    #table.tbl-4111 .lp{font-weight:bold;background-color: #ac66da40;}
    #table.tbl-4111 .msc{font-weight:normal;background-color:#00000011;}
    #table.tbl-4111 .al{text-align:left;}
    #table.tbl-4111 .ar{text-align:right;}
    #table.tbl-4111 .ac{text-align:center;}
    #table.tbl-4111 .bld{font-weight:bold;}
    #table.tbl-4111 .it{font-style:italic;}
    #table.tbl-4111 .ul{text-decoration:underline;}
    #table.tbl-4111 .st{text-decoration:line-through;}</style>

     #     ISO 6429 color sequences are composed of sequences of numbers separated
     #     by semicolons.  The most common codes are:

	 # 0	to restore default color
	 # 1	for brighter colors
	 # 4	for underlined text
	 # 5	for flashing text
	 #30	for black foreground
	 #31	for red foreground
	 #32	for green foreground
	 #33	for yellow (or brown) foreground
	 #34	for blue foreground
	 #35	for purple foreground
	 #36	for cyan foreground
	 #37	for white (or gray) foreground
	 #40	for black background
	 #41	for red background
	 #42	for green background
	 #43	for yellow (or brown) background
	 #44	for blue background
	 #45	for purple background
	 #46	for cyan background
	 #47	for white (or gray) background
     #     Not all commands will work on all systems or display devices.

    # color palette for dark backgrounds
    darkbg_styles = {
        "html_styles" : {
            #DisplayColumnColors.Default : "<td class='",           # no color
            DisplayColumnColors.Default : "<td",                  # no color
            DisplayColumnColors.Rownum  : "<td class='lc",         # blue
            DisplayColumnColors.Sort    : "<td class='lg",         # green
            DisplayColumnColors.Groupby : "<td class='lg",         # green
            DisplayColumnColors.Multiset_head_a : "<td class='lc", # blue
            DisplayColumnColors.Multiset_head_b : "<td class='lp", # purple
            DisplayColumnColors.Multiset_col_a  : "<td class='",       # no color
            DisplayColumnColors.Multiset_col_b  : "<td class='msc",    # light shade
            DisplayColumnColors.Purple          : "<td class='lp", # purple
            DisplayColumnColors.Pink            : "<td class='lp", # purple
            DisplayColumnColors.Red             : "<td><font color=#F00000", # red fg
            DisplayColumnColors.GrayItalic      : "<td class='msc it", # gray italic
            DisplayColumnColors.DarkBlue        : "<td class='c",  # 
            DisplayColumnColors.BGColor         : "<td bgcolor=#",  # 
            DisplayColumnColors.FGColor         : "<td><font color=#",  # 
            #DisplayColumnColors.RedBG           : "<td bgcolor=#F00000", # 
        },
        "console_styles" : {
            DisplayColumnColors.Default         : "",           # no color
            DisplayColumnColors.Rownum          : "\x1b[1;36m", # blue (light cyan)
            DisplayColumnColors.Sort            : "\x1b[1;32m", # green
            DisplayColumnColors.Groupby         : "\x1b[1;33m", # yellow  
            DisplayColumnColors.Multiset_head_a : "\x1b[1;36m", # blue
            DisplayColumnColors.Multiset_head_b : "\x1b[1;32m", # green
            DisplayColumnColors.Multiset_col_a  : '\x1b[0m',    # gray
            DisplayColumnColors.Multiset_col_b  : '\x1b[1;37m', # white
            DisplayColumnColors.Purple          : '\x1b[2;35m', # purple
            DisplayColumnColors.Pink            : '\x1b[1;35m', # purple
            DisplayColumnColors.Red             : '\x1b[1;41m', # red
            DisplayColumnColors.GrayItalic      : '\x1b[0;41m', # red bg instead of gray
            DisplayColumnColors.DarkBlue        : '\x1b[1;34m', # dkblue
            DisplayColumnColors.BGColor         : "",  # 
            DisplayColumnColors.FGColor         : "",  # 
        },
        "html_end" : "</td>",
        "console_end" : "\x1b[0m"
    }

    # color palette for light backgrounds
    lightbg_styles = {
        "html_styles" : {
            DisplayColumnColors.Default : "<td class='",            # no color
            DisplayColumnColors.Rownum  : "<td class='lc", # blue
            DisplayColumnColors.Sort    : "<td class='lg", # green
            DisplayColumnColors.Groupby : "<td class='lg", # green
            DisplayColumnColors.Red     : "<td><font color=#600000", # red fg
            DisplayColumnColors.GrayItalic     : "<td class='msc it", # gray italic
            DisplayColumnColors.DarkBlue: "<td class='c",  # 
            DisplayColumnColors.GrayItalic      : "<td class='msc it", # gray italic
            DisplayColumnColors.DarkBlue        : "<td class='c",  # 
            DisplayColumnColors.BGColor         : "<td bgcolor=#",  # 
            DisplayColumnColors.FGColor         : "<td><font color=#",  # 
        },
        "console_styles" : {
            DisplayColumnColors.Default         : "",           # no color
            DisplayColumnColors.Rownum          : "\x1b[0;34m", # blue (dark)
            DisplayColumnColors.Sort            : "\x1b[2;32m", # green (dark)
            DisplayColumnColors.Groupby         : "\x1b[2;35m", # purple
            DisplayColumnColors.Multiset_head_a : "\x1b[1;34m", # blue
            DisplayColumnColors.Multiset_head_b : "\x1b[1;32m", # green
            DisplayColumnColors.Multiset_col_a  : '\x1b[0m',    # gray
            DisplayColumnColors.Multiset_col_b  : '\x1b[1;37m', # white
            DisplayColumnColors.Purple          : '\x1b[2;35m', # purple
            DisplayColumnColors.Pink            : '\x1b[2;35m', # purple
            DisplayColumnColors.Red             : '\x1b[0;31m', # red
            DisplayColumnColors.GrayItalic      : '\x1b[0;41m', # red bg instead of gray
            DisplayColumnColors.DarkBlue        : '\x1b[0;34m', # dkblue
            DisplayColumnColors.BGColor         : "",  # 
            DisplayColumnColors.FGColor         : "",  # 
        },
        "html_end" : "</td>",
        "console_end" : "\x1b[0m"
    }

    # color palette for no styling (seems trivial, but allows us to display cells with same routine)
    no_styles = {
        "html_styles" : {
            DisplayColumnColors.Default : "<td",
            DisplayColumnColors.Rownum  : "<td",
            DisplayColumnColors.Sort    : "<td",
            DisplayColumnColors.Groupby : "<td",
            DisplayColumnColors.Red     : "<td",
            DisplayColumnColors.GrayItalic      : "<td",
            DisplayColumnColors.DarkBlue: "<td",
            DisplayColumnColors.BGColor : "<td",  # 
            DisplayColumnColors.FGColor : "<td",  # 
        },
        "console_styles" : {
            DisplayColumnColors.Default         : "",
            DisplayColumnColors.Rownum          : "",
            DisplayColumnColors.Sort            : "",
            DisplayColumnColors.Groupby         : "",
            DisplayColumnColors.Multiset_head_a : "",
            DisplayColumnColors.Multiset_head_b : "",
            DisplayColumnColors.Multiset_col_a  : "",
            DisplayColumnColors.Multiset_col_b  : "",
            DisplayColumnColors.Red             : "", 
            DisplayColumnColors.DarkBlue        : "", 
            DisplayColumnColors.BGColor         : "",  # 
            DisplayColumnColors.FGColor         : "",  # 
        },
        "html_end" : "</td>",
        "console_end" : ""
    }
    
    color_mode_dict = {
        DisplayColorMode.NoColors : no_styles,
        DisplayColorMode.Light    : lightbg_styles,
        DisplayColorMode.Dark     : darkbg_styles
    }

    #----------------------------------------------------------------------------
    def __init__(self, string, value=None, color=None, html=False, colspan=None,
                 align=None):
        self.prefix = ""
        self.suffix = ""
        self.string = string
        self.value = value
        self.color = color
        self.html = html
        self.colspan = colspan
        self.align = align

    #----------------------------------------------------------------------------
    def __repr__(self):
        return f"DisplayCell({self.string}, value={self.value}, color={self.color}, html={self.html}, colspan={self.colspan}, align={self.align})"

    #----------------------------------------------------------------------------
    def __str__(self):
        return self.string

    #----------------------------------------------------------------------------
    def __len__(self):
        return len(self.string)

    #----------------------------------------------------------------------------
    def display(self, html=False, plain=False):
        if plain: return self.string
        # apply color return the final styled string
        if self.html:
            if self.colspan is not None:
                self.prefix += "' colspan='" + str(self.colspan)
            self.prefix+="'>"
        else:
            self.paint_cell()

        #NOTE this will often close the '<td' with a '>'
        # then add the string contents of the cell
        # then add '</td>' to end the table data for html
        return self.prefix+self.string+self.suffix

    #----------------------------------------------------------------------------
    def paint_cell(self):
        if self.color is None:
            self.color = DisplayColumnColors.Default
        # set the prefix and suffix, otherwise leave them blank
        color_mode = DisplayCell.color_mode_dict[DisplayDetect.ColorMode]
        if self.html:
            style_dict = color_mode["html_styles"]
            self.suffix = color_mode["html_end"]
        else:
            style_dict = color_mode["console_styles"]
            self.suffix = color_mode["console_end"]
        self.prefix = style_dict[self.color]


# get the current display mode
DisplayDetect.get_display_mode()

TypeRegister.DisplayTable = DisplayTable
TypeRegister.DisplayDetect = DisplayDetect
TypeRegister.DisplayString = DisplayString
TypeRegister.DisplayAttributes = DisplayAttributes
TypeRegister.DisplayText = DisplayText
