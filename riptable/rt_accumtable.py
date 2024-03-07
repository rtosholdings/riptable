__all__ = ["AccumTable", "accum_ratio", "accum_ratiop", "accum_cols"]


import warnings
from collections import OrderedDict

import numpy as np

from .rt_accum2 import Accum2
from .rt_categorical import Categorical
from .rt_enum import TypeRegister
from .rt_numpy import full


class AccumTable(Accum2):
    """
    Enables the creation of tables with values calculated by various reducing functions.

    :py:class:`~.rt_accumtable.AccumTable` is a wrapper on :py:class:`~.rt_accum2.Accum2`
    and can generate tables with multiple footer rows and margin columns, which
    represent values calculated by a variety of reducing functions.

    An :py:class:`~.rt_accumtable.AccumTable` holds multiple tables at once. For
    example, an :py:class:`~.rt_accumtable.AccumTable` can hold the tables calculated by
    the mean, sum, and variance reducing functions. All tables in the
    :py:class:`~.rt_accumtable.AccumTable` are grouped by the same two
    :py:class:`~.rt_categorical.Categorical` objects.

    Each table in the :py:class:`~.rt_accumtable.AccumTable` has these three parts:

    * **Inner table** - a table of values calculated by a reducing function and indexed
      by row and column groups.
    * **Footer row** - a row on the bottom margin that contains the calculated value
      for each column group.
    * **Margin column** - a column on the right margin that contains the calculated
      value for each row group.

    After creating an :py:class:`~.rt_accumtable.AccumTable`, you can generate a
    :py:class:`~.rt_dataset.Dataset` to view the calculated values as a table. You can
    customize the generated table by specifying one inner table, a set of footer rows,
    and a set of margin columns.

    You create an :py:class:`~.rt_accumtable.AccumTable` and generate a table with the
    following multistep process:

    #. Pass two :py:class:`~.rt_categorical.Categorical` objects to create an
       :py:class:`~.rt_accumtable.AccumTable` and to specify the row and column groups.
    #. Add tables to the :py:class:`~.rt_accumtable.AccumTable` by setting its elements
       to :py:class:`~.rt_dataset.Dataset` objects of values calculated by a reducing
       function. For a list of reducing functions, see
       :doc:`/tutorial/tutorial_cat_reduce`.
    #. Specify which summary rows and columns you want to include in a generated table
       using :py:meth:`~.rt_accumtable.AccumTable.set_footer_rows` and
       :py:meth:`~.rt_accumtable.AccumTable.set_margin_columns`.
    #. Generate a table view with the specified summary rows and columns using
       :py:meth:`~.rt_accumtable.AccumTable.gen`.

    Parameters
    ----------
    cat_rows : :py:class:`~.rt_categorical.Categorical`
        The row groups used to accumlate the values.
    cat_cols : :py:class:`~.rt_categorical.Categorical`
        The column groups used to accumlate the values.
    filter : ndarray
        Boolean mask array applied to arrays before grouping, reducing, and addition
        to the :py:class:`~.rt_accumtable.AccumTable`.
    showfilter : bool
        Controls whether the returned table contains row or column groups that result
        entirely in ``0`` or ``nan`` when the filter is applied.

    See Also
    --------
    :py:class:`.rt_accum2.Accum2` :
        The parent class for :py:class:`~.rt_accumtable.AccumTable`.
    :py:class:`.rt_categorical.Categorical` :
        A class that efficiently stores an array of repeated strings and is used for
        groupby operations.
    :py:class:`.rt_groupbyops.GroupByOps` :
        A class that holds the reducing functions used to create an
        :py:class:`~.rt_accumtable.AccumTable`.

    Examples
    --------
    Construct a :py:class:`~.rt_dataset.Dataset` for the following examples:

    >>> ds = rt.Dataset()
    >>> ds.Zeros = [0, 0, 0, 0, 0]
    >>> ds.Ones = [1, 1, 1, 1, 1]
    >>> ds.Twos = [2, 2, 2, 2, 2]
    >>> ds.Nans = [rt.nan, rt.nan, rt.nan, rt.nan, rt.nan]
    >>> ds.Ints = [0, 1, 2, 3, 4]
    >>> ds.Groups = rt.Cat(["Group1", "Group2", "Group1", "Group1", "Group2"])
    >>> ds.Letters = rt.Cat(["A", "B", "C", "A", "C"])
    >>> ds
    #   Zeros   Ones   Twos   Nans   Ints   Groups   Letters
    -   -----   ----   ----   ----   ----   ------   -------
    0       0      1      2    nan      0   Group1   A
    1       0      1      2    nan      1   Group2   B
    2       0      1      2    nan      2   Group1   C
    3       0      1      2    nan      3   Group1   A
    4       0      1      2    nan      4   Group2   C
    <BLANKLINE>
    [5 rows x 7 columns] total bytes: 225.0 B

    **Create an AccumTable**

    Pass two :py:class:`~.rt_categorical.Categorical` objects to create the row and
    column groups for the :py:class:`~.rt_accumtable.AccumTable`:

    >>> at = rt.AccumTable(ds.Groups, ds.Letters)
    >>> at
    Inner Tables: []
    Margin Columns: []
    Footer Rows: []

    The :py:class:`~.rt_accumtable.Accumtable` doesn't yet hold any inner tables. Add a
    table using a reducing function. This example adds a table with values calculated by
    :py:meth:`~.rt_groupbyops.GroupByOps.count`:

    >>> at["Count"] = at.count()
    >>> at["Count"]
    *Groups   A   B   C   Count
    -------   -   -   -   -----
    Group1    2   0   1       3
    Group2    0   1   1       2
    -------   -   -   -   -----
      Count   2   1   2       5
    <BLANKLINE>
    [2 rows x 5 columns] total bytes: 52.0 B

    The :py:class:`~.rt_accumtable.AccumTable` now holds the Count table:

    >>> at
    Inner Tables: ['Count']
    Margin Columns: ['Count']
    Footer Rows: ['Count']

    Add more tables to the :py:class:`~.rt_accumtable.AccumTable` using different
    reducing functions:

    >>> at["Sum Ints"] = at.sum(ds.Ints)
    >>> at["Mean Double"] = at.mean(ds.Ints * ds.Twos)
    >>> at["Variance Ints"] = at.var(ds.Ints)
    >>> at
    Inner Tables: ['Count', 'Sum Ints', 'Mean Double', 'Variance Ints']
    Margin Columns: ['Count', 'Sum Ints', 'Mean Double', 'Variance Ints']
    Footer Rows: ['Count', 'Sum Ints', 'Mean Double', 'Variance Ints']

    Generate a table with multiple summary rows and columns using
    :py:meth:`~.rt_accumtable.AccumTable.gen`. Pass the name of the inner table that you
    want to include in the generated table:

    >>> at.gen("Sum Ints")
    *Groups            A      B      C   Sum Ints   Count   Mean Double   Variance Ints
    -------------   ----   ----   ----   --------   -----   -----------   -------------
    Group1             3      0      2          5       3          3.33            2.33
    Group2             0      1      4          5       2          5.00            4.50
    -------------   ----   ----   ----   --------   -----   -----------   -------------
         Sum Ints      3      1      6         10
            Count      2      1      2                  5
      Mean Double   3.00   2.00   6.00                             4.00
    Variance Ints   4.50    nan   2.00                                             2.00
    <BLANKLINE>
    [2 rows x 8 columns] total bytes: 124.0 B

    By default, all summary rows and columns appear in the generated table. Specify
    which summary rows and columns appear using
    :py:meth:`~.rt_accumtable.AccumTable.set_footer_rows` and
    :py:meth:`~.rt_accumtable.AccumTable.set_margin_columns`:

    >>> at.set_footer_rows(["Count", "Sum Ints"])
    >>> at.set_margin_columns(["Variance Ints"])
    >>> at
    Inner Tables: ['Count', 'Sum Ints', 'Mean Double', 'Variance Ints']
    Margin Columns: ['Variance Ints']
    Footer Rows: ['Count', 'Sum Ints']

    Generate the table with the specified summary rows and columns:

    >>> at.gen("Sum Ints")
    *Groups    A   B   C   Sum Ints   Variance Ints
    --------   -   -   -   --------   -------------
    Group1     3   0   2          5            2.33
    Group2     0   1   4          5            4.50
    --------   -   -   -   --------   -------------
    Sum Ints   3   1   6         10
       Count   2   1   2
    <BLANKLINE>
    [2 rows x 6 columns] total bytes: 92.0 B
    """

    # -------------------------------------------------------
    def __init__(cls, cat_rows, cat_cols, filter=None, showfilter=False):
        pass

    def __new__(cls, cat_rows, cat_cols, filter=None, showfilter=False):
        instance = super(AccumTable, cls).__new__(cls, cat_rows, cat_cols, filter, showfilter)
        instance._inner = OrderedDict()
        instance._rows = OrderedDict()
        instance._cols = OrderedDict()
        instance._default_inner_name = None
        return instance

    # -------------------------------------------------------
    def __repr__(self):
        """
        Return a string representation of the :py:class:`~.rt_accumtable.AccumTable`.

        Returns
        -------
        str
            The :py:class:`~.rt_accumtable.AccumTable` as a string.
        """
        res = "Inner Tables: " + str(list(self._inner.keys())) + "\n"
        res += "Margin Columns: " + str(list(self._cols.keys())) + "\n"
        res += "Footer Rows: " + str(list(self._rows.keys()))
        return res

    # -------------------------------------------------------
    def __setitem__(self, name: str, ds):
        """
        Add an inner table, corresponding footer row, and corresponding margin column to
        the :py:class:`~.rt_accumtable.AccumTable`.

        Parameters
        ----------
        name : str
            Name of the inner table and its corresponding footer row and margin column.
        ds : :py:class:`~.rt_dataset.Dataset`
            The :py:class:`~.rt_dataset.Dataset` that provides data for the inner table,
            footer row, and margin column.

        Raises
        ------
        IndexError
            If ``name`` is not a string.
        ValueError
            If ``ds`` is not a :py:class:`~.rt_dataset.Dataset`.
        """
        if not type(name) is str:
            raise IndexError("name must be a string table name")
        if not isinstance(ds, TypeRegister.Dataset):
            raise ValueError("ds must be a Dataset")
        self._inner[name] = ds
        self._rows[name] = None
        self._cols[name] = None
        self._rename_summary_row_and_col(ds, name)
        self._default_inner_name = name

    # -------------------------------------------------------
    def __getitem__(self, index: str):
        """
        Return the inner table, footer row, and margin column corresponding to ``index``.

        Parameters
        ----------
        index : str
            Name of the inner table, footer row, and margin column to return.

        Returns
        -------
        :py:class:`~.rt_dataset.Dataset`
            The inner table, footer row, and margin column corresponding to ``index``.

        Raises
        ------
        IndexError
            If ``index`` is not a string.
        """
        if not type(index) is str:
            raise IndexError("Index must be a string table name")
        self._default_inner_name = index
        return self._inner[index]

    # -------------------------------------------------------
    def _rename_summary_row_and_col(self, ds, new_name: str):
        """
        Parameters
        ----------
        ds : Dataset
            The dataset
        new_name : str
            the new name for the summary column and footer row

        Returns
        -------
        Dataset
        """
        col_names = ds.summary_get_names()
        if len(col_names) == 1:
            ds.col_rename(col_names[0], new_name)
        footers = ds.footer_get_dict()
        if len(footers) == 1:
            old_name = list(footers.keys())[0]
            nd = list(footers.values())[0]
            ds.footer_remove(old_name)
            ds.footer_set_values(new_name, nd)
        return ds

    # -------------------------------------------------------
    def gen(self, table_name=None, format=None, ref_table=None, remove_blanks=True):
        """
        Generate a table with one inner table and multiple footer rows and margin
        columns from an :py:class:`~.rt_accumtable.AccumTable`.

        Parameters
        ----------
        table_name : str, optional
            The name of the inner table that appears in the generated table. If not
            provided, the last-created inner table appears in the generated table.
        format : dict of {str : func}, optional
            (Not yet implemented) A dictionary used to specify the formatting of each
            cell in the table. Each key is a formatting type, such as "bold", "color",
            and "background", and each value is a function that applies conditional
            formatting to each table cell. For example, ``format={"bold": lambda v: v > 0}``
            applies bold formatting to all cells with positive values.
        ref_table : str or :py:class:`~.rt_dataset.Dataset`, optional
            (Not yet implemented) The name of an :py:class:`~.rt_accumtable.AccumTable`
            or a :py:class:`~.rt_dataset.Dataset` of the same shape that acts as a
            format reference for the generated table.
        remove_blanks : bool, default `True`
            Controls whether rows and columns consisting entirely of ``0`` or ``nan``
            are removed from the generated table.

        Returns
        -------
        :py:class:`.rt_dataset.Dataset`
            A table generated from the :py:class:`~.rt_accumtable.AccumTable`, including
            footer rows and margin columns.

        See Also
        --------
        :py:class:`.rt_accumtable.AccumTable` :
            The class containing :py:meth:`~.rt_accumtable.AccumTable.gen`.
        :py:meth:`.rt_accumtable.AccumTable.set_footer_rows` :
            The method that sets the footer rows for the :py:class:`.rt_accumtable.AccumTable`
            and its generated tables.
        :py:meth:`.rt_accumtable.AccumTable.set_margin_columns` :
            The method that sets the margin columns for the :py:class:`.rt_accumtable.AccumTable`
            and its generated tables.

        Examples
        --------
        Construct a :py:class:`~.rt_dataset.Dataset` for the following examples:

        >>> ds = rt.Dataset()
        >>> ds.Zeros = [0, 0, 0, 0, 0]
        >>> ds.Ones = [1, 1, 1, 1, 1]
        >>> ds.Twos = [2, 2, 2, 2, 2]
        >>> ds.Nans = [rt.nan, rt.nan, rt.nan, rt.nan, rt.nan]
        >>> ds.Ints = [0, 1, 2, 3, 4]
        >>> ds.Groups = rt.Cat(["Group1", "Group2", "Group1", "Group1", "Group2"])
        >>> ds.Letters = rt.Cat(["A", "B", "C", "A", "C"])
        >>> ds
        #   Zeros   Ones   Twos   Nans   Ints   Groups   Letters
        -   -----   ----   ----   ----   ----   ------   -------
        0       0      1      2    nan      0   Group1   A
        1       0      1      2    nan      1   Group2   B
        2       0      1      2    nan      2   Group1   C
        3       0      1      2    nan      3   Group1   A
        4       0      1      2    nan      4   Group2   C
        <BLANKLINE>
        [5 rows x 7 columns] total bytes: 225.0 B

        Construct an :py:class:`~.rt_accumtable.AccumTable` from that data:

        >>> at = rt.AccumTable(ds.Groups, ds.Letters)
        >>> at["Count"] = at.count()
        >>> at["Sum Ints"] = at.sum(ds.Ints)
        >>> at["Mean Double"] = at.mean(ds.Ints * ds.Twos)
        >>> at
        Inner Tables: ['Count', 'Sum Ints', 'Mean Double']
        Margin Columns: ['Count', 'Sum Ints', 'Mean Double']
        Footer Rows: ['Count', 'Sum Ints', 'Mean Double']

        Generate a table from this :py:class:`~.rt_accumtable.AccumTable` using default
        parameter values:

        >>> at.gen()
        *Groups          A      B      C   Mean Double   Count   Sum Ints
        -----------   ----   ----   ----   -----------   -----   --------
        Group1        3.00    nan   4.00          3.33       3          5
        Group2         nan   2.00   8.00          5.00       2          5
        -----------   ----   ----   ----   -----------   -----   --------
        Mean Double   3.00   2.00   6.00          4.00
              Count      2      1      2                     5
           Sum Ints      3      1      6                               10
        <BLANKLINE>
        [2 rows x 7 columns] total bytes: 108.0 B

        Without specifying ``table_name``, the last-created inner table, Mean Double,
        appears as the generated inner table and the first footer row and margin column.

        Pass an inner table name to generate a specific table:

        >>> at.gen("Sum Ints")
        *Groups          A      B      C   Sum Ints   Count   Mean Double
        -----------   ----   ----   ----   --------   -----   -----------
        Group1           3      0      2          5       3          3.33
        Group2           0      1      4          5       2          5.00
        -----------   ----   ----   ----   --------   -----   -----------
           Sum Ints      3      1      6         10
              Count      2      1      2                  5
        Mean Double   3.00   2.00   6.00                             4.00
        <BLANKLINE>
        [2 rows x 7 columns] total bytes: 108.0 B

        """
        # Get the displayed, inner table
        table_name = self._default_inner_name if table_name is None else table_name
        self._default_inner_name = table_name
        if table_name is None:
            raise ValueError("Must specify a table name")
        orig = self._inner[table_name]

        # Remove blanks, as required, and set the row filter
        if remove_blanks:
            (clean, row_filter, _) = orig.copy().trim(ret_filters=True)
            row_filter = row_filter if row_filter is not None else slice(None, None, None)
        else:
            clean = orig.copy()
            row_filter = slice(None, None, None)

        # Add the margin columns to the right
        summary_names = clean.summary_get_names()
        for mar_col in [col for col in list(self._cols.keys()) if col != table_name]:
            clean[mar_col] = self._inner[mar_col][row_filter, mar_col]
            summary_names += [mar_col]
        clean.summary_set_names(summary_names)

        # Add the footer rows at the bottom
        for footer_row in [row for row in list(self._rows.keys()) if row != table_name]:
            fd = list(self._inner[footer_row].footer_get_dict(footer_row).values())[0]
            delete = [k for k in fd.keys() if not k in clean.keys()]
            for key in delete:
                del fd[key]
            clean.footer_set_values(footer_row, fd)

        return clean

    # -------------------------------------------------------
    def set_margin_columns(self, cols):
        """
        Specify the margin columns that appear in a generated
        :py:class:`~.rt_accumtable.Accumtable`.

        Pass a list of inner table names to set the corresponding margin columns for
        the :py:class:`~.rt_accumtable.AccumTable` instance. The margin columns contain
        values calculated by a reducing function and grouped by the
        :py:class:`~.rt_accumtable.AccumTable` rows.

        When you generate a table using :py:meth:`~.rt_accumtable.AccumTable.gen`, the
        margin column corresponding to the inner table appears first. Then, the
        remaining margin columns appear in the order you passed them to
        :py:meth:`~.rt_accumtable.AccumTable.set_margin_columns`.

        Passing an empty list removes all margin columns from the generated table, except
        for the margin column corresponding to the inner table.

        Parameters
        ----------
        cols : list of str
            A list of inner table names, in the order you want the margin columns to
            appear in a generated table.

        Returns
        -------
        None
            Returns nothing.

        See Also
        --------
        :py:class:`.rt_accumtable.AccumTable` :
            The class containing :py:meth:`~.rt_accumtable.AccumTable.set_margin_columns`.
        :py:meth:`.rt_accumtable.AccumTable.gen` :
            The method that generates a table from an :py:class:`.rt_accumtable.AccumTable`.
        :py:meth:`.rt_accumtable.AccumTable.set_footer_rows` :
            The method that sets the footer rows for the :py:class:`.rt_accumtable.AccumTable`
            and its generated tables.

        Examples
        --------
        Construct a :py:class:`~.rt_dataset.Dataset` for the following examples:

        >>> ds = rt.Dataset()
        >>> ds.Zeros = [0, 0, 0, 0, 0]
        >>> ds.Ones = [1, 1, 1, 1, 1]
        >>> ds.Twos = [2, 2, 2, 2, 2]
        >>> ds.Nans = [rt.nan, rt.nan, rt.nan, rt.nan, rt.nan]
        >>> ds.Ints = [0, 1, 2, 3, 4]
        >>> ds.Groups = rt.Cat(["Group1", "Group2", "Group1", "Group1", "Group2"])
        >>> ds.Letters = rt.Cat(["A", "B", "C", "A", "C"])
        >>> ds
        #   Zeros   Ones   Twos   Nans   Ints   Groups   Letters
        -   -----   ----   ----   ----   ----   ------   -------
        0       0      1      2    nan      0   Group1   A
        1       0      1      2    nan      1   Group2   B
        2       0      1      2    nan      2   Group1   C
        3       0      1      2    nan      3   Group1   A
        4       0      1      2    nan      4   Group2   C
        <BLANKLINE>
        [5 rows x 7 columns] total bytes: 225.0 B

        Construct an :py:class:`~.rt_accumtable.AccumTable` from that data:

        >>> at = rt.AccumTable(ds.Groups, ds.Letters)
        >>> at["Count"] = at.count()
        >>> at["Sum Ints"] = at.sum(ds.Ints)
        >>> at["Mean Double"] = at.mean(ds.Ints * ds.Twos)
        >>> at["Variance Ints"] = at.var(ds.Ints)
        >>> at
        Inner Tables: ['Count', 'Sum Ints', 'Mean Double', 'Variance Ints']
        Margin Columns: ['Count', 'Sum Ints', 'Mean Double', 'Variance Ints']
        Footer Rows: ['Count', 'Sum Ints', 'Mean Double', 'Variance Ints']

        When you generate a table from the :py:class:`~.rt_accumtable.AccumTable`
        without setting the margin columns, all margin columns appear in the generated
        table:

        >>> at.gen("Sum Ints")
        *Groups            A      B      C   Sum Ints   Count   Mean Double   Variance Ints
        -------------   ----   ----   ----   --------   -----   -----------   -------------
        Group1             3      0      2          5       3          3.33            2.33
        Group2             0      1      4          5       2          5.00            4.50
        -------------   ----   ----   ----   --------   -----   -----------   -------------
             Sum Ints      3      1      6         10
                Count      2      1      2                  5
          Mean Double   3.00   2.00   6.00                             4.00
        Variance Ints   4.50    nan   2.00                                             2.00
        <BLANKLINE>
        [2 rows x 8 columns] total bytes: 124.0 B

        Pass a list of inner table names from the :py:class:`~.rt_accumtable.AccumTable`
        to set the corresponding margin columns in a generated table:

        >>> at.set_margin_columns(["Variance Ints", "Count"])
        >>> at
        Inner Tables: ['Count', 'Sum Ints', 'Mean Double', 'Variance Ints']
        Margin Columns: ['Variance Ints', 'Count']
        Footer Rows: ['Count', 'Sum Ints', 'Mean Double', 'Variance Ints']

        Generate a table to see the new set of margin columns:

        >>> at.gen("Sum Ints")
        *Groups            A      B      C   Sum Ints   Variance Ints   Count
        -------------   ----   ----   ----   --------   -------------   -----
        Group1             3      0      2          5            2.33       3
        Group2             0      1      4          5            4.50       2
        -------------   ----   ----   ----   --------   -------------   -----
             Sum Ints      3      1      6         10
                Count      2      1      2                                  5
          Mean Double   3.00   2.00   6.00
        Variance Ints   4.50    nan   2.00                       2.00
        <BLANKLINE>
        [2 rows x 7 columns] total bytes: 108.0 B

        Pass an empty list to remove all margin columns from the generated table, except
        for the margin column corresponding to the inner table. In this example, the
        Sum Ints margin column remains in the generated table:

        >>> at.set_margin_columns([])
        >>> at
        Inner Tables: ['Count', 'Sum Ints', 'Mean Double', 'Variance Ints']
        Margin Columns: []
        Footer Rows: ['Count', 'Sum Ints', 'Mean Double', 'Variance Ints']
        >>> at.gen("Sum Ints")
        *Groups            A      B      C   Sum Ints
        -------------   ----   ----   ----   --------
        Group1             3      0      2          5
        Group2             0      1      4          5
        -------------   ----   ----   ----   --------
             Sum Ints      3      1      6         10
                Count      2      1      2
          Mean Double   3.00   2.00   6.00
        Variance Ints   4.50    nan   2.00
        <BLANKLINE>
        [2 rows x 5 columns] total bytes: 76.0 B
        """
        self._cols = OrderedDict()
        for k in cols:
            self._cols[k] = None

    # -------------------------------------------------------
    def set_footer_rows(self, rows):
        """
        Specify the footer rows that appear in a generated
        :py:class:`~.rt_accumtable.Accumtable`.

        Pass a list of inner table names to set the corresponding footer rows for
        the :py:class:`~.rt_accumtable.AccumTable` instance. The footer rows contain
        values calculated by a reducing function and grouped by the
        :py:class:`~.rt_accumtable.AccumTable` columns.

        When you generate a table using :py:meth:`~.rt_accumtable.AccumTable.gen`, the
        footer row corresponding to the inner table appears first. Then, the remaining
        footer rows appear in the order you passed them to
        :py:meth:`~.rt_accumtable.AccumTable.set_margin_columns`.

        Passing an empty list removes all footer rows from the generated table, except
        for the footer row corresponding to the inner table.

        Parameters
        ----------
        rows : list
            A list of inner table names, in the order you want the footer rows to
            appear in a generated table.

        Returns
        -------
        None
            Returns nothing.

        See Also
        --------
        :py:class:`.rt_accumtable.AccumTable` :
            The class containing :py:meth:`~.rt_accumtable.AccumTable.set_footer_rows`.
        :py:meth:`.rt_accumtable.AccumTable.gen` :
            The method that generates a table from an :py:class:`.rt_accumtable.AccumTable`.
        :py:meth:`.rt_accumtable.AccumTable.set_margin_columns` :
            The method that sets the margin columns for the
            :py:class:`.rt_accumtable.AccumTable` and its generated tables.

        Examples
        --------
        Construct a :py:class:`~.rt_dataset.Dataset` for the following examples:

        >>> ds = rt.Dataset()
        >>> ds.Zeros = [0, 0, 0, 0, 0]
        >>> ds.Ones = [1, 1, 1, 1, 1]
        >>> ds.Twos = [2, 2, 2, 2, 2]
        >>> ds.Nans = [rt.nan, rt.nan, rt.nan, rt.nan, rt.nan]
        >>> ds.Ints = [0, 1, 2, 3, 4]
        >>> ds.Groups = rt.Cat(["Group1", "Group2", "Group1", "Group1", "Group2"])
        >>> ds.Letters = rt.Cat(["A", "B", "C", "A", "C"])
        >>> ds
        #   Zeros   Ones   Twos   Nans   Ints   Groups   Letters
        -   -----   ----   ----   ----   ----   ------   -------
        0       0      1      2    nan      0   Group1   A
        1       0      1      2    nan      1   Group2   B
        2       0      1      2    nan      2   Group1   C
        3       0      1      2    nan      3   Group1   A
        4       0      1      2    nan      4   Group2   C
        <BLANKLINE>
        [5 rows x 7 columns] total bytes: 225.0 B

        Construct an :py:class:`~.rt_accumtable.AccumTable` from that data:

        >>> at = rt.AccumTable(ds.Groups, ds.Letters)
        >>> at["Count"] = at.count()
        >>> at["Sum Ints"] = at.sum(ds.Ints)
        >>> at["Mean Double"] = at.mean(ds.Ints * ds.Twos)
        >>> at["Variance Ints"] = at.var(ds.Ints)
        >>> at
        Inner Tables: ['Count', 'Sum Ints', 'Mean Double', 'Variance Ints']
        Margin Columns: ['Count', 'Sum Ints', 'Mean Double', 'Variance Ints']
        Footer Rows: ['Count', 'Sum Ints', 'Mean Double', 'Variance Ints']

        When you generate a table from the :py:class:`~.rt_accumtable.AccumTable`
        without setting the footer rows, all footer rows appear in the generated
        table:

        >>> at.gen("Sum Ints")
        *Groups            A      B      C   Sum Ints   Count   Mean Double   Variance Ints
        -------------   ----   ----   ----   --------   -----   -----------   -------------
        Group1             3      0      2          5       3          3.33            2.33
        Group2             0      1      4          5       2          5.00            4.50
        -------------   ----   ----   ----   --------   -----   -----------   -------------
             Sum Ints      3      1      6         10
                Count      2      1      2                  5
          Mean Double   3.00   2.00   6.00                             4.00
        Variance Ints   4.50    nan   2.00                                             2.00
        <BLANKLINE>
        [2 rows x 8 columns] total bytes: 124.0 B

        Pass a list of inner table names from the :py:class:`~.rt_accumtable.AccumTable`
        to set the corresponding footer rows in a generated table:

        >>> at.set_footer_rows(["Variance Ints", "Count"])
        >>> at
        Inner Tables: ['Count', 'Sum Ints', 'Mean Double', 'Variance Ints']
        Margin Columns: ['Count', 'Sum Ints', 'Mean Double', 'Variance Ints']
        Footer Rows: ['Variance Ints', 'Count']

        Generate a table to see the new set of footer rows:

        >>> at.gen("Sum Ints")
        *Groups            A     B      C   Sum Ints   Count   Mean Double   Variance Ints
        -------------   ----   ---   ----   --------   -----   -----------   -------------
        Group1             3     0      2          5       3          3.33            2.33
        Group2             0     1      4          5       2          5.00            4.50
        -------------   ----   ---   ----   --------   -----   -----------   -------------
             Sum Ints      3     1      6         10
        Variance Ints   4.50   nan   2.00                                             2.00
                Count      2     1      2                  5
        <BLANKLINE>
        [2 rows x 8 columns] total bytes: 124.0 B

        Pass an empty list to remove all footer rows from the generated table, except
        for the footer row corresponding to the inner table. In this example, the
        Sum Ints footer row remains in the generated table:

        >>> at.set_footer_rows([])
        >>> at
        Inner Tables: ['Count', 'Sum Ints', 'Mean Double', 'Variance Ints']
        Margin Columns: ['Count', 'Sum Ints', 'Mean Double', 'Variance Ints']
        Footer Rows: []
        >>> at.gen("Sum Ints")
        *Groups    A   B   C   Sum Ints   Count   Mean Double   Variance Ints
        --------   -   -   -   --------   -----   -----------   -------------
        Group1     3   0   2          5       3          3.33            2.33
        Group2     0   1   4          5       2          5.00            4.50
        --------   -   -   -   --------   -----   -----------   -------------
        Sum Ints   3   1   6         10
        <BLANKLINE>
        [2 rows x 8 columns] total bytes: 124.0 B
        """
        self._rows = OrderedDict()
        for k in rows:
            self._rows[k] = None


def accum_ratio(
    cat1,
    cat2=None,
    val1=None,
    val2=None,
    filt1=None,
    filt2=None,
    func1="nansum",
    func2=None,
    return_table=False,
    include_numer=False,
    include_denom=True,
    remove_blanks=False,
):
    """
    Generate a :py:class:`~.rt_dataset.Dataset` of ratios between values calculated by
    reducing functions for two arrays.

    :py:func:`~.rt_accumtable.accum_ratio` performs the following actions:

    #. Creates an :py:class:`~.rt_accumtable.AccumTable` using the groups of ``cat1``
       and ``cat2``.
    #. Aggregates the data from the ``val1`` and ``val2`` arrays according to the
       ``func1`` and ``func2`` reducing functions.
    #. Calculates a ratio between the values calculated by the reducing functions for
       ``val1`` and ``val2``.
    #. Returns either a :py:class:`~.rt_dataset.Dataset` or an
       :py:class:`~.rt_accumtable.AccumTable`, depending on the value of
       ``return_table``.

    By default, :py:func:`~.rt_accumtable.accum_ratio` returns a
    :py:class:`~.rt_dataset.Dataset` with a ``"Ratio"`` inner table. If ``return_table``
    is set to `True`, the function returns an :py:class:`~.rt_accumtable.AccumTable`,
    which can be converted to a :py:class:`~.rt_dataset.Dataset` using the
    :py:meth:`~.rt_accumtable.AccumTable.gen` method. Generating a
    :py:class:`~.rt_dataset.Dataset` gives you more control over which inner table,
    footer rows, and margin columns are included in the result.

    :py:func:`~.rt_accumtable.accum_ratio` supports only reducing functions that take an
    array as a parameter. For example ``count()`` isn't valid, as it doesn't accept an
    array as an input argument. For a list of reducing functions, see
    :doc:`/tutorial/tutorial_cat_reduce`.

    Parameters
    ----------
    cat1 : :py:class:`~.rt_categorical.Categorical`
        The row groups used to accumulate the values.
    cat2 : :py:class:`~.rt_categorical.Categorical`, optional
        The column groups used to accumulate the values. If not
        provided, :py:func:`~.rt_accumtable.accum_ratio` uses a
        :py:class:`~.rt_categorical.Categorical` with a single group, ``"NotGrouped"``.
    val1 : array
        The numerator for the calculated ratio.
    val2 : array
        The denominator for the calculated ratio.
    filt1 : array of bool, optional
        Boolean filter for ``val1`` array. The filter array must be the same length as
        ``val1`` and ``val2``.
    filt2 : array of bool, optional
        Boolean filter for ``val2`` array. The filter array must be the same length as
        ``val1`` and ``val2``. If not provided, the filter is the same as ``filt1``.
    func1 : str, default ``"nansum"``
        String of the name of the reducing function (for example, ``"sum"`` or
        ``"nanmean"``) used to reduce ``val1`` before calculating the ratio.
    func2 : str, optional
        String of the name of the reducing function (for example, ``"sum"`` or
        ``"nanmean"``) used to reduce ``val2`` before calculating the ratio. If not
        provided, the ``func1`` is applied to ``val2``.
    return_table : bool, default `False`
        If `False` (the default), returns a :py:class:`~.rt_dataset.Dataset` with the
        calculated ratio. If set to `True`, returns an
        :py:class:`~.rt_accumtable.AccumTable` from which you can generate a
        :py:class:`~.rt_dataset.Dataset`. The returned
        :py:class:`~.rt_accumtable.AccumTable` has ``"Numer"``, ``"Denom"``, and
        ``"Ratio"`` inner tables, footer rows, and margin columns.
    include_numer : bool, default `False`
        If set to `True`, include the values calculated by the reducing function for
        ``val1`` as a row and column in the returned table. Ignored if ``return_table``
        is `True`.
    include_denom : bool, default `True`
        If `True` (the default), include the values calculated by the reducing function
        for ``val2`` as a row and column in the returned table. Ignored if
        ``return_table`` is `True`.
    remove_blanks : bool, default `False`
        If set to `True`, removes rows and columns that consist entirely of ``0`` or
        ``nan`` from the returned table.

    Returns
    -------
    :py:class:`.rt_dataset.Dataset` or :py:class:`.rt_accumtable.AccumTable`
        Either a :py:class:`~.rt_dataset.Dataset` with a view of the calculated ratio,
        or an :py:class:`~.rt_accumtable.AccumTable`, depending on ``return_table``.

    See Also
    --------
    :py:class:`.rt_accum2.Accum2` :
        The parent class for :py:class:`~.rt_accumtable.AccumTable`.
    :py:class:`.rt_accumtable.AccumTable` :
        A wrapper on :py:class:`~.rt_accum2.Accum2` that enables the creation of tables
        that combine the results of multiple tables generated from the
        :py:class:`~.rt_accum2.Accum2` object.
    :py:class:`.rt_categorical.Categorical` :
        A class that efficiently stores an array of repeated strings and is used for
        groupby operations.
    :py:class:`.rt_groupbyops.GroupByOps` :
        A class that holds the reducing functions used by
        :py:func:`~.rt_accumtable.accum_ratio`.

    Examples
    --------
    Construct a :py:class:`~.rt_dataset.Dataset` for the following examples:

    >>> ds = rt.Dataset()
    >>> ds.Zeros = [0, 0, 0, 0, 0]
    >>> ds.Ones = [1, 1, 1, 1, 1]
    >>> ds.Twos = [2, 2, 2, 2, 2]
    >>> ds.Nans = [rt.nan, rt.nan, rt.nan, rt.nan, rt.nan]
    >>> ds.Ints = [0, 1, 2, 3, 4]
    >>> ds.Groups = rt.Cat(["Group1", "Group2", "Group1", "Group1", "Group2"])
    >>> ds.Letters = rt.Cat(["A", "B", "C", "A", "C"])
    >>> ds
    #   Zeros   Ones   Twos   Nans   Ints   Groups   Letters
    -   -----   ----   ----   ----   ----   ------   -------
    0       0      1      2    nan      0   Group1   A
    1       0      1      2    nan      1   Group2   B
    2       0      1      2    nan      2   Group1   C
    3       0      1      2    nan      3   Group1   A
    4       0      1      2    nan      4   Group2   C
    <BLANKLINE>
    [5 rows x 7 columns] total bytes: 225.0 B

    **Calculate a ratio between the values calculated by a reducing function**

    This example returns a :py:class:`~.rt_dataset.Dataset` that holds ratios between
    the values calculated by the default reducing function
    (:py:meth:`~.rt_groupbyops.GroupByOps.nansum`) for Ints and Ones.

    >>> rt.accum_ratio(cat1=ds.Groups,
    ...                cat2=ds.Letters,
    ...                val1=ds.Ints,
    ...                val2=ds.Ones)
    *Groups      A      B      C   Ratio   Denom
    -------   ----   ----   ----   -----   -----
    Group1    1.50    nan   2.00    1.67       3
    Group2     nan   1.00   4.00    2.50       2
    -------   ----   ----   ----   -----   -----
      Ratio   1.50   1.00   3.00    2.00
      Denom      2      1      2               5
    <BLANKLINE>
    [2 rows x 6 columns] total bytes: 92.0 B

    **Return an AccumTable**

    Pass `True` to ``return_table`` to return an :py:class:`~.rt_accumtable.AccumTable`
    instead of a :py:class:`~.rt_dataset.Dataset`:

    >>> returned_accumtable = rt.accum_ratio(cat1=ds.Groups,
    ...                                      cat2=ds.Letters,
    ...                                      val1=ds.Ints,
    ...                                      val2=ds.Ones,
    ...                                      func1="nansum",
    ...                                      return_table=True)
    >>> returned_accumtable
    Inner Tables: ['Numer', 'Denom', 'Ratio']
    Margin Columns: ['Numer', 'Denom', 'Ratio']
    Footer Rows: ['Numer', 'Denom', 'Ratio']

    Use :py:meth:`~.rt_accumtable.AccumTable.gen` to create a
    :py:class:`~.rt_dataset.Dataset` from the returned
    :py:class:`~.rt_accumtable.AccumTable`:

    >>> returned_accumtable.gen()
    *Groups      A      B      C   Ratio   Numer   Denom
    -------   ----   ----   ----   -----   -----   -----
    Group1    1.50    nan   2.00    1.67       5       3
    Group2     nan   1.00   4.00    2.50       5       2
    -------   ----   ----   ----   -----   -----   -----
      Ratio   1.50   1.00   3.00    2.00
      Numer      3      1      6              10
      Denom      2      1      2                       5
    <BLANKLINE>
    [2 rows x 7 columns] total bytes: 108.0 B

    **Filter the arrays before calculating ratios**

    Pass filters to ``filt1`` and ``filt2`` to filter ``val1`` and ``val2`` before
    reducing and ratio calculation:

    >>> c_filter = ds.Letters == "C"
    >>> even_filter = ds.Ints % 2 == 0
    >>> rt.accum_ratio(cat1=ds.Groups,
    ...                cat2=ds.Letters,
    ...                val1=ds.Ints,
    ...                val2=ds.Ones,
    ...                func1="nansum",
    ...                filt1=c_filter,
    ...                filt2=even_filter)
    *Groups      A     B      C   Ratio   Denom
    -------   ----   ---   ----   -----   -----
    Group1    0.00   nan   2.00    1.00       2
    Group2     nan   nan   4.00    4.00       1
    -------   ----   ---   ----   -----   -----
      Ratio   0.00   nan   3.00    2.00
      Denom      1     0      2               3
    <BLANKLINE>
    [2 rows x 6 columns] total bytes: 92.0 B

    **Remove blank rows and columns**

    Pass `True` to ``remove_blanks`` to remove the rows and columns consisting entirely
    of ``0`` or ``nan``. This example removes the blank lines from the filtered
    :py:class:`~.rt_dataset.Dataset`:

    >>> c_filter = ds.Letters == "C"
    >>> even_filter = ds.Ints % 2 == 0
    >>> rt.accum_ratio(cat1=ds.Groups,
    ...                cat2=ds.Letters,
    ...                val1=ds.Ints,
    ...                val2=ds.Ones,
    ...                func1="nansum",
    ...                filt1=c_filter,
    ...                filt2=even_filter,
    ...                remove_blanks=True)
    *Groups      C   Ratio   Denom
    -------   ----   -----   -----
    Group1    2.00    1.00       2
    Group2    4.00    4.00       1
    -------   ----   -----   -----
      Ratio   3.00    2.00
      Denom      2               3
    <BLANKLINE>
    [2 rows x 4 columns] total bytes: 60.0 B

    **Include non-ratio values calculated by reducing functions**

    Pass `True` to ``include_numer`` and ``include_denom`` to add summary rows and
    columns with the non-ratio values calculated by the reducing functions. Numer
    contains values for ``val1`` calculated with ``func1``. Denom contains values for
    ``val2`` calculated with ``func2``. This example doesn't include ``func2``, so
    :py:func:`~.rt_accumtable.accum_ratio` uses ``func1`` for ``val2``.

    >>> rt.accum_ratio(cat1=ds.Groups,
    ...                cat2=ds.Letters,
    ...                val1=ds.Ints,
    ...                val2=ds.Ones,
    ...                func1="nansum",
    ...                include_numer=True,
    ...                include_denom=True)
    *Groups      A      B      C   Ratio   Numer   Denom
    -------   ----   ----   ----   -----   -----   -----
    Group1    1.50    nan   2.00    1.67       5       3
    Group2     nan   1.00   4.00    2.50       5       2
    -------   ----   ----   ----   -----   -----   -----
      Ratio   1.50   1.00   3.00    2.00
      Numer      3      1      6              10
      Denom      2      1      2                       5
    <BLANKLINE>
    [2 rows x 7 columns] total bytes: 108.0 B
    """
    # Handle missing inputs
    if val1 is None:
        raise ValueError("Missing argument val1")
    if (
        (val2 is None) & (cat2 is not None) & (val1 is not None)
    ):  # Passing as accum_ratio(cat1, val1, val2), omitting cat2 argument
        val2 = val1
        val1 = cat2
        cat2 = None
    if filt1 is None:
        filt1 = full(val1.shape[0], True, dtype=bool)
    if filt2 is None:
        filt2 = filt1
    if func2 is None:
        func2 = func1
    if cat2 is None:
        cat2 = Categorical(full(val1.shape[0], 1, dtype=np.int8), ["NotGrouped"])

    # Handle name collisions
    for key in ["Numer", "Denom", "Ratio"]:
        if key in cat2.categories():
            cat2.category_replace(key, key + "_")

    # Compute accum
    accum = AccumTable(cat1, cat2)

    func1 = getattr(accum, func1)
    func2 = getattr(accum, func2)
    # TODO: In the future, when arbitrary functions are allowed in Accum2 calls, handle a missing attr here by passing it in by name
    accum["Numer"] = func1(val1, filter=filt1)
    accum["Denom"] = func2(val2, filter=filt2)

    accum["Ratio"] = accum["Numer"] / accum["Denom"]

    if return_table:
        return accum
    else:
        footers = [label for (label, boolean) in zip(["Numer", "Denom"], [include_numer, include_denom]) if boolean]
        accum.set_margin_columns(footers)
        accum.set_footer_rows(footers)
        return accum.gen("Ratio", remove_blanks=remove_blanks)


def accum_ratiop(
    cat1,
    cat2=None,
    val=None,
    filter=None,
    func="nansum",
    norm_by="T",
    include_total=True,
    remove_blanks=False,
    filt=None,
):
    """
    Generate a :py:class:`~.rt_dataset.Dataset` of ratios displayed as percentages
    between the individual values of a table calculated with a reducing function and
    the value of the entire :py:class:`~.rt_accumtable.AccumTable`, its rows,
    or its columns calculated with the same reducing function.

    :py:func:`~.rt_accumtable.accum_ratiop` performs the following actions:

    * Creates an :py:class:`~rt_accumtable.Accumtable` using the groups of ``cat1`` and
      ``cat2``.
    * Aggregates the data from the ``val`` array according to the ``func`` reducing
      function.
    * Calculates a ratio as a percent for each cell in the inner table, footer row, and
      margin column. The numerator of each ratio is the calculated value for the cell,
      and the denominator is the calculated value for that row, that column, or the
      table, depending on the value of ``norm_by``.
    * Generates and returns a :py:class:`~.rt_dataset.Dataset` from the
      :py:class:`~.rt_accumtable.AccumTable` with percentile values.

    :py:func:`~.rt_accumtable.accum_ratiop` supports only reducing functions that take an
    array as a parameter. For example, ``count()`` isn't valid, as it doesn't accept an
    array as an input argument. For a list of reducing functions, see
    :doc:`/tutorial/tutorial_cat_reduce`.

    Parameters
    ----------
    cat1 : :py:class:`~.rt_categorical.Categorical`
        The row groups used for accumulation.
    cat2 : :py:class:`~.rt_categorical.Categorical`, optional
        The column groups used for accumulation. If not provided,
        :py:func:`~.rt_accumtable.accum_ratiop` uses a
        :py:class:`~.rt_categorical.Categorical` with a single group, ``"NotGrouped"``.
    val : array
        The array used as the numerator for percentile calculation.
    filter : array of bool, optional
        Filter for ``val``. The ``filter`` array must be the same length as ``val``.
        Replaces the deprecated ``filt`` parameter.
    func : str
        String of the name of the reducing function used to reduce ``val`` before
        calculating the percentile.
    norm_by : {"T", "C", "R"}, default "T"
        Controls the values used as the denominator for the ratio calculation:

        * "T" selects the calculated value for the entire
          :py:class:`~.rt_accumtable.AccumTable`.
        * "C" selects the calculated value for each column.
        * "R" selects the calculated value for each row.
    include_total : bool, default `True`
        Adds a summary row and column of values calculated by ``func`` to the returned
        :py:class:`~.rt_dataset.Dataset`.
    remove_blanks : bool, default `True`
        If `True`, removes rows and columns that consist entirely of ``0`` or
        ``nan`` from the returned table.
    filt : array of bool, optional
        Deprecated and replaced with ``filter``.

    Returns
    -------
    :py:class:`~.rt_dataset.Dataset`
        A table of percent ratios for the array.

    See Also
    --------
    :py:class:`.rt_accum2.Accum2` :
        The parent class for :py:class:`~.rt_accumtable.AccumTable`.
    :py:class:`.rt_accumtable.AccumTable` :
        A wrapper on :py:class:`~.rt_accum2.Accum2` that enables the creation of tables
        that combine the results of multiple tables generated from the
        :py:class:`~.rt_accum2.Accum2` object.
    :py:class:`.rt_categorical.Categorical` :
        A class that efficiently stores an array of repeated strings and is used for
        groupby operations.
    :py:class:`.rt_groupbyops.GroupByOps` :
        A class that holds the reducing functions used by
        :py:func:`~.rt_accumtable.accum_ratiop`.

    Examples
    --------
    Construct a :py:class:`~.rt_dataset.Dataset` for the following examples:

    >>> ds = rt.Dataset()
    >>> ds.Zeros = [0, 0, 0, 0, 0]
    >>> ds.Ones = [1, 1, 1, 1, 1]
    >>> ds.Twos = [2, 2, 2, 2, 2]
    >>> ds.Ints = [0, 1, 2, 3, 4]
    >>> ds.Groups = rt.Cat(["Group1", "Group2", "Group1", "Group1", "Group2"])
    >>> ds.Letters = rt.Cat(["A", "B", "C", "A", "C"])
    >>> ds
    #   Zeros   Ones   Twos   Ints   Groups   Letters
    -   -----   ----   ----   ----   ------   -------
    0       0      1      2      0   Group1   A
    1       0      1      2      1   Group2   B
    2       0      1      2      2   Group1   C
    3       0      1      2      3   Group1   A
    4       0      1      2      4   Group2   C
    <BLANKLINE>
    [5 rows x 6 columns] total bytes: 185.0 B

    **Calculate percentiles compared to total**

    >>> rt.accum_ratiop(cat1=ds.Groups,
    ...                 cat2=ds.Letters,
    ...                 val=ds.Ints)
    *Groups          A       B       C   TotalRatio   Total
    ----------   -----   -----   -----   ----------   -----
    Group1       30.00    0.00   20.00        50.00       5
    Group2        0.00   10.00   40.00        50.00       5
    ----------   -----   -----   -----   ----------   -----
    TotalRatio   30.00   10.00   60.00       100.00
         Total       3       1       6                   10
    <BLANKLINE>
    [2 rows x 6 columns] total bytes: 92.0 B

    Pass ``"nanmean"`` to ``func`` to calculate the ratio as a percent between the
    mean for each inner table cell and the total mean:

    >>> rt.accum_ratiop(cat1=ds.Groups,
    ...                 cat2=ds.Letters,
    ...                 val=ds.Ints,
    ...                 func="nanmean")
    *Groups          A       B        C   TotalRatio   Total
    ----------   -----   -----   ------   ----------   -----
    Group1       75.00     nan   100.00        83.33    1.67
    Group2         nan   50.00   200.00       125.00    2.50
    ----------   -----   -----   ------   ----------   -----
    TotalRatio   75.00   50.00   150.00       100.00
         Total    1.50    1.00     3.00                 2.00
    <BLANKLINE>
    [2 rows x 6 columns] total bytes: 92.0 B

    **Calculate percentiles compared to row**

    Pass ``"R"`` to ``norm_by``:

    >>> rt.accum_ratiop(cat1=ds.Groups,
    ...                 cat2=ds.Letters,
    ...                 val=ds.Ints,
    ...                 func="nanmean",
    ...                 norm_by="R",
    ...                 include_total=False)
    *Groups          A       B        C   TotalRatio
    ----------   -----   -----   ------   ----------
    Group1       90.00     nan   120.00       100.00
    Group2         nan   40.00   160.00       100.00
    ----------   -----   -----   ------   ----------
    TotalRatio   75.00   50.00   150.00       100.00
    <BLANKLINE>
    [2 rows x 5 columns] total bytes: 76.0 B

    **Calculate percentiles compared to column**

    Pass ``"C"`` to ``norm_by``:

    >>> rt.accum_ratiop(cat1=ds.Groups,
    ...                 cat2=ds.Letters,
    ...                 val=ds.Ints,
    ...                 func="nanmean",
    ...                 norm_by="C",
    ...                 include_total=False)
    *Groups           A        B        C   TotalRatio
    ----------   ------   ------   ------   ----------
    Group1       100.00      nan    66.67        83.33
    Group2          nan   100.00   133.33       125.00
    ----------   ------   ------   ------   ----------
    TotalRatio   100.00   100.00   100.00       100.00
    <BLANKLINE>
    [2 rows x 5 columns] total bytes: 76.0 B

    **Filter the array before calculating percentiles**

    Create a filter for ``val`` and pass it to ``filter``. This example selects for data
    in ``val`` in the ``"carrot"`` group:

    >>> c_filter = ds.Letters == "C"
    >>> rt.accum_ratiop(cat1=ds.Groups,
    ...                 cat2=ds.Letters,
    ...                 val=ds.Ints,
    ...                 filter=c_filter,
    ...                 func="nansum",
    ...                 include_total=False)
    *Groups         A      B        C   TotalRatio
    ----------   ----   ----   ------   ----------
    Group1       0.00   0.00    33.33        33.33
    Group2       0.00   0.00    66.67        66.67
    ----------   ----   ----   ------   ----------
    TotalRatio   0.00   0.00   100.00       100.00
    <BLANKLINE>
    [2 rows x 5 columns] total bytes: 76.0 B

    **Remove blank rows and columns**

    Pass `True` to ``remove_blanks`` to remove the rows and columns consisting entirely
    of ``0`` or ``nan``. This example removes the blank lines from the filtered
    :py:class:`~.rt_dataset.Dataset`:

    >>> rt.accum_ratiop(cat1=ds.Groups,
    ...                 cat2=ds.Letters,
    ...                 val=ds.Ints,
    ...                 filter=c_filter,
    ...                 func="nansum",
    ...                 include_total=False,
    ...                 remove_blanks=True)
    *Groups           C   TotalRatio
    ----------   ------   ----------
    Group1        33.33        33.33
    Group2        66.67        66.67
    ----------   ------   ----------
    TotalRatio   100.00       100.00
    <BLANKLINE>
    [2 rows x 3 columns] total bytes: 44.0 B

    **Include the total values calculated by reducing functions**

    Pass `True` to ``include_total`` to add a ``"Total"`` row and column to the returned
    :py:class:`~.rt_dataset.Dataset`. The total represents the values calculated by the
    reducing function before percentile calculation.

    >>> rt.accum_ratiop(cat1=ds.Groups,
    ...                 cat2=ds.Letters,
    ...                 val=ds.Ints,
    ...                 include_total=True)
    *Groups          A       B       C   TotalRatio   Total
    ----------   -----   -----   -----   ----------   -----
    Group1       30.00    0.00   20.00        50.00       5
    Group2        0.00   10.00   40.00        50.00       5
    ----------   -----   -----   -----   ----------   -----
    TotalRatio   30.00   10.00   60.00       100.00
         Total       3       1       6                   10
    <BLANKLINE>
    [2 rows x 6 columns] total bytes: 92.0 B
    """
    # Handle missing inputs
    if val is None:
        val = full(cat1.shape[0], 1, dtype=np.float64)
    if filter is None:
        if filt is not None:  # Temporary until deprecated
            warnings.warn(
                'Kwarg "filt" is being deprecated for "filter" to align with common syntax. "filt" will be removed in a future version',
                FutureWarning,
                stacklevel=2,
            )
            filter = filt
        else:
            filter = full(val.shape[0], True, dtype=bool)
    if cat2 is None:
        cat2 = Categorical(full(val.shape[0], 1, dtype=np.int8), ["NotGrouped"])

    # Compute accum
    accum = AccumTable(cat1, cat2)

    func_name = func
    func = getattr(accum, func_name)
    # TODO: In the future, when arbitrary functions are allowed in Accum2 calls, handle a missing attr here by passing it in by name
    accum["TotalRatio"] = func(val, filter=filter)
    if include_total:
        accum["Total"] = func(val, filter=filter)

    accumr = accum.gen("TotalRatio", remove_blanks=remove_blanks)
    keys = [
        key
        for key in accumr.keys()
        if key not in set(accumr.label_get_names() + accumr.summary_get_names()) - set(['TotalRatio'])
    ]

    if norm_by.upper() == "T":
        total = accumr.footer_get_dict()["TotalRatio"]["TotalRatio"]
        accumr.footer_set_values(
            "TotalRatio", {key: 100 * item / total for (key, item) in accumr.footer_get_dict()["TotalRatio"].items()}
        )
        for col in keys:
            accumr[col] = 100 * accumr[col] / total
    elif norm_by.upper() == "R":
        total = accumr.footer_get_dict()["TotalRatio"]["TotalRatio"]
        accumr.footer_set_values(
            "TotalRatio", {key: 100 * item / total for (key, item) in accumr.footer_get_dict()["TotalRatio"].items()}
        )
        for col in keys:
            accumr[col] = 100 * accumr[col] / accumr.TotalRatio
    elif norm_by.upper() == "C":
        for col in keys:
            total = accumr.footer_get_dict()["TotalRatio"][col]
            accumr[col] = 100 * accumr[col] / total
        accumr.footer_set_values(
            "TotalRatio", {key: 100.0 for (key, item) in accumr.footer_get_dict()["TotalRatio"].items()}
        )
    else:
        raise ValueError(f"Invalid norm_by selection: {norm_by}. Valid choices are T, R, C.")

    return accumr


def accum_cols(cat, val_list, name_list=None, filt_list=None, func_list="nansum", remove_blanks=False):
    """
    Apply reducing functions to multiple arrays that are grouped by a
    :py:class:`~.rt_categorical.Categorical`.

    The returned :py:class:`~.rt_dataset.Dataset` contains values calculated by a
    reducing function for each :py:class:`~.rt_categorical.Categorical` group from each
    of the arrays in ``val_list``. It also contains the calculated value for each of the
    original arrays in the ``Total`` row.

    :py:func:`~.rt_accumtable.accum_cols` supports only reducing functions that take an
    array as a parameter. For example ``count()`` isn't valid, as it doesn't accept an
    array as an input argument. For a list of reducing functions, see
    :doc:`/tutorial/tutorial_cat_reduce`.

    Parameters
    ----------
    cat : :py:class:`~.rt_categorical.Categorical`
        A :py:class:`~.rt_categorical.Categorical` that specifies the groups for
        reducing the ``val_list`` array.
    val_list : array or list of arrays
        Array or list of arrays that ``func_list`` is applied to.
        :py:func:`~.rt_accumtable.accum_cols` returns an array for each element in
        ``val_list``. If an element of ``val_list`` is itself a two-element list of two
        arrays, :py:func:`~.rt_accumtable.accum_cols` calculates a ratio between the
        values calculated by a reducing function for the two arrays.
        :py:func:`~.rt_accumtable.accum_ratio` performs this calculation using ``cat``,
        the two arrays, the respective filter, and the respective reducing function as
        arguments.

        If the second element of the two-element list is ``"p"`` or ``"P"``,
        :py:func:`~.rt_accumtable.accum_cols` calculates a ratio displayed as a
        percentage between the individual values of a table calculated with a reducing
        function and the calculated value of the entire
        :py:class:`~.rt_accumtable.AccumTable`. :py:func:`~.rt_accumtable.accum_ratiop`
        performs this calculation using ``cat``, the first element of the two element
        list, the respective filter, and the respective reducing function as arguments.
    name_list : list, optional
        List of column names in the returned :py:class:`~.rt_dataset.Dataset`. If not
        provided, the returned columns have names ``colN``.
    filt_list : array of bool or list of array of bool, optional
        Either a filter array that applies to all arrays in ``val_list`` or a list of
        filters, where each filter applies to the respective array in ``val_list``. Each
        filter must be the same length as the arrays in ``val_list``.
    func_list : str or list of str, default "nansum"
        Either a string of the name of a reducing function (for example, ``"sum"`` or
        ``"nanmean"``) or a list of strings of reducing function names. Passing a string
        applies the single reducing function to all arrays in ``val_list``. Passing a
        list of strings applies each reducing function to the respective array in
        ``val_list``. Note the following two exceptions:

        * If you pass more functions than there are arrays in ``val_list``, the extra
          functions without respective arrays in ``val_list`` are ignored.
        *  If you pass fewer functions than arrays, the returned
          :py:class:`~.rt_dataset.Dataset` contains only same number of columns as there
          are functions in ``func_list``.
    remove_blanks : bool, default `False`
        If `True`, removes rows and columns that consist entirely of ``0`` or ``nan`` from
        the returned :py:class:`~.rt_dataset.Dataset`.

    Returns
    -------
    :py:class:`.rt_dataset.Dataset`
        A table of the values calculated by the reducing functions for each element of
        ``val_list``.

    See Also
    --------
    :py:class:`.rt_accum2.Accum2` :
        The parent class for :py:class:`~.rt_accumtable.AccumTable`.
    :py:class:`.rt_accumtable.AccumTable` :
        A wrapper on :py:class:`~.rt_accum2.Accum2` that enables the creation of tables
        that combine the results of multiple tables generated from the
        :py:class:`~.rt_accum2.Accum2` object.
    :py:class:`.rt_categorical.Categorical` :
        A class that efficiently stores an array of repeated strings and is used for
        groupby operations.
    :py:class:`.rt_groupbyops.GroupByOps` :
        A class that holds the reducing functions used by
        :py:func:`~.rt_accumtable.accum_cols`.

    Examples
    --------
    Construct a :py:class:`~.rt_dataset.Dataset` for the following examples:

    >>> ds = rt.Dataset()
    >>> ds.Zeros = [0, 0, 0, 0, 0]
    >>> ds.Ones = [1, 1, 1, 1, 1]
    >>> ds.Twos = [2, 2, 2, 2, 2]
    >>> ds.Nans = [rt.nan, rt.nan, rt.nan, rt.nan, rt.nan]
    >>> ds.Ints = [0, 1, 2, 3, 4]
    >>> ds.Groups = ["Group1", "Group2", "Group1", "Group1", "Group2"]
    >>> ds.Groups = rt.Cat(ds.Groups)
    >>> ds
    #   Zeros   Ones   Twos   Nans   Ints   Groups
    -   -----   ----   ----   ----   ----   ------
    0       0      1      2    nan      0   Group1
    1       0      1      2    nan      1   Group2
    2       0      1      2    nan      2   Group1
    3       0      1      2    nan      3   Group1
    4       0      1      2    nan      4   Group2
    <BLANKLINE>
    [5 rows x 6 columns] total bytes: 217.0 B

    **Apply one reducing function to all arrays**

    Pass a single function name as a string to ``func_list``. This example applies the
    :py:meth:`~.rt_groupbyops.GroupByOps.sum` reducing function to all arrays in
    ``val_list``:

    >>> rt.accum_cols(cat=ds.Groups,
    ...               val_list=[ds.Zeros, ds.Ones, ds.Twos, ds.Nans, ds.Ints],
    ...               func_list="sum")
    *Groups   col0   col1   col2   col3   col4
    -------   ----   ----   ----   ----   ----
    Group1       0      3      6    nan      5
    Group2       0      2      4    nan      5
    -------   ----   ----   ----   ----   ----
      Total      0      5     10    nan     10
    <BLANKLINE>
    [2 rows x 6 columns] total bytes: 92.0 B

    Without passing a ``name_list`` to :py:func:`~.rt_accumtable.accum_cols`, the
    default column names appear in the returned table.

    **Apply a different reducing function to each array**

    Pass a list of function names as strings to ``func_list``. This example applies a
    respective function in ``func_list`` to each of the arrays in ``val_list``:

    >>> rt.accum_cols(cat=ds.Groups,
    ...               val_list=[ds.Zeros, ds.Ones, ds.Twos, ds.Nans, ds.Ints],
    ...               name_list=["Zeros sum", "Ones mean", "Twos var", "NaNs nansum", "Ints mean"],
    ...               func_list=["sum", "mean", "var", "nansum", "mean"])
    *Groups   Zeros sum   Ones mean   Twos var   NaNs nansum   Ints mean
    -------   ---------   ---------   --------   -----------   ---------
    Group1            0        1.00       0.00          0.00        1.67
    Group2            0        1.00       0.00          0.00        2.50
    -------   ---------   ---------   --------   -----------   ---------
      Total           0        1.00       0.00          0.00        2.00
    <BLANKLINE>
    [2 rows x 6 columns] total bytes: 92.0 B

    **Include ratio arrays**

    Pass a list of two arrays to ``val_list`` to return the ratio of the
    values calculated by :py:meth:`~.rt_groupbyops.GroupByOps.sum` for the two arrays:

    >>> rt.accum_cols(cat=ds.Groups,
    ...               val_list=[ds.Ints, ds.Ones, [ds.Ints, ds.Ones]],
    ...               name_list=["Ints sum", "Ones sum", "Ints:Ones sum ratio"],
    ...               func_list="sum")
    *Groups   Ints sum   Ones sum   Ints:Ones sum ratio
    -------   --------   --------   -------------------
    Group1           5          3                  1.67
    Group2           5          2                  2.50
    -------   --------   --------   -------------------
      Total         10          5                  2.00
    <BLANKLINE>
    [2 rows x 4 columns] total bytes: 60.0 B

    The values returned for the two-element list in ``val_list`` are ratios between the
    values calculated by :py:meth:`~.rt_groupbyops.GroupByOps.sum` for Ints as the
    numerator and for Ones as the denominator. :py:func:`~.rt_accumtable.accum_cols`
    uses :py:func:`~.rt_accumtable.accum_ratio` to calculate this ratio. In the previous
    example, :py:func:`~.rt_accumtable.accum_ratio` is passed the following arguments:

    >>> ints_ones_ratio = rt.accum_ratio(cat1=ds.Groups,
    ...                                  cat2=rt.Categorical(np.full(ds.Groups.shape[0], 1, dtype=np.int8), ["NotGrouped"]),
    ...                                  val1=ds.Ints,
    ...                                  val2=ds.Ones,
    ...                                  func1="sum",
    ...                                  func2="sum",
    ...                                  remove_blanks=False)
    >>> ints_ones_ratio["NotGrouped"]
    FastArray([1.66666667, 2.5       ])

    **Include percentile arrays**

    Pass two-element lists with an array and ``"p"`` to ``val_list`` to return the ratio
    of the values calculated by :py:meth:`~.rt_groupbyops.GroupByOps.sum` for the
    grouped array values compared to the total value for the array, displayed as a
    percent:

    >>> rt.accum_cols(cat=ds.Groups,
    ...               val_list=[ds.Ones, ds.Ints, [ds.Ones, "p"], [ds.Ints, "p"]],
    ...               name_list=["Ones sum", "Ints sum", "Ones percent", "Ints percent"],
    ...               func_list="sum")
    *Groups   Ones sum   Ints sum   Ones percent   Ints percent
    -------   --------   --------   ------------   ------------
    Group1           3          5          60.00          50.00
    Group2           2          5          40.00          50.00
    -------   --------   --------   ------------   ------------
      Total          5         10         100.00         100.00
    <BLANKLINE>
    [2 rows x 5 columns] total bytes: 76.0 B

    The values returned for the two-element lists are the percent ratios. Group1
    of the Ones sum column is 3 and the Total for Ones sum is 5. The ratio of these two
    numbers as a percent is 60.00, as displayed in the Ones percent column.

    :py:func:`~.rt_accumtable.accum_cols` uses :py:func:`~.rt_accumtable.accum_ratiop`
    to calculate this percent ratio. In the previous example,
    :py:func:`~.rt_accumtable.accum_ratiop` is passed the following arguments to
    calculate the Ones percent column:

    >>> ones_ratiop = rt.accum_ratiop(cat1=ds.Groups,
    ...                               cat2=rt.Categorical(np.full(ds.Groups.shape[0], 1, dtype=np.int8), ["NotGrouped"]),
    ...                               val=ds.Ones,
    ...                               filter=None,
    ...                               func="sum",
    ...                               norm_by="T",
    ...                               include_total=False,
    ...                               remove_blanks=False)
    >>> ones_ratiop["NotGrouped"]
    FastArray([60., 40.])

    **Filter all arrays with a single boolean mask**

    Pass an array of booleans to ``filt_list`` to filter all arrays in ``val_list``:

    >>> greater_3_filter = ds.Ints > 3
    >>> rt.accum_cols(cat=ds.Groups,
    ...               val_list=[ds.Zeros, ds.Ones, ds.Twos, ds.Nans, ds.Ints],
    ...               name_list=["Zeros sum", "Ones sum", "Twos sum", "NaNs sum", "Ints sum"],
    ...               filt_list=greater_3_filter,
    ...               func_list="sum")
    *Groups   Zeros sum   Ones sum   Twos sum   NaNs sum   Ints sum
    -------   ---------   --------   --------   --------   --------
    Group1            0          0          0       0.00          0
    Group2            0          1          2        nan          4
    -------   ---------   --------   --------   --------   --------
      Total           0          1          2        nan          4
    <BLANKLINE>
    [2 rows x 6 columns] total bytes: 92.0 B

    **Filter each array with a different boolean mask**

    Pass an array of boolean arrays to ``filt_list`` to filter the respective arrays in
    ``val_list``:

    >>> even_zeros = ds.Zeros % 2 == 0
    >>> even_ones = ds.Ones % 2 == 0
    >>> even_twos = ds.Twos % 2 == 0
    >>> even_nans = ds.Nans % 2 == 0
    >>> even_ints = ds.Ints % 2 == 0
    >>> rt.accum_cols(cat=ds.Groups,
    ...               val_list=[ds.Zeros, ds.Ones, ds.Twos, ds.Nans, ds.Ints],
    ...               name_list=["Zeros sum", "Ones sum", "Twos sum", "NaNs sum", "Ints sum"],
    ...               filt_list=[even_zeros, even_ones, even_twos, even_nans, even_ints],
    ...               func_list="sum")
    *Groups   Zeros sum   Ones sum   Twos sum   NaNs sum   Ints sum
    -------   ---------   --------   --------   --------   --------
    Group1            0          0          6       0.00          2
    Group2            0          0          4       0.00          4
    -------   ---------   --------   --------   --------   --------
      Total           0          0         10       0.00          6
    <BLANKLINE>
    [2 rows x 6 columns] total bytes: 92.0 B

    **Remove blank values**

    Pass `True` to ``remove_blanks`` to remove all rows and columns from the returned
    :py:class:`~.rt_dataset.Dataset` that consist entirely of ``0`` or ``nan``:

    >>> rt.accum_cols(cat=ds.Groups,
    ...               val_list=[ds.Zeros, ds.Ones, ds.Twos, ds.Nans, ds.Ints],
    ...               name_list=["Zeros sum", "Ones sum", "Twos sum", "NaNs sum", "Ints sum"],
    ...               func_list="sum",
    ...               remove_blanks=True)
    *Groups   Ones sum   Twos sum   Ints sum
    -------   --------   --------   --------
    Group1           3          6          5
    Group2           2          4          5
    -------   --------   --------   --------
      Total          5         10         10
    <BLANKLINE>
    [2 rows x 4 columns] total bytes: 60.0 B
    """

    # Handle mistyped inputs
    if not isinstance(cat, Categorical):
        cat = Categorical(cat)
    if not isinstance(val_list, list):
        val_list = [val_list]

    # Handle missing inputs
    if name_list is None:
        name_list = [f"col{n}" for n in range(len(val_list))]
    if filt_list is None:
        val_fst = val_list[0]
        shape = val_fst.shape[0] if isinstance(val_fst, np.ndarray) else val_fst[0].shape[0]
        filt_list = full(shape, True, dtype=bool)
    if not isinstance(func_list, list):
        func_list = [func_list for _ in val_list]
    if not isinstance(filt_list, list):
        filt_list = [filt_list for _ in val_list]

    # Compute accum
    temp_cat = Categorical(full(cat.shape[0], 1, dtype=np.int8), ["NotGrouped"])
    accum = Accum2(cat, temp_cat)

    for val, name, filt, func in zip(val_list, name_list, filt_list, func_list):
        func_name = func
        func = getattr(accum, func_name)
        if isinstance(val, list):  # Special cases
            if isinstance(val[1], str):  # Named cases
                if val[1] in "pP":  # accum_ratiop type
                    curr_data = accum_ratiop(cat, temp_cat, val[0], filt, func_name, "T", False, False)
                else:
                    raise ValueError(f'Invalid accum_cols specifier "{val[1]}" in second argument for column {name}')
            else:  # accum_ratio type
                curr_data = accum_ratio(
                    cat, temp_cat, val[0], val[1], filt, filt, func_name, func_name, remove_blanks=False
                )
        else:
            curr_data = func(val, filter=filt)
        try:
            results[name] = curr_data["NotGrouped"]
        except NameError:
            # Get number of keys in (potentially) multikey categorical. This only happens once.
            cat_width = len(cat.category_dict)
            results = curr_data[:, 0:cat_width]
            results.footer_remove()
            results[name] = curr_data["NotGrouped"]
        footer_val = list(curr_data.footer_get_dict().values())[0].get("NotGrouped", 0.0)
        results.footer_set_values("Total", {name: footer_val})

    if remove_blanks:
        return results.trim()
    else:
        return results


# keep this as the last line
TypeRegister.AccumTable = AccumTable
