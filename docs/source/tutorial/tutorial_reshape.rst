Reshape Data with Pivot and Transpose
=====================================

Riptable is designed to efficiently work with column-oriented data,
which is also called long-format data. This isn’t always the best format
for displaying data for human consumption, however.

For example, suppose your data consists of a measurement (say, trade
volume) per date and symbol. The long-format, Riptable-friendly way to
represent this is to have three columns – for date, symbol, and volume::

    >>> long_ds = rt.Dataset({'Date': ['20191111', '20191111', '20191111', '20191112', 
    ...                                '20191112', '20191112'],
    ...                       'Symbol': ['AAPL', 'MSFT', 'TSLA', 'MSFT', 'AAPL', 'TSLA'],
    ...                       'Volume': [10, 20, 30, 20, 10, 30]})
    >>> long_ds
    #   Date       Symbol   Volume
    -   --------   ------   ------
    0   20191111   AAPL         10
    1   20191111   MSFT         20
    2   20191111   TSLA         30
    3   20191112   MSFT         20
    4   20191112   AAPL         10
    5   20191112   TSLA         30

While this format is ideal for Riptable’s work, the repeated date and
symbol values make it a bit unintuitive for humans to read and make
sense of.

In this case, a simple transform from long format to wide doesn’t help
much::

    >>> long_ds._T
    Fields:	       0	       1	       2	       3	       4	       5
       Date	20191111	20191111	20191111	20191112	20191112	20191112
     Symbol	    AAPL	    MSFT	    TSLA	    MSFT	    AAPL	    TSLA
     Volume	      10	      20	      30	      20	      10	      30

A more human-friendly presentation can be gotten from the Dataset method
``pivot()``, which reorganizes data with multiple keys (here, our keys
are the date and the symbol).

We can use ``pivot()`` to show one row per date and one column for each
symbol::

    >>> wide_ds = long_ds.pivot('Date', 'Symbol', 'Volume')
    >>> wide_ds
    *Date      AAPL   MSFT   TSLA
    --------   ----   ----   ----
    20191111     10     20     30
    20191112     10     20     30

The first argument passed is used for the row labels; the second is for
the column labels. The third argument specifies which column’s (or
columns’) data to use to populate the table. (If none are specified, all
remaining columns are used.)

Notice the output’s similarity to that of ``Accum2()``::

    >>> long_ds.accum2(long_ds.Date, long_ds.Symbol).sum(long_ds.Volume)
    *Date      AAPL   MSFT   TSLA   Total
    --------   ----   ----   ----   -----
    20191111     10     20     30      60
    20191112     10     20     30      60
       Total     20     40     60     120

Also note that some wide-format data may be too wide for reasonable
display.

To undo your pivot (or “unpivot”), use ``melt()``::

    >>> melted_ds = wide_ds.melt('Date')
    >>> melted_ds
    #   Date       variable   value
    -   --------   --------   -----
    0   20191111   AAPL          10
    1   20191112   AAPL          10
    2   20191111   MSFT          20
    3   20191112   MSFT          20
    4   20191111   TSLA          30
    5   20191112   TSLA          30

Here, we specified the Date column as the “identifier variable.” The
other columns, considered “measured variables,” are unpivoted to the row
axis with the column headers “variable” and “value.”

We could have specified our original column labels with ``var_name`` and
``value_name``::

    >>> melted_ds = wide_ds.melt('Date', var_name='Symbol', value_name='Volume')
    >>> melted_ds
    #   Date       Symbol   Volume
    -   --------   ------   ------
    0   20191111   AAPL         10
    1   20191112   AAPL         10
    2   20191111   MSFT         20
    3   20191112   MSFT         20
    4   20191111   TSLA         30
    5   20191112   TSLA         30

Next, we’ll change gears to give you a high-level overview of tools you
can use to `Visualize Data <tutorial_visualize.rst>`__.

--------------

Questions or comments about this guide? Email
RiptableDocumentation@sig.com.
