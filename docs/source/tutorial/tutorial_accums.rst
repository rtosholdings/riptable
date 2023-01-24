Accums
======

Accums aggregate data similarly to Categoricals, but they distinguish
themselves by providing a fancier output with overall aggregates in
summary footers and columns.

``Accum2()``
------------

``Accum2()`` is very much like a multi-key Categorical: It computes
aggregates values for pairs of groups. The difference is in the output –
an ``Accum2()`` result looks more like a pivot table, with the first
group passed to the function providing row labels and the second
providing the column labels.

The function is also applied to each row and column, with results shown
in a summary column and row, as well as to all columns and rows combined
(with the result shown in the bottom right corner).

We’ll use a Dataset that’s similar to the one we used for multi-key
Categoricals, so we can compare the output::

    >>> rng = np.random.default_rng(seed=42)
    >>> ds = rt.Dataset()
    >>> N = 100
    >>> ds.Symbol = rt.FA(rng.choice(['AAPL', 'AMZN', 'MSFT'], N))
    >>> ds.Value = rt.FA(rng.random(N))
    >>> ds.Date = rt.Date.range('20210101', days = 100)  # Dates from January to mid-April
    >>> ds.Month = ds.Date.start_of_month
    >>> # Accum2 can take Categoricals or FastArrays as input.
    >>> # To use this ds for accum_ratio, we need Symbol and Month to be Categoricals.
    >>> ds.Symbol = rt.Cat(ds.Symbol)
    >>> ds.Month = rt.Cat(ds.Month)
    >>> ds
      #   Symbol   Value         Date   Month     
    ---   ------   -----   ----------   ----------
      0   AAPL      0.20   2021-01-01   2021-01-01
      1   MSFT      0.01   2021-01-02   2021-01-01
      2   AMZN      0.79   2021-01-03   2021-01-01
      3   AMZN      0.66   2021-01-04   2021-01-01
      4   AMZN      0.71   2021-01-05   2021-01-01
      5   MSFT      0.78   2021-01-06   2021-01-01
      6   AAPL      0.46   2021-01-07   2021-01-01
      7   MSFT      0.57   2021-01-08   2021-01-01
      8   AAPL      0.14   2021-01-09   2021-01-01
      9   AAPL      0.11   2021-01-10   2021-01-01
     10   AMZN      0.67   2021-01-11   2021-01-01
     11   MSFT      0.47   2021-01-12   2021-01-01
     12   MSFT      0.57   2021-01-13   2021-01-01
     13   MSFT      0.76   2021-01-14   2021-01-01
     14   MSFT      0.63   2021-01-15   2021-01-01
    ...   ...        ...          ...   ...       
     85   MSFT      0.02   2021-03-27   2021-03-01
     86   AAPL      0.96   2021-03-28   2021-03-01
     87   AAPL      0.48   2021-03-29   2021-03-01
     88   MSFT      0.78   2021-03-30   2021-03-01
     89   MSFT      0.08   2021-03-31   2021-03-01
     90   AMZN      0.49   2021-04-01   2021-04-01
     91   MSFT      0.49   2021-04-02   2021-04-01
     92   MSFT      0.94   2021-04-03   2021-04-01
     93   AMZN      0.57   2021-04-04   2021-04-01
     94   MSFT      0.47   2021-04-05   2021-04-01
     95   AAPL      0.27   2021-04-06   2021-04-01
     96   AAPL      0.33   2021-04-07   2021-04-01
     97   MSFT      0.52   2021-04-08   2021-04-01
     98   AMZN      0.44   2021-04-09   2021-04-01
     99   AAPL      0.02   2021-04-10   2021-04-01

Here’s the ``Accum2()`` table before we apply an aggregation function.
You can see how many values fall into each group pair::

    >>> rt.Accum2(ds.Symbol, ds.Month)
    Accum2 Keys
     X:Date(['2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01'])
     Y:FastArray([b'AAPL', b'AMZN', b'MSFT'], dtype='|S4')
     Bins:20   Rows:100
    
    *Symbol   2021-01-01   2021-02-01   2021-03-01   2021-04-01   Sum
    -------   ----------   ----------   ----------   ----------   ---
    AAPL               6            9            9            3    27
    AMZN              13            8            9            3    33
    MSFT              12           11           13            4    40

If we aggregate with ``count()``, it has the same data, but we see the
output formatting::

    >>> rt.Accum2(ds.Symbol, ds.Month).count()
    *Symbol   2021-01-01   2021-02-01   2021-03-01   2021-04-01   Sum
    -------   ----------   ----------   ----------   ----------   ---
    AAPL               6            9            9            3    27
    AMZN              13            8            9            3    33
    MSFT              12           11           13            4    40
        Sum           31           28           31           10   100

The bottom row and rightmost column provide summary data.

Now we’ll get the average value per symbol-month pair::

    >>> rt.Accum2(ds.Symbol, ds.Month).mean(ds.Value)
    *Symbol   2021-01-01   2021-02-01   2021-03-01   2021-04-01   Mean
    -------   ----------   ----------   ----------   ----------   ----
    AAPL            0.35         0.40         0.54         0.21   0.41
    AMZN            0.54         0.48         0.45         0.50   0.50
    MSFT            0.44         0.47         0.42         0.61   0.46
       Mean         0.47         0.45         0.46         0.45   0.46

Note that the summary row and column show the mean values for all the
input values for each group, not just the means of the displayed group
means.

To illustrate: Here’s the mean of the displayed group mean values for
AAPL::

    >>> (0.35 + 0.40 + 0.54 + 0.21) / 4
    0.375

And here’s the mean of all AAPL values::

    >>> ds.Value.nanmean(filter=ds.Symbol == 'AAPL')
    0.41317486824408933

For comparison, here’s the multi-key Categorical version::

    >>> ds.Symbol_Month = rt.Cat([ds.Symbol, ds.Month])
    >>> ds.Symbol_Month.mean(ds.Value)
    *Symbol   *Month       Value
    -------   ----------   -----
    AAPL      2021-01-01    0.35
    MSFT      2021-01-01    0.44
    AMZN      2021-01-01    0.54
    AAPL      2021-02-01    0.40
    AMZN      2021-02-01    0.48
    MSFT      2021-02-01    0.47
    .         2021-03-01    0.42
    AMZN      2021-03-01    0.45
    AAPL      2021-03-01    0.54
    AMZN      2021-04-01    0.50
    MSFT      2021-04-01    0.61
    AAPL      2021-04-01    0.21

You can pass a filter keyword argument to the function you call on
``Accum2()``::

    >>> rt.Accum2(ds.Symbol, ds.Month).mean(ds.Value, filter=ds.Value > 0.5)
    *Symbol   2021-01-01   2021-02-01   2021-03-01   2021-04-01   Mean
    -------   ----------   ----------   ----------   ----------   ----
    AAPL            0.85         0.74         0.76          nan   0.77
    AMZN            0.67         0.75         0.72         0.57   0.69
    MSFT            0.65         0.78         0.70         0.73   0.71
       Mean         0.67         0.76         0.72         0.68   0.71

``accum_ratio()``
-----------------

For each group pair, ``accum_ratio()`` computes a ratio of values you
specify. The results are presented in an Accum table.

For our example we’ll add PnL and Size (number of sales) columns, and
we’ll use ``accum_ratio()`` to get the PnL for each symbol-month bucket,
weighted by size::

    >>> ds.PnL = rng.normal(10, 20, 100)
    >>> ds.Size = rng.random(100) * 100

Like ``Accum2()``, ``accum_ratio()`` takes two Categoricals (a row
Categorical and a column Categorical). You also specify the numerator
values and denominator values. For each group pair, it sums the
numerator values and denominator values and presents the ratios in a
table::

    >>> rt.accum_ratio(ds.Symbol, ds.Month, ds.PnL * ds.Size, ds.Size, include_numer=True)
    *Symbol   2021-01-01   2021-02-01   2021-03-01   2021-04-01   Ratio       Numer      Denom
    -------   ----------   ----------   ----------   ----------   -----   ---------   --------
    AAPL            3.13        11.93         1.95        28.86    8.81   12,363.71   1,404.13
    AMZN            5.54         2.36        23.34        -2.94   10.01   16,971.55   1,695.67
    MSFT           23.90        22.78        -1.40        -9.61   10.35   17,501.11   1,690.46
      Ratio        10.13        13.17         7.31         8.25    9.78
      Numer    10,604.13    18,953.08    13,471.17     3,807.98           46,836.36
      Denom     1,047.18     1,438.84     1,842.65       461.59                       4,790.26

The result is the ratio of the following two tables.

Numerator::

    >>> rt.Accum2(ds.Symbol, ds.Month).nansum(ds.Size * ds.PnL)
    *Symbol   2021-01-01   2021-02-01   2021-03-01   2021-04-01      Nansum
    -------   ----------   ----------   ----------   ----------   ---------
    AAPL          699.07     5,075.98     1,100.76     5,487.90   12,363.71
    AMZN        2,956.74     1,065.03    13,358.59      -408.81   16,971.55
    MSFT        6,948.32    12,812.08      -988.18    -1,271.11   17,501.11
     Nansum    10,604.13    18,953.08    13,471.17     3,807.98   46,836.36

Denominator::

    >>> rt.Accum2(ds.Symbol, ds.Month).nansum(ds.Size)
    *Symbol   2021-01-01   2021-02-01   2021-03-01   2021-04-01     Nansum
    -------   ----------   ----------   ----------   ----------   --------
    AAPL          223.12       425.49       565.38       190.13   1,404.13
    AMZN          533.28       450.83       572.34       139.22   1,695.67
    MSFT          290.78       562.52       704.92       132.24   1,690.46
     Nansum     1,047.18     1,438.84     1,842.65       461.59   4,790.26

When the numerator and denominator are the same, the result is as you
might expect::

    >>> rt.accum_ratio(ds.Symbol, ds.Month, ds.Size, ds.Size, include_numer=True)
    *Symbol   2021-01-01   2021-02-01   2021-03-01   2021-04-01   Ratio      Numer      Denom
    -------   ----------   ----------   ----------   ----------   -----   --------   --------
    AAPL            1.00         1.00         1.00         1.00    1.00   1,404.13   1,404.13
    AMZN            1.00         1.00         1.00         1.00    1.00   1,695.67   1,695.67
    MSFT            1.00         1.00         1.00         1.00    1.00   1,690.46   1,690.46
      Ratio         1.00         1.00         1.00         1.00    1.00
      Numer     1,047.18     1,438.84     1,842.65       461.59           4,790.26
      Denom     1,047.18     1,438.84     1,842.65       461.59                      4,790.26

``accum_ratiop()``
------------------

``accum_ratiop()`` takes one column of values as numerators and computes
an internal ratio for each group pair, where the denominator is one of
three sums:

-  The row sum (``norm_by='R'``)
-  The column sum (``norm_by='C'``)
-  The total sum (``norm_by='T'``)

For example, this table shows that 30.30% of AAPL sales were in
February::

    >>> rt.accum_ratiop(ds.Symbol, ds.Month, ds.Size, norm_by='R')
    *Symbol      2021-01-01   2021-02-01   2021-03-01   2021-04-01   TotalRatio      Total
    ----------   ----------   ----------   ----------   ----------   ----------   --------
    AAPL              15.89        30.30        40.27        13.54       100.00   1,404.13
    AMZN              31.45        26.59        33.75         8.21       100.00   1,695.67
    MSFT              17.20        33.28        41.70         7.82       100.00   1,690.46
    TotalRatio        21.86        30.04        38.47         9.64       100.00
         Total     1,047.18     1,438.84     1,842.65       461.59                4,790.26

Note that the percentages in each row sum to 100%.

We can check the math by computing the ratio of AAPL’s February sales to
AAPL’s total sales::

    >>> filt_feb_aapl = (ds.Symbol == 'AAPL') & (ds.Month.as_string_array == rt.Date('20210201'))
    >>> filt_total_aapl = ds.Symbol == 'AAPL'
    >>> ds.Size[filt_feb_aapl].nansum() / ds.Size[filt_total_aapl].nansum()
    0.3030291108538412

This table shows that AAPL’s sales are 29.57% of February sales::

    >>> rt.accum_ratiop(ds.Symbol, ds.Month, ds.Size, norm_by='C')
    *Symbol      2021-01-01   2021-02-01   2021-03-01   2021-04-01   TotalRatio      Total
    ----------   ----------   ----------   ----------   ----------   ----------   --------
    AAPL              21.31        29.57        30.68        41.19        29.31   1,404.13
    AMZN              50.93        31.33        31.06        30.16        35.40   1,695.67
    MSFT              27.77        39.10        38.26        28.65        35.29   1,690.46
    TotalRatio       100.00       100.00       100.00       100.00       100.00
         Total     1,047.18     1,438.84     1,842.65       461.59                4,790.26

Note that the percentages in each column sum to 100%.

Check the math::

    >>> filt_feb_total = ds.Month.as_string_array == rt.Date('20210201')
    >>> ds.Size[filt_feb_aapl].nansum() / ds.Size[filt_feb_total].nansum()
    0.29571866540362846

This table shows that AAPL’s February sales represent 8.88% of all
sales::

    >>> rt.accum_ratiop(ds.Symbol, ds.Month, ds.Size, norm_by='T')
    *Symbol      2021-01-01   2021-02-01   2021-03-01   2021-04-01   TotalRatio      Total
    ----------   ----------   ----------   ----------   ----------   ----------   --------
    AAPL               4.66         8.88        11.80         3.97        29.31   1,404.13
    AMZN              11.13         9.41        11.95         2.91        35.40   1,695.67
    MSFT               6.07        11.74        14.72         2.76        35.29   1,690.46
    TotalRatio        21.86        30.04        38.47         9.64       100.00
         Total     1,047.18     1,438.84     1,842.65       461.59                4,790.26

Note that the “TotalRatio” row and column percentages each sum to 100%.

Check the math::

    >>> ds.Size[filt_feb_aapl].nansum() / ds.Size.nansum()
    0.08882445025331744

Next, for something completely different, we’ll explore ways to
`Concatenate Datasets <tutorial_concat.rst>`__.

--------------

Questions or comments about this guide? Email
RiptableDocumentation@sig.com.
