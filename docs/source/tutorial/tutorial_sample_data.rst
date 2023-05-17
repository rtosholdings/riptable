Instantiate with Placeholder Values and Generate Sample Data
============================================================

It’s useful to have a few tools in your back pocket for generating data
quickly – either placeholder values (like 1s or 0s) meant to temporarily
fill a certain structure you’re instantiating, or sample values that
mimic real data, which you can use to explore and experiment with
Riptable.

Here’s a brief sampling of Riptable and NumPy methods you can use. For
complete details about these functions, see their API reference
documentation.

Generate Placeholder Values
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following methods generate repeated 0s and 1s.

10 floating-point zeros::

    >>> rt.zeros(10)
    FastArray([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

10 integer zeros::

    >>> rt.zeros(10, int)
    FastArray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

10 floating-point ones::

    >>> rt.ones(10)
    FastArray([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

10 integer ones::

    >>> rt.ones(10, int)
    FastArray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

The following methods generate repeated specified values.

10 fives::

    >>> rt.repeat(5, 10)
    FastArray([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])

10 repeats of each array element::

    >>> rt.repeat([1, 2], 10)
    FastArray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

10 "tiles" of the entire array::

    >>> rt.tile([1, 2], 10)
    FastArray([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2])

10 twos::

    >>> rt.full(10, 2)
    FastArray([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

Generate Sample Data
~~~~~~~~~~~~~~~~~~~~

Most of these methods generate a range of values.

``arange()`` generates evenly spaced floating-point or integer values
(depending on the input) within a given interval, including the start
value but excluding the stop value. You can also specify a step size
(the spacing between the values; the default is 1). It’s like Python’s
*range* function, but it returns a FastArray rather than a list.

Numbers 0 through 9::

    >>> rt.arange(10) 
    FastArray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

Every second number from 1 to 9::

    >>> rt.arange(1, 10, 2) 
    FastArray([1, 3, 5, 7, 9])

For evenly spaced values where the step is a non-integer, it’s better to
use ``np.linspace``. Instead of specifying the step value, you specify
the number of elements. Both the start and stop values are included::

    >>> np.linspace(2.0, 3.0, num=5)  # Five evenly spaced numbers from 2.0 to 3.0 (the step is 0.25)
    array([2.  , 2.25, 2.5 , 2.75, 3.  ])

For randomly generated values, you have several options.

For integers and floating-point values, NumPy has you covered. Call
``default_rng`` to get a new instance of a NumPy Generator, then call
its methods. To generate values that can be replicated, initialize with
a seed value of your choice to initialize the BitGenerator::

    >>> rng = np.random.default_rng(seed=42)

10 floating-point numbers between 0.0 and 1.0 (1.0 excluded)::

    >>> rng.random(10)
    array([0.77395605, 0.43887844, 0.85859792, 0.69736803, 0.09417735,
           0.97562235, 0.7611397 , 0.78606431, 0.12811363, 0.45038594])

10 uniformly distributed floats between 0 and 50 (50 excluded)::

    >>> rng.uniform(0, 50, 10)
    array([18.53990121, 46.33824944, 32.193256  , 41.13808066, 22.17070994,
           11.36193609, 27.72923935,  3.19086281, 41.3815586 , 31.58321996])

10 integers between 1 and 50 (50 excluded)::

    >>> rng.integers(1, 50, 10)
    array([ 9, 38, 35, 18,  4, 48, 22, 44, 34, 39], dtype=int64)

10 strings chosen from a list::

    >>> rng.choice(['GME', 'AMZN', 'TSLA', 'SPY'], 10)
    array(['SPY', 'GME', 'AMZN', 'AMZN', 'AMZN', 'GME', 'TSLA', 'GME', 'TSLA', 'TSLA'], dtype='<U4')

10 random Booleans::

    >>> rng.choice([True, False], 10)
    array([False, False,  True, False,  True,  True, False,  True,  True, True])

See `NumPy’s
documentation <https://numpy.org/doc/stable/user/index.html>`__ for more details and other methods.

Riptable has methods for generating random Date and DateTimeNano arrays.

5 DateTimeNanos with NYT time zone::

    >>> rt.DateTimeNano.random(5)
    DateTimeNano(['20000507 22:02:14.350793900', '20040720 00:24:28.668289697', '19771017 22:34:39.521017110', '20130819 05:29:22.584265022', '20170622 00:50:06.970974486'], to_tz='NYC')

Dates between a start date and an end date (start and end dates
included; the default step is 1 day)::

    >>> rt.Date.range('20190201', '20190208')
    Date(['2019-02-01', '2019-02-02', '2019-02-03', '2019-02-04', '2019-02-05', '2019-02-06', '2019-02-07', '2019-02-08'])

5 dates, spaced two days apart, with a specified start date (start date
included)::

    >>> rt.Date.range('20190201', days=5, step=2)
    Date(['2019-02-01', '2019-02-03', '2019-02-05', '2019-02-07', '2019-02-09'])

Though ``Date`` objects don’t (yet) have a ``random`` method, you can
use ``rng.choice`` to pick dates from a range::

    >>> rt.Date(rng.choice(rt.Date.range('20220201', '20220430'), 5))
    Date(['2022-04-12', '2022-02-17', '2022-03-14', '2022-02-12', '2022-04-03'])

Next we cover ways to get data in and out of Riptable: `Work with Riptable Files and Other File Formats <tutorial_io.rst>`__.

--------------

Questions or comments about this guide? Email
RiptableDocumentation@sig.com.
