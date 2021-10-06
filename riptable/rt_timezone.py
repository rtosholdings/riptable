__all__ = [
    'TimeZone'
]

import numpy as np

import riptide_cpp as rc

from .rt_fastarray import FastArray
from .rt_enum import TypeRegister
from .rt_datetime import NANOS_PER_HOUR
from .rt_numpy import putmask, searchsorted, zeros, where

DST_CUTOFFS_NYC = FastArray([
'1970-04-26 02:00:00', '1970-10-25 02:00:00', 
'1971-04-25 02:00:00', '1971-10-31 02:00:00', 
'1972-04-30 02:00:00', '1972-10-29 02:00:00', 
'1973-04-29 02:00:00', '1973-10-28 02:00:00', 
'1974-01-06 02:00:00', '1974-10-27 02:00:00', 
'1975-02-23 02:00:00', '1975-10-26 02:00:00', 
'1976-04-25 02:00:00', '1976-10-31 02:00:00', 
'1977-04-24 02:00:00', '1977-10-30 02:00:00', 
'1978-04-30 02:00:00', '1978-10-29 02:00:00', 
'1979-04-29 02:00:00', '1979-10-28 02:00:00', 
'1980-04-27 02:00:00', '1980-10-26 02:00:00', 
'1981-04-26 02:00:00', '1981-10-25 02:00:00', 
'1982-04-25 02:00:00', '1982-10-31 02:00:00', 
'1983-04-24 02:00:00', '1983-10-30 02:00:00', 
'1984-04-29 02:00:00', '1984-10-28 02:00:00', 
'1985-04-28 02:00:00', '1985-10-27 02:00:00', 
'1986-04-27 02:00:00', '1986-10-26 02:00:00', 
'1987-04-05 02:00:00', '1987-10-25 02:00:00', 
'1988-04-03 02:00:00', '1988-10-30 02:00:00', 
'1989-04-02 02:00:00', '1989-10-29 02:00:00', 
'1990-04-01 02:00:00', '1990-10-28 02:00:00', 
'1991-04-07 02:00:00', '1991-10-27 02:00:00', 
'1992-04-05 02:00:00', '1992-10-25 02:00:00', 
'1993-04-04 02:00:00', '1993-10-31 02:00:00', 
'1994-04-03 02:00:00', '1994-10-30 02:00:00', 
'1995-04-02 02:00:00', '1995-10-29 02:00:00', 
'1996-04-07 02:00:00', '1996-10-27 02:00:00', 
'1997-04-06 02:00:00', '1997-10-26 02:00:00', 
'1998-04-05 02:00:00', '1998-10-25 02:00:00', 
'1999-04-04 02:00:00', '1999-10-31 02:00:00',
'2000-04-02 02:00:00', '2000-10-29 02:00:00',
'2001-04-01 02:00:00', '2001-10-28 02:00:00',
'2002-04-07 02:00:00', '2002-10-27 02:00:00',
'2003-04-06 02:00:00', '2003-10-26 02:00:00',
'2004-04-04 02:00:00', '2004-10-31 02:00:00',
'2005-04-03 02:00:00', '2005-10-30 02:00:00',
'2006-04-02 02:00:00', '2006-10-29 02:00:00',
'2007-03-11 02:00:00', '2007-11-04 02:00:00',
'2008-03-09 02:00:00', '2008-11-02 02:00:00',
'2009-03-08 02:00:00', '2009-11-01 02:00:00',
'2010-03-14 02:00:00', '2010-11-07 02:00:00',
'2011-03-13 02:00:00', '2011-11-06 02:00:00',
'2012-03-11 02:00:00', '2012-11-04 02:00:00',
'2013-03-10 02:00:00', '2013-11-03 02:00:00',
'2014-03-09 02:00:00', '2014-11-02 02:00:00',
'2015-03-08 02:00:00', '2015-11-01 02:00:00',
'2016-03-13 02:00:00', '2016-11-06 02:00:00',
'2017-03-12 02:00:00', '2017-11-05 02:00:00',
'2018-03-11 02:00:00', '2018-11-04 02:00:00',
'2019-03-10 02:00:00', '2019-11-03 02:00:00',
'2020-03-08 02:00:00', '2020-11-01 02:00:00',
'2021-03-14 02:00:00', '2021-11-07 02:00:00',
'2022-03-13 02:00:00', '2022-11-06 02:00:00',
'2023-03-12 02:00:00', '2023-11-05 02:00:00',
'2024-03-10 02:00:00', '2024-11-03 02:00:00',
'2025-03-09 02:00:00', '2025-11-02 02:00:00',
'2026-03-08 02:00:00', '2026-11-01 02:00:00',
'2027-03-14 02:00:00', '2027-11-07 02:00:00',
'2028-03-12 02:00:00', '2028-11-05 02:00:00',
'2029-03-11 02:00:00', '2029-11-04 02:00:00',
'2030-03-10 02:00:00', '2030-11-03 02:00:00',
'2031-03-09 02:00:00', '2031-11-02 02:00:00',
'2032-03-14 02:00:00', '2032-11-07 02:00:00',
'2033-03-13 02:00:00', '2033-11-06 02:00:00',
'2034-03-12 02:00:00', '2034-11-05 02:00:00',
'2035-03-11 02:00:00', '2035-11-04 02:00:00',
'2036-03-09 02:00:00', '2036-11-02 02:00:00',
'2037-03-08 02:00:00', '2037-11-01 02:00:00',
'2038-03-14 02:00:00', '2038-11-07 02:00:00',
'2039-03-13 02:00:00', '2039-11-06 02:00:00',
'2040-03-11 02:00:00', '2040-11-04 02:00:00',
])

DST_REVERSE_NYC = FastArray([
'1970-04-26 07:00:00', '1970-10-25 06:00:00',
'1971-04-25 07:00:00', '1971-10-31 06:00:00',
'1972-04-30 07:00:00', '1972-10-29 06:00:00',
'1973-04-29 07:00:00', '1973-10-28 06:00:00',
'1974-01-06 07:00:00', '1974-10-27 06:00:00',
'1975-02-23 07:00:00', '1975-10-26 06:00:00',
'1976-04-25 07:00:00', '1976-10-31 06:00:00',
'1977-04-24 07:00:00', '1977-10-30 06:00:00',
'1978-04-30 07:00:00', '1978-10-29 06:00:00',
'1979-04-29 07:00:00', '1979-10-28 06:00:00',
'1980-04-27 07:00:00', '1980-10-26 06:00:00',
'1981-04-26 07:00:00', '1981-10-25 06:00:00',
'1982-04-25 07:00:00', '1982-10-31 06:00:00',
'1983-04-24 07:00:00', '1983-10-30 06:00:00',
'1984-04-29 07:00:00', '1984-10-28 06:00:00',
'1985-04-28 07:00:00', '1985-10-27 06:00:00',
'1986-04-27 07:00:00', '1986-10-26 06:00:00',
'1987-04-05 07:00:00', '1987-10-25 06:00:00',
'1988-04-03 07:00:00', '1988-10-30 06:00:00',
'1989-04-02 07:00:00', '1989-10-29 06:00:00',
'1990-04-01 07:00:00', '1990-10-28 06:00:00',
'1991-04-07 07:00:00', '1991-10-27 06:00:00',
'1992-04-05 07:00:00', '1992-10-25 06:00:00',
'1993-04-04 07:00:00', '1993-10-31 06:00:00',
'1994-04-03 07:00:00', '1994-10-30 06:00:00',
'1995-04-02 07:00:00', '1995-10-29 06:00:00',
'1996-04-07 07:00:00', '1996-10-27 06:00:00',
'1997-04-06 07:00:00', '1997-10-26 06:00:00',
'1998-04-05 07:00:00', '1998-10-25 06:00:00',
'1999-04-04 07:00:00', '1999-10-31 06:00:00',
'2000-04-02 07:00:00', '2000-10-29 06:00:00',
'2001-04-01 07:00:00', '2001-10-28 06:00:00',
'2002-04-07 07:00:00', '2002-10-27 06:00:00',
'2003-04-06 07:00:00', '2003-10-26 06:00:00',
'2004-04-04 07:00:00', '2004-10-31 06:00:00',
'2005-04-03 07:00:00', '2005-10-30 06:00:00',
'2006-04-02 07:00:00', '2006-10-29 06:00:00',
'2007-03-11 07:00:00', '2007-11-04 06:00:00',
'2008-03-09 07:00:00', '2008-11-02 06:00:00',
'2009-03-08 07:00:00', '2009-11-01 06:00:00',
'2010-03-14 07:00:00', '2010-11-07 06:00:00',
'2011-03-13 07:00:00', '2011-11-06 06:00:00',
'2012-03-11 07:00:00', '2012-11-04 06:00:00',
'2013-03-10 07:00:00', '2013-11-03 06:00:00',
'2014-03-09 07:00:00', '2014-11-02 06:00:00',
'2015-03-08 07:00:00', '2015-11-01 06:00:00',
'2016-03-13 07:00:00', '2016-11-06 06:00:00',
'2017-03-12 07:00:00', '2017-11-05 06:00:00',
'2018-03-11 07:00:00', '2018-11-04 06:00:00',
'2019-03-10 07:00:00', '2019-11-03 06:00:00',
'2020-03-08 07:00:00', '2020-11-01 06:00:00',
'2021-03-14 07:00:00', '2021-11-07 06:00:00',
'2022-03-13 07:00:00', '2022-11-06 06:00:00',
'2023-03-12 07:00:00', '2023-11-05 06:00:00',
'2024-03-10 07:00:00', '2024-11-03 06:00:00',
'2025-03-09 07:00:00', '2025-11-02 06:00:00',
'2026-03-08 07:00:00', '2026-11-01 06:00:00',
'2027-03-14 07:00:00', '2027-11-07 06:00:00',
'2028-03-12 07:00:00', '2028-11-05 06:00:00',
'2029-03-11 07:00:00', '2029-11-04 06:00:00',
'2030-03-10 07:00:00', '2030-11-03 06:00:00',
'2031-03-09 07:00:00', '2031-11-02 06:00:00',
'2032-03-14 07:00:00', '2032-11-07 06:00:00',
'2033-03-13 07:00:00', '2033-11-06 06:00:00',
'2034-03-12 07:00:00', '2034-11-05 06:00:00',
'2035-03-11 07:00:00', '2035-11-04 06:00:00',
'2036-03-09 07:00:00', '2036-11-02 06:00:00',
'2037-03-08 07:00:00', '2037-11-01 06:00:00',
'2038-03-14 07:00:00', '2038-11-07 06:00:00',
'2039-03-13 07:00:00', '2039-11-06 06:00:00',
'2040-03-11 07:00:00', '2040-11-04 06:00:00'
])


DST_CUTOFFS_DUBLIN = FastArray([
 '1972-03-19 02:00:00', '1972-10-29 02:00:00',
 '1973-03-18 02:00:00', '1973-10-28 02:00:00',
 '1974-03-17 02:00:00', '1974-10-27 02:00:00',
 '1975-03-16 02:00:00', '1975-10-26 02:00:00',
 '1976-03-21 02:00:00', '1976-10-24 02:00:00',
 '1977-03-20 02:00:00', '1977-10-23 02:00:00',
 '1978-03-19 02:00:00', '1978-10-29 02:00:00',
 '1979-03-18 02:00:00', '1979-10-28 02:00:00',
 '1980-03-16 02:00:00', '1980-10-26 02:00:00',
 '1981-03-29 02:00:00', '1981-10-25 02:00:00',
 '1982-03-28 02:00:00', '1982-10-24 02:00:00',
 '1983-03-27 02:00:00', '1983-10-23 02:00:00',
 '1984-03-25 02:00:00', '1984-10-28 02:00:00',
 '1985-03-31 02:00:00', '1985-10-27 02:00:00',
 '1986-03-30 02:00:00', '1986-10-26 02:00:00',
 '1987-03-29 02:00:00', '1987-10-25 02:00:00',
 '1988-03-27 02:00:00', '1988-10-23 02:00:00',
 '1989-03-26 02:00:00', '1989-10-29 02:00:00',
 '1990-03-25 02:00:00', '1990-10-28 02:00:00',
 '1991-03-31 02:00:00', '1991-10-27 02:00:00',
 '1992-03-29 02:00:00', '1992-10-25 02:00:00',
 '1993-03-28 02:00:00', '1993-10-24 02:00:00',
 '1994-03-27 02:00:00', '1994-10-23 02:00:00',
 '1995-03-26 02:00:00', '1995-10-22 02:00:00',
 '1996-03-31 02:00:00', '1996-10-27 02:00:00',
 '1997-03-30 02:00:00', '1997-10-26 02:00:00',
 '1998-03-29 02:00:00', '1998-10-25 02:00:00',
 '1999-03-28 02:00:00', '1999-10-31 02:00:00',
 '2000-03-26 02:00:00', '2000-10-29 02:00:00',
 '2001-03-25 02:00:00', '2001-10-28 02:00:00',
 '2002-03-31 02:00:00', '2002-10-27 02:00:00',
 '2003-03-30 02:00:00', '2003-10-26 02:00:00',
 '2004-03-28 02:00:00', '2004-10-31 02:00:00',
 '2005-03-27 02:00:00', '2005-10-30 02:00:00',
 '2006-03-26 02:00:00', '2006-10-29 02:00:00',
 '2007-03-25 02:00:00', '2007-10-28 02:00:00',
 '2008-03-30 02:00:00', '2008-10-26 02:00:00',
 '2009-03-29 02:00:00', '2009-10-25 02:00:00',
 '2010-03-28 02:00:00', '2010-10-31 02:00:00',
 '2011-03-27 02:00:00', '2011-10-30 02:00:00',
 '2012-03-25 02:00:00', '2012-10-28 02:00:00',
 '2013-03-31 02:00:00', '2013-10-27 02:00:00',
 '2014-03-30 02:00:00', '2014-10-26 02:00:00',
 '2015-03-29 02:00:00', '2015-10-25 02:00:00',
 '2016-03-27 02:00:00', '2016-10-30 02:00:00',
 '2017-03-26 02:00:00', '2017-10-29 02:00:00',
 '2018-03-25 02:00:00', '2018-10-28 02:00:00',
 '2019-03-31 02:00:00', '2019-10-27 02:00:00',
 '2020-03-29 02:00:00', '2020-10-25 02:00:00',
 '2021-03-28 02:00:00', '2021-10-31 02:00:00',
 '2022-03-27 02:00:00', '2022-10-30 02:00:00',
 '2023-03-26 02:00:00', '2023-10-29 02:00:00',
 '2024-03-31 02:00:00', '2024-10-27 02:00:00',
 '2025-03-30 02:00:00', '2025-10-26 02:00:00',
 '2026-03-29 02:00:00', '2026-10-25 02:00:00',
 '2027-03-28 02:00:00', '2027-10-31 02:00:00',
 '2028-03-26 02:00:00', '2028-10-29 02:00:00',
 '2029-03-25 02:00:00', '2029-10-28 02:00:00',
 '2030-03-31 02:00:00', '2030-10-27 02:00:00',
 '2031-03-30 02:00:00', '2031-10-26 02:00:00',
 '2032-03-28 02:00:00', '2032-10-31 02:00:00',
 '2033-03-27 02:00:00', '2033-10-30 02:00:00',
 '2034-03-26 02:00:00', '2034-10-29 02:00:00',
 '2035-03-25 02:00:00', '2035-10-28 02:00:00',
 '2036-03-30 02:00:00', '2036-10-26 02:00:00',
 '2037-03-29 02:00:00', '2037-10-25 02:00:00',
 '2038-03-28 02:00:00', '2038-10-31 02:00:00',
 '2039-03-27 02:00:00', '2039-10-30 02:00:00',
 '2040-03-25 02:00:00', '2040-10-28 02:00:00'
 ])


DST_REVERSE_DUBLIN = FastArray([
 '1972-03-19 01:00:00', '1972-10-29 01:00:00',
 '1973-03-18 01:00:00', '1973-10-28 01:00:00',
 '1974-03-17 01:00:00', '1974-10-27 01:00:00',
 '1975-03-16 01:00:00', '1975-10-26 01:00:00',
 '1976-03-21 01:00:00', '1976-10-24 01:00:00',
 '1977-03-20 01:00:00', '1977-10-23 01:00:00',
 '1978-03-19 01:00:00', '1978-10-29 01:00:00',
 '1979-03-18 01:00:00', '1979-10-28 01:00:00',
 '1980-03-16 01:00:00', '1980-10-26 01:00:00',
 '1981-03-29 01:00:00', '1981-10-25 01:00:00',
 '1982-03-28 01:00:00', '1982-10-24 01:00:00',
 '1983-03-27 01:00:00', '1983-10-23 01:00:00',
 '1984-03-25 01:00:00', '1984-10-28 01:00:00',
 '1985-03-31 01:00:00', '1985-10-27 01:00:00',
 '1986-03-30 01:00:00', '1986-10-26 01:00:00',
 '1987-03-29 01:00:00', '1987-10-25 01:00:00',
 '1988-03-27 01:00:00', '1988-10-23 01:00:00',
 '1989-03-26 01:00:00', '1989-10-29 01:00:00',
 '1990-03-25 01:00:00', '1990-10-28 01:00:00',
 '1991-03-31 01:00:00', '1991-10-27 01:00:00',
 '1992-03-29 01:00:00', '1992-10-25 01:00:00',
 '1993-03-28 01:00:00', '1993-10-24 01:00:00',
 '1994-03-27 01:00:00', '1994-10-23 01:00:00',
 '1995-03-26 01:00:00', '1995-10-22 01:00:00',
 '1996-03-31 01:00:00', '1996-10-27 01:00:00',
 '1997-03-30 01:00:00', '1997-10-26 01:00:00',
 '1998-03-29 01:00:00', '1998-10-25 01:00:00',
 '1999-03-28 01:00:00', '1999-10-31 01:00:00',
 '2000-03-26 01:00:00', '2000-10-29 01:00:00',
 '2001-03-25 01:00:00', '2001-10-28 01:00:00',
 '2002-03-31 01:00:00', '2002-10-27 01:00:00',
 '2003-03-30 01:00:00', '2003-10-26 01:00:00',
 '2004-03-28 01:00:00', '2004-10-31 01:00:00',
 '2005-03-27 01:00:00', '2005-10-30 01:00:00',
 '2006-03-26 01:00:00', '2006-10-29 01:00:00',
 '2007-03-25 01:00:00', '2007-10-28 01:00:00',
 '2008-03-30 01:00:00', '2008-10-26 01:00:00',
 '2009-03-29 01:00:00', '2009-10-25 01:00:00',
 '2010-03-28 01:00:00', '2010-10-31 01:00:00',
 '2011-03-27 01:00:00', '2011-10-30 01:00:00',
 '2012-03-25 01:00:00', '2012-10-28 01:00:00',
 '2013-03-31 01:00:00', '2013-10-27 01:00:00',
 '2014-03-30 01:00:00', '2014-10-26 01:00:00',
 '2015-03-29 01:00:00', '2015-10-25 01:00:00',
 '2016-03-27 01:00:00', '2016-10-30 01:00:00',
 '2017-03-26 01:00:00', '2017-10-29 01:00:00',
 '2018-03-25 01:00:00', '2018-10-28 01:00:00',
 '2019-03-31 01:00:00', '2019-10-27 01:00:00',
 '2020-03-29 01:00:00', '2020-10-25 01:00:00',
 '2021-03-28 01:00:00', '2021-10-31 01:00:00',
 '2022-03-27 01:00:00', '2022-10-30 01:00:00',
 '2023-03-26 01:00:00', '2023-10-29 01:00:00',
 '2024-03-31 01:00:00', '2024-10-27 01:00:00',
 '2025-03-30 01:00:00', '2025-10-26 01:00:00',
 '2026-03-29 01:00:00', '2026-10-25 01:00:00',
 '2027-03-28 01:00:00', '2027-10-31 01:00:00',
 '2028-03-26 01:00:00', '2028-10-29 01:00:00',
 '2029-03-25 01:00:00', '2029-10-28 01:00:00',
 '2030-03-31 01:00:00', '2030-10-27 01:00:00',
 '2031-03-30 01:00:00', '2031-10-26 01:00:00',
 '2032-03-28 01:00:00', '2032-10-31 01:00:00',
 '2033-03-27 01:00:00', '2033-10-30 01:00:00',
 '2034-03-26 01:00:00', '2034-10-29 01:00:00',
 '2035-03-25 01:00:00', '2035-10-28 01:00:00',
 '2036-03-30 01:00:00', '2036-10-26 01:00:00',
 '2037-03-29 01:00:00', '2037-10-25 01:00:00',
 '2038-03-28 01:00:00', '2038-10-31 01:00:00',
 '2039-03-27 01:00:00', '2039-10-30 01:00:00',
 '2040-03-25 01:00:00', '2040-10-28 01:00:00'
 ])

NYC_OFFSET_DST = 4
NYC_OFFSET     = 5

DUBLIN_OFFSET_DST = -1
DUBLIN_OFFSET = 0


class TimeZone:
    """
    Stores daylight savings cutoff information so UTC times can be translated to zone-specific times.
    Every `DateTimeNano` object holds a `TimeZone` object.
    All timezone-related conversions / fixups will be handled by the `TimeZone` class.

    Parameters
    ----------
    from_tz : str, defaults to None
    to_tz : str

    Attributes
    ----------
    _from_tz : str
        shorthand timezone string from the constructor - the timezone that the time originates from
    _dst_cutoffs : numpy.ndarray
        lookup array for converting times from constructor to UTC nano in GMT time
    _to_tz : str
        shorthand timezone string from the constructor - the timezone that the time will be displayed in
    _timezone_str
        Python-friendly timezone string used for displaying individual times.
        NOTE: This is actually a property, not a regular attribute.
    _dst_reverse : numpy.ndarray
        lookup array for DateTimeNano to display time in the correct timezone, accounting for daylight savings.
    _offset
        offset from GMT for display (non daylight savings)
    _fix_offset
        the offset from the timezone of origin

    Notes
    -----
    'UTC' is not a timezone, but accepted as an alias for GMT
    """
    valid_timezones = ('NYC', 'DUBLIN', 'GMT', 'UTC')
    timezone_long_strings = {
        'NYC'    : 'America/New_York',
        'DUBLIN' : 'Europe/Dublin',
        'GMT'    : 'GMT',
        'UTC'    : 'UTC'
    }
    long_to_short_timezone_names = {
        'America/New_York': 'NYC',
        'Europe/Dublin': 'DUBLIN',
        'UTC': 'UTC',
        'GMT': 'GMT'
    }
    tz_error_msg = f"If constructing from strings specify a timezone in from_tz keyword. Valid options: {valid_timezones}. Example: dtn = DateTimeNano(['2018-12-13 10:30:00'], from_tz='NYC')"

    #------------------------------------------------------------
    def __init__(self, from_tz: str = None, to_tz: str = 'NYC'):

        if from_tz is None:
            raise ValueError(self.tz_error_msg)

        # might not need these, hang on to them for now
        self._from_tz = from_tz
        self._to_tz = to_tz

        # get appropriate daylight savings dictionaries
        self._dst_cutoffs, self._fix_offset = self._init_from_tz(from_tz)
        self._dst_reverse, self._offset = self._init_to_tz(to_tz)

    #------------------------------------------------------------
    @classmethod
    def _init_from_tz(cls, from_tz):
        # TODO: as we add more timezone support, put into a dictionary
        if from_tz == 'NYC':
            _dst_cutoffs = DST_CUTOFFS_NYC
            _fix_offset = NYC_OFFSET
        elif from_tz == 'DUBLIN':
            _dst_cutoffs = DST_CUTOFFS_DUBLIN
            _fix_offset = DUBLIN_OFFSET
        elif from_tz in ('GMT', 'UTC'):
            _dst_cutoffs = None
            _fix_offset = 0
        else:
            raise ValueError(f"{from_tz} is not a valid entry for from_tz keyword. Valid options: {cls.valid_timezones}.")

        # fix_offset is different than display offset
        # fix_offset is only used in initial conversion to UTC
        return _dst_cutoffs, _fix_offset

    #------------------------------------------------------------
    @classmethod
    def _init_to_tz(cls, to_tz):
        '''
        Return daylight savings information, timezone string for correctly displaying the datetime
        based on the to_tz keyword in the constructor.
        '''
        # TODO: as we add more timezone support, put into a dictionary
        # probably dont need _timezone_str
        if to_tz == 'NYC':
            _dst_reverse = DST_REVERSE_NYC
            _timezone_offset = NYC_OFFSET

        elif to_tz == 'DUBLIN':
            _dst_reverse = DST_REVERSE_DUBLIN
            _timezone_offset = DUBLIN_OFFSET

        elif to_tz in ('GMT', 'UTC'):
            _dst_reverse = None
            _timezone_offset = 0

        else:
            raise ValueError(f"{to_tz} is not a valid entry for from_tz keyword. Valid options: {cls.valid_timezones}.")

        return _dst_reverse, _timezone_offset

    #------------------------------------------------------------
    @property
    def _timezone_str(self):
        return self.timezone_long_strings[self._to_tz]

    #------------------------------------------------------------
    def _set_timezone(self, tz):
        '''
        See DateTimeNano.set_timezone()
        '''
        self._dst_reverse, self._offset = self._init_to_tz(tz)
        self._to_tz = tz

    #------------------------------------------------------------
    def _mask_dst(self, arr, cutoffs=None):
        '''
        :param arr: int64 UTC nanoseconds
        :param cutoffs: an array containing daylight savings time starts/ends at midnight
                        possibly a reverse array for GMT that compensates for New York timezone (see DST_REVERSE_NYC)
        '''
        if cutoffs is None:
            cutoffs = self._dst_cutoffs
        if cutoffs is None:
            return zeros(len(arr), dtype=bool)

        #is_dst = (FastArray(np.searchsorted(DST_CUTOFFS, arr)) & 1).astype(bool)
        #is_dst = (rc.BinsToCutsBSearch(arr, cutoffs, 0) & 1).astype(bool)

        is_dst = (searchsorted(cutoffs, arr) & 1).astype(np.bool_)
        return is_dst

    #------------------------------------------------------------
    def _is_dst(self, arr):
        return self._mask_dst(arr, self._dst_reverse)

    #------------------------------------------------------------
    def _tz_offset(self, arr):
        if self._dst_reverse is None:
            result = zeros(len(arr), dtype=np.int32)

        else:
            is_dst = self._mask_dst(arr, self._dst_reverse)
            reg_offset = -1 * self._offset
            dst_offset = reg_offset + 1
            result = where(is_dst, dst_offset, reg_offset)

        return result

    def __repr__(self):
        return f"{type(self).__qualname__}(from_tz='{self._from_tz}', to_tz='{self._to_tz}')"

    def __eq__(self, other: 'TimeZone'):
        return \
            self.__class__ == other.__class__ and \
            self._from_tz == other._from_tz and \
            self._to_tz == other._to_tz

    #------------------------------------------------------------
    def copy(self):
        """A shallow copy of the TimeZone - all attributes are scalars or references 
        to constants.
        """
        new_tz = TimeZone(from_tz=self._from_tz, to_tz=self._to_tz)

        # other attributes may have been changed
        new_tz._dst_cutoffs = self._dst_cutoffs
        new_tz._fix_offset = self._fix_offset
        new_tz._dst_reverse = self._dst_reverse
        new_tz._offset = self._offset

        return new_tz

    #------------------------------------------------------------
    def fix_dst(self, arr, cutoffs=None):
        '''
        Called by DateTimeNano routines that need to adjust time for timezone.
        Also called by DateTimeNanoScalar

        Parameters:
        -----------
        arr     : underlying array of int64, UTC nanoseconds OR a scalar np.int64
        cutoffs : lookup array for daylight savings time cutoffs for the active timezone

        Notes:
        ------
        There is a difference in daylight savings fixup for Dublin timezone. The python 
        datetime.astimezone() routine works differently than fromutctimestamp(). Python datetime 
        may set a 'fold' attribute, indicating that the time is invalid, within an ambiguous daylight 
        savings hour.

        >>> import datetime
        >>> from dateutil import tz

        >>> zone = tz.gettz('Europe/Dublin')
        >>> pdt0 = datetime.datetime(2018, 10, 28, 1, 59, 0, tzinfo=zone)
        >>> pdt1 = datetime.datetime(2018, 10, 28, 2, 59, 0, tzinfo=zone)
        >>> dtn = DateTimeNano(['2018-10-28 01:59', '2018-10-28 02:59'], from_tz='DUBLIN', to_tz='DUBLIN')
        >>> utc = datetime.timezone.utc

        >>> pdt0.astimezone(utc)
        datetime.datetime(2018, 10, 28, 0, 59, tzinfo=datetime.timezone.utc)

        >>> pdt1.astimezone(utc)
        datetime.datetime(2018, 10, 28, 1, 59, tzinfo=datetime.timezone.utc)

        >>> dtn.astimezone('GMT')
        DateTimeNano([20181028 00:59:00.000000000, 20181028 02:59:00.000000000])

        '''
        if cutoffs is None:
            cutoffs = self._dst_reverse

        if cutoffs is None:
            return arr

        # get whether or not daylight savings
        is_dst = self._mask_dst(arr, cutoffs=cutoffs)
        arr = arr - (NANOS_PER_HOUR * self._offset)

        # scalar check
        if isinstance(is_dst, np.bool_):
            if is_dst:
                arr += NANOS_PER_HOUR
        else:
            arr[is_dst] += NANOS_PER_HOUR

        return arr

    #------------------------------------------------------------
    def to_utc(self, dtn, inv_mask=None):
        '''
        Called in the DateTimeNano constructor. If necessary, integer arrays of nanoseconds 
        are converted from their timezone of origin to UTC nanoseconds in GMT.
        Restores any invalids (0) from the original array.
        This differs from fix_dst() because it adds the offset to the array.
        '''
        if self._from_tz not in ('GMT', 'UTC'):
            #print('was not gmt or utc', self._from_tz)
            # create an invalid mask before adjusting
            if inv_mask is None:
                inv_mask = dtn == 0

            # adjust the times so they are in UTC nanoseconds
            is_dst = self._mask_dst(dtn, cutoffs=self._dst_cutoffs)

            # future optimization: offet might be zero - don't bother with the first addition
            dtn = dtn + (NANOS_PER_HOUR * self._fix_offset)
            dtn[is_dst] -= NANOS_PER_HOUR

            # restore invalid times
            putmask(dtn, inv_mask, 0)

        return dtn

#========================================================
class Calendar():
    '''
    *** not implemented
    Holds information regarding holidays, trade days, etc. depending on market/country.
    Every TimeZone object holds a Calendar object.
    '''

    def __init__(self):
        raise NotImplementedError

## flip timestring DST arrays to UTC nano ints
DST_CUTOFFS_NYC = rc.DateTimeStringToNanos(DST_CUTOFFS_NYC)
DST_REVERSE_NYC = rc.DateTimeStringToNanos(DST_REVERSE_NYC)

DST_CUTOFFS_DUBLIN = rc.DateTimeStringToNanos(DST_CUTOFFS_DUBLIN)
DST_REVERSE_DUBLIN = rc.DateTimeStringToNanos(DST_REVERSE_DUBLIN)

TypeRegister.TimeZone = TimeZone
TypeRegister.Calendar = Calendar
