__all__ = ['IMatrix', ]

from numpy import array, empty
from .rt_numpy import vstack
from .rt_enum import TypeRegister

class IMatrix():
    '''
    Experimental class designed to take a Dataset and make a 2d matrix efficiently.
    It uses rt.vstack order='F' which uses rt.hstack plus np.reshape.

    The matrix is shaped so that it can be inserted back into the Dataset.

    Other Parameters
    ----------------
    dtype:
    order:
    colnames:

    See Also
    --------
    rt.vstack

    '''
    def __init__(self, ds, dtype=None, order='F', colnames=None):
        if not isinstance(ds, TypeRegister.Dataset):
            raise TypeError(f"The first argument must be a dataset")

        if not isinstance(colnames, list):
            raise TypeError(f"Pass in a list of column names such as ['Exch1','Exch2', 'Exch3']")
 
        self.rebuild(ds, dtype=dtype, order=order, colnames=colnames)

    @property
    def dataset(self):
        return self._dataset

    @property
    def imatrix(self):
        return self._imatrix

    def rebuild(self, ds=None, dtype=None, order='F', colnames=None):
        if ds is None:
            ds = self._dataset

        if colnames is None:
            colnames = [*ds.keys()]

        # make the matrix from the dataset
        self._imatrix=vstack([*ds[colnames].values()], dtype=dtype, order=order)

        # make a dataset from the matrix
        self._dataset = TypeRegister.Dataset({c:self._imatrix[:,i] for i,c in enumerate(colnames)})

    def __getitem__(self, fld):
        '''
        row slicing
        '''
        # apply the row mask or slice
        self._dataset = self._dataset[fld]

        # have to reconstruct the 2d array since it has been sliced
        self.rebuild()


    # -------------------------------------------------------
    # 2d arithmetic functions.
    def apply2d(self, func, name=None, showfilter=True):
        '''
        Parameters
        ----------
        func: function or method name of function

        Returns
        -------
        X and Y axis calculations

        '''
        imatrix = self._imatrix

        if not callable(func):
            func = getattr(array, func)

        if callable(func):
            if name is None:
                name = func.__name__
                name = str.capitalize(name)

            row_count, col_count = imatrix.shape

            # horizontal func
            #print("im0", imatrix.nansum())
            totalsY = func(imatrix, axis=1)  

            # possibly remove filtered top row
            if not showfilter:
                totalsY = totalsY[1:]

            # reserve an extra for the total of totals
            totalsX = empty(col_count+1, dtype=totalsY.dtype)

            # consider #imatrix.nansum(axis=0, out=totalsX)
            for i in range(col_count):
                arrslice = imatrix[:,i]

                # possibly skip over first value
                if not showfilter:
                    arrslice =arrslice[1:]

                totalsX[i] = func(arrslice)

            # calc total of totals - cell on far right and bottom
            totalsX[-1] = func(totalsY)

            return totalsX, totalsY

        return None, None

