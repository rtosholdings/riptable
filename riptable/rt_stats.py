__all__ = ['class_error', 'groupScatter', 'linear_spline', 'lm', 'mae',
           'plotPrediction', 'plot_hist', 'r2', 'statx', 'winsorize',]

import riptable as rt
import numpy as np
from .rt_enum import TypeRegister

from .rt_fastarray import FastArray
from .rt_numpy import zeros

# extra classes
import pandas as pd
from bokeh.plotting import output_notebook, figure, show
from bokeh.models import Label

#TODO: Organize the functions in here better
#TODO: Add documentation
#TODO: Replace pandas dependence with display util
def statx(X):
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    pVals = [0.1, 1, 10, 25, 50, 75, 90, 99, 99.9]
    pValNames = ['min', '0.1%', '1%', '10%', '25%', '50%', '75%', '90%','99%','99.9%' , 'max' , 'Mean', 'StdDev', 'Count', 'NaN_Count']
    filt = np.isfinite(X)
    X_sub = X[filt]
    vals = np.percentile(X_sub,pVals)
    vals =np.insert(vals,0,np.min(X_sub))
    vals =np.append(vals,np.max(X_sub))
    vals = np.append(vals,np.mean(X_sub))
    vals = np.append(vals,np.std(X_sub))
    validcount = np.sum(filt)

    # plain count
    vals = np.append(vals, X.size)

    #nancount
    vals = np.append(vals, np.sum(np.isnan(X)))
    out = pd.DataFrame({'Stat' : pValNames ,'Value' : vals})
    return out

#NOTE: people might prefer name clip/bound?
def winsorize(Y, lb, ub):
    out = np.maximum(np.minimum(Y, ub), lb)
    return out


def plot_hist(Y, bins):
    df = pd.DataFrame({'Y': Y})
    df.hist(bins=bins)


def r2(X, Y):
    # why are these flipped back?
    xmean = np.mean(X)
    ymean = np.mean(Y)
    xycov = np.mean(np.multiply((X - xmean), (Y - ymean)))
    xvar = np.var(X)
    yvar = np.var(Y)
    r2_value = (xycov * xycov) / (xvar * yvar)
    return r2_value


def mae(X, Y):
    return np.nanmean(np.abs(X - Y))


def class_error(X, Y):
    X2 = np.round(X)
    Y2 = np.round(Y)
    class_err = np.sum(np.abs(X2 - Y2)) / X2.shape[0]
    return class_err


def lm(X, Y, intercept=True, removeNaN=True, displayStats=True):
    #TODO: Better display for stats
    X0 = X.copy()
    Y0 = Y.copy()
    if len(X0.shape) == 1:
        X0 = X0.reshape(X0.shape[0], 1)
    if len(Y0.shape) == 1:
        Y0 = Y0.reshape(Y0.shape[0], 1)
    if intercept:
        X0 = np.hstack([np.ones((X0.shape[0],1)), X0])

    if removeNaN:
        goodData = ~np.isnan(np.sum(X0, axis=1)) & ~np.isnan(np.sum(Y0, axis=1))
        X0 = X0[goodData, :]
        Y0 = Y0[goodData, :]
    VXX = np.matmul(np.transpose(X0), X0)
    VXY = np.matmul(np.transpose(X0), Y0)
    coeff = np.linalg.solve(VXX, VXY)

    if displayStats:
        YHat = np.matmul(X0, coeff)
        err = Y0 - YHat
        err = err.reshape(Y0.shape[0], Y0.shape[1])
        RMS = np.sqrt(np.mean(err * err))
        MAE = np.mean(np.abs(err))
        A = np.linalg.solve(VXX, np.transpose(X0))
        SE = np.sqrt(np.sum(A * A, axis=1)) * RMS
        tStat = coeff / SE.reshape(coeff.shape[0], 1)
        R = np.mean(YHat * Y0) / (np.std(YHat) * np.std(Y0))
        R2 = R * R

        print('R2 = ', R2)
        print('RMSE = ', RMS)
        print('MAE = ', MAE)
        print('tStats: ')
        print(tStat)
    return coeff


def linear_spline(X0, Y0, knots, display = True):
    X = X0.copy()
    Y = Y0.copy()
    X = X.reshape(X.shape[0], 1)
    Y = Y.reshape(Y.shape[0], 1)
    knots.sort()
    numKnots = len(knots)
    goodData = ~np.isnan(X) & ~np.isnan(Y)
    X = X[goodData]
    Y = Y[goodData]
    XAug = np.nan * np.zeros((X.shape[0], 2 + numKnots))
    XAug[:, 0] = np.ones_like(X)
    XAug[:, 1] = X

    for j in range(numKnots):
        XAug[:, 2 + j] = np.maximum(X - knots[j], 0.0)

    coeff = lm(XAug, Y, intercept=False, removeNaN=True, displayStats=False)
    YHat = np.matmul(XAug, coeff)

    X_uniq, X_idx = np.unique(X, return_index=True)
    YHat_uniq = YHat[X_idx]
    # idx = X_uniq <= ub & X_uniq >= lb

    output_notebook()
    # create a new plot
    p = figure(tools="pan,box_zoom,reset,save",
               title="example",
               x_axis_label='sections',
               y_axis_label='particles')

    p.circle(X_uniq.flatten(), YHat_uniq.flatten(), legend="y", fill_color="red", size=8)
    if display:
        show(p)
    return knots, coeff

#TODO: Make formatting aware of environment, e.g., Spyder, jupyter, etc. in groupScatter and plotPrediction
#NOTE: Can we use regPlot from seaborn
#won't display in jupyter lab
#better auto-detect bounds
#suppress nan warnings

def plotPrediction(X, Yhat, Y, N, lb=None, ub=None):

    if lb is None:
        lowerBound = np.nanmin(X)
    else:
        lowerBound = lb

    if lb is None:
        upperBound = np.nanmax(X)
    else:
        upperBound = ub

    goodFilt = np.isfinite(X) & np.isfinite(Y) & (X <= upperBound) & (X >= lowerBound) & \
               np.isfinite(Yhat) & np.isfinite(Y)

    dF = pd.DataFrame({'X': X[goodFilt], 'Y': Y[goodFilt], 'Yhat': Yhat[goodFilt]})
    dF.sort_values('X', inplace=True)
    dF.reset_index(drop=True, inplace=True)
    groupSize = np.floor(dF.shape[0] / N)
    dF['Bucket'] = np.int32(np.floor(dF.index.values / groupSize))
    out = dF.groupby('Bucket')[['X', 'Y', 'Yhat']].agg(np.nanmean)

    output_notebook()
    # output_file("test.html")
    p = figure(tools="pan,box_zoom,reset,save",
               title="example",
               x_axis_label='sections',
               y_axis_label='particles')

    p.circle(out.X, out.Y, legend="y", fill_color="red", size=8)
    p.circle(out.X, out.Yhat, legend="yhat", fill_color="blue", size=8)
    show(p)

    # xloc = np.min(out.X)
    # yloc = np.max(out.Y)
    # plt.text(xloc,yloc,r'$r^2 =$ {0:.2e}'.format(r2Val),fontsize=16)






def polyFit(x,y,d=1, filter=None):
    '''
    Fits a polynimial least-squares regression.
    
    Parameters
    ----------
    x : ndarray
        The x data points.
    y : ndarray
        The y data points.
    d : integer, optional
        The degree of regression (d=1 for standard linear regression).
    filter : ndarray, optional
        Boolean mask of data to include
    
    Returns
    -------
    c : ndarray
        The array of coefficients of the polynomial (constant term first).
    
    '''
    if filter is not None:
        x = x[filter]
        y = y[filter]
    A = np.vander(x, d+1, increasing=True)
    AtA = np.matmul( A.transpose(), A )
    Ay = np.matmul(A.transpose(), y)
    c = np.linalg.solve( AtA, Ay )
    return c


def groupScatter(*arg, **kwarg):
    '''
    This function has been moved to playa.stats.
    '''
    raise NotImplementedError('This function has been moved to playa.plot')

