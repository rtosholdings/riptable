__all__ = ['printh','write_dset_to_np','read_dset_from_np']

import os
import numpy as np

from IPython.display import display, HTML
from sotpath import path2platform
from riptable import Struct, Dataset


#------------------------------------------------------------------------------------------
def write_dset_to_np(ds: Dataset, outdir: str, fname: str) -> None:
    """
    Write the columns of a dataset to numpy binary files, one file per column in the specified directory.

    Parameters
    ----------
    ds : Dataset
        A Dataset to write out to disk.
    outdir : str
        The path to the folder where the output will be written.
    fname : str
        The name of the subdirectory to store the columns.

    See Also
    --------
    read_dset_from_np
    """
    os.makedirs(os.path.join(outdir, fname))
    fname = os.path.join(outdir, fname)
    fname = os.path.join(fname, 'columns')
    os.makedirs(path2platform(fname))
    for name, value in ds.items():
        fname_col = os.path.join(fname, str(name))
        np.save(path2platform(fname_col), value)


def read_dset_from_np(outdir: str, fname: str, mmap:bool=False) -> Dataset:
    """
    Read columns stored as numpy follows to a Dataset.

    Parameters
    ----------
    outdir is the path and fname is the name of the
    subdirectory containing the columns of the dataset
    set mmap = True for memmory mapping. Note this will
    allow quick loading, but has some latency cost elsewhere

    Returns
    -------
    Dataset
        The dataset read in from the specified folder.

    See Also
    --------
    write_dset_to_np
    """
    mmap_mode = None
    if mmap:
        mmap_mode = 'r'

    fname = os.path.join(outdir, fname)
    fname = os.path.join(fname, 'columns')
    col_dict = dict()
    col_names = os.listdir(path2platform(fname))
    for i in range(len(col_names)):
        fname_col = path2platform(os.path.join(fname, col_names[i]))
        curr_col_name = col_names[i].replace('.npy', '')
        col_dict[curr_col_name] = np.load(fname_col, mmap_mode=mmap_mode)
    return Dataset(col_dict)

#-----------------------------------------------------------------------------------------
def h5io_to_dataset(io):
    pass

#-----------------------------------------------------------------------------------------
def printh(data):
    """
    Allows jupyter lab/notebook to print multiple HTML renderings in the same output frame.

    Suppose you have three datasets: d1, d2, d3
    In one jupyter cell you could write:
    printh(d1)
    printh(d2)
    printh(d3)
    And all three would be displayed, versus the default, where only the last is shown.
    Will also work for anything else with a _repr_html_ method.

    You can also input a list of elements with _repr_html_ methods so that they display side by side.
    If the jupyter frame isn't wide enough, they'll just display below.

    Parameters
    ----------
    data : object or list of objects
        The object(s) to be rendered for display.
    """
    # multiple items
    if isinstance(data,list):
        html_frames = []
        for i,d in enumerate(data):
            if hasattr(d, '_repr_html_'):
                hfunc = d.__getattr__('_repr_html_')
                if callable(hfunc):
                    html_frames.append(hfunc())
                else:
                    print("_repr_html_ was not callable for item",str(i))
            else: print("No _repr_html_ found for item",str(i)+". Moving to next item.")
        if len(html_frames)>0:
            master_display ="""
                <html>
                <head>
                <style>
                ul.masterlist{
                    list-style-type:none;
                    padding:none;
                }
                ul.masterlist li{
                    margin-left: 30px;
                    display:inline-block;
                    float:left;
                }
                </style>
                </head>
                <body>
                <ul class='masterlist'>
            """
            for d in html_frames:
                master_display+="<li>"+d+"</li>"
            master_display+="""
            </ul>
            </body>
            </html>
            """
            display(HTML(master_display))
        else:
            print("No items with _repr_html_ found in list.")

    # single item
    else:
        if hasattr(data, '_repr_html_'):
            hfunc = data.__getattr__('_repr_html_')
            if callable(hfunc):
                display(HTML(hfunc()))
            else:
                print("_repr_html_ was not callable.")
        else:
            print("Input had no _repr_html_ method.")
