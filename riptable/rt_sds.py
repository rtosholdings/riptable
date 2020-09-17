__all__ = [
    # misc riptable utility funcs
    # public .sds methods
    'load_sds', 'save_sds', 'load_sds_mem', 'sds_tree', 'sds_info', 'sds_flatten', 'sds_concat',
    # private .sds methods
    'save_struct', 'container_from_filetype', 'compress_dataset_internal', 'decompress_dataset_internal',
    # debugging
    'SDSMakeDirsOn', 'SDSMakeDirsOff',
    'SDSVerboseOn', 'SDSVerboseOff',
    'SDSRebuildRootOn', 'SDSRebuildRootOff'
]

import os
# from pathlib import Path
import sys
import shutil
from typing import TYPE_CHECKING, List, Optional, Tuple, Union, Callable, Any
import warnings
#import logging

import numpy as np
import riptide_cpp as rc

from .rt_grouping import merge_cats
from .rt_numpy import zeros, ismember, arange, empty
from .rt_enum import (
    TypeRegister, INVALID_DICT, CategoryMode,
    CompressionMode, CompressionType, INVALID_FILE_CHARS, SDSFlag, SDSFileType, SDS_EXTENSION
)
from .rt_timers import utcnow
from .Utils.rt_metadata import MetaData
from .rt_utils import h5io_to_struct

if TYPE_CHECKING:
    from .rt_dataset import Dataset
    from .rt_struct import Struct
    from .rt_datetime import TimeSpan


# TODO: Replace these two COMPRESSION_TYPE_ constants with the CompressionType enum?

COMPRESSION_TYPE_NONE: int = 0
"""Designator for not using compression when saving data to SDS file."""
COMPRESSION_TYPE_ZSTD: int = 1
"""Designator for using ZSTD compression when saving data to SDS file."""


# TODO: Consider a global mode variable to be used when a directory is created
SDSMakeDirs = True
"""
When ``SDSMakeDirs`` is set to ``True``, ``rt_sds`` will call ``os.makedirs`` to make one or more subdirectories,
otherwise ``rt_sds`` will create a directory one level deep when there is nesting.
"""
SDSVerbose = False
"""
If enabled, the SDS module will include verbose logging.
"""
SDSRebuildRoot = False
"""
If enabled, SDS will rebuilds the root SDS file, ``_root.sds``, in the event that a dataset is saved to an existing
directory that was part of a previous ``Struct`` save.
"""

#-----------------------------------------------------------------------------------------
def SDSMakeDirsOn() -> None:
    """Enables ``SDSMakeDirs``."""
    global SDSMakeDirs
    SDSMakeDirs = True


def SDSMakeDirsOff() -> None:
    """Disables ``SDSMakeDirs``."""
    global SDSMakeDirs
    SDSMakeDirs = False


#-----------------------------------------------------------------------------------------
def SDSVerboseOn() -> None:
    """Enables ``SDSVerbose``."""
    global SDSVerbose
    SDSVerbose = True


def SDSVerboseOff() -> None:
    """Disables ``SDSVerbose``."""
    global SDSVerbose
    SDSVerbose = False


def VerbosePrint(s: str, time: bool = True) -> Optional['TimeSpan']:
    """
    Prints a message `s` and an optional timestamp to a stream, or to sys.stdout by default.
    If `time` is enabled, print the message with the current time and return the time.

    Parameters
    ----------
    s: str
        Message to print.
    time: bool
        Whether to include a timestamp (defaults True).

    Returns
    -------
    TimeSpan, optional

    See Also
    --------
    VerbosePrintElapsed: Prints a message along with the elapsed time.
    """
    if time:
        t = utcnow().hour_span
        print(f'{t} '+s)
        return t
    else:
        print(s)


def VerbosePrintElapsed(s: str, start: 'TimeSpan') -> None:
    """
    Prints a message `s` and includes the elapsed time since `start`.

    Parameters
    ----------
    s: str
        Message to print.
    start: TimeSpan
        The start time that will be used to calculate the elapsed time span.

    See Also
    --------
    VerbosePrint: Prints a message and an optional timestamp.
    """
    end = utcnow().hour_span
    print(f'{end} '+s+f' elapsed time: {(end-start).seconds[0]} seconds')


#-----------------------------------------------------------------------------------------
def SDSRebuildRootOn() -> None:
    """Enables ``SDSRebuildRoot``."""
    global SDSRebuildRoot
    print(f'Setting SDSRebuildRoot to True. Currently set to {SDSRebuildRoot}.')
    SDSRebuildRoot = True


def SDSRebuildRootOff() -> None:
    """Disables ``SDSRebuildRoot``."""
    global SDSRebuildRoot
    print(f'Setting SDSRebuildRoot to False. Currently set to {SDSRebuildRoot}.')
    SDSRebuildRoot = False


#-----------------------------------------------------------------------------------------
def sds_os(func: Callable, path: Union[str, bytes]) -> Any:
    """
    Wrapper around Python ``os`` and ``os.path`` functions that has SDS related logic (for instance
    verbose printing, if enabled in ``rt_sds``).

    Parameters
    ----------
    func: callable
        Python function to call that accepts a path-like parameter.
    path: str, or bytes
        The pathname that `func` operates on.

    Returns
    -------
    Any
        The return value(s) of the called callable.

    See Also
    --------
    SDSVerboseOn: Enables SDS Verbose mode.
    SDSVerboseOff: Disables SDS Verbose mode.
    VerbosePrintElapsed: Prints a message along with the elapsed time.
    VerbosePrint: Prints a message and an optional timestamp.
    """
    if SDSVerbose: start = VerbosePrint(f'calling {func.__name__} on {path}')
    d = func(path)
    if SDSVerbose: VerbosePrintElapsed(f'finished {func.__name__}', start)
    return d


def sds_isdir(path: Union[str, bytes]) -> bool:
    """
    Return ``True`` if pathname `path` refers to an existing directory.

    Parameters
    ----------
    path: str or bytes

    Returns
    -------
    bool

    See Also
    --------
    SDSVerboseOn: Enables SDS Verbose mode.
    SDSVerboseOff: Disables SDS Verbose mode.

    Notes
    -----
    If SDS verbose mode is toggled, verbose logging will appear.
    """
    return sds_os(os.path.isdir, path)


def sds_isfile(path: Union[str, bytes]) -> bool:
    """
    Return ``True`` if pathname `path` refers to an existing file.

    Parameters
    ----------
    path: str or bytes

    Returns
    -------
    bool

    See Also
    --------
    SDSVerboseOn: Enables SDS Verbose mode.
    SDSVerboseOff: Disables SDS Verbose mode.

    Notes
    -----
    If SDS verbose mode is toggled, verbose logging will appear.
    """
    return sds_os(os.path.isfile, path)


def sds_exists(path: Union[str, bytes]) -> bool:
    """
    Return ``True`` if pathname `path` exists.

    Parameters
    ----------
    path: str or bytes

    Returns
    -------
    bool

    See Also
    --------
    SDSVerboseOn: Enables SDS Verbose mode.
    SDSVerboseOff: Disables SDS Verbose mode.

    Notes
    -----
    If SDS verbose mode is toggled, verbose logging will appear.
    """
    return sds_os(os.path.exists, path)


def sds_listdir(path: Union[str, bytes]) -> List[str]:
    """
    Returns a list of pathnames referred to by directory path `path`.

    Parameters
    ----------
    path: str or bytes

    Returns
    -------
    list of str

    See Also
    --------
    SDSVerboseOn: Enables SDS Verbose mode.
    SDSVerboseOff: Disables SDS Verbose mode.

    Notes
    -----
    If SDS verbose mode is toggled, verbose logging will appear.
    """
    return sds_os(os.listdir, path)


def sds_endswith(path: Union[bytes, str, List[Union[bytes, str]]], add: bool = False) -> Union[bool, str, List[Union[str]]]:
    """
    Returns true if the pathname ends with SDS extension, ``.sds``, unless `add` is enabled then it returns
    the SDS pathname.

    Parameters
    ----------
    path: bytes, str, or list of str or bytes
        Pathname or list of pathnames to check if they are SDS file types.

    Returns
    -------
    bool, str, or list of str

    Notes
    -----
    Although a list of pathnames is accepted, the current implementation assumes these are SDS pathnames and returns
    them as is.
    """
    endswith = False
    if isinstance(path, bytes):
        path = path.decode()

    # user can pass in a list of filenames, right now we assume these end with SDS
    if isinstance(path,list) or path.lower().endswith(SDS_EXTENSION):
        endswith = True
    if endswith:
        if add:
            return path
        else:
            return True
    else:
        if add:
            return path+SDS_EXTENSION
        else:
            return False


#-----------------------------------------------------------------------------------------
def sds_flatten(rootpath: Union[str, bytes]) -> None:
    """
    `sds_flatten` brings all structs and nested structures in sub-directories into the main directory.

    Parameters
    ----------
    rootpath: str or bytes
        The pathname to the SDS root directory.

    Examples
    --------
    >>> sds_flatten(r'D:\junk\PYTHON_SDS')

    Notes
    -----
    - The current implementation of `sds_flatten`  crawls one subdirectory.
    - If a nested directory contains items that are not sds files, the flatten will be skipped for the nested directory.
    - If a there is a name conflict with items already in the base directory, the flatten will be skipped for the nested directory.
    - No files will be moved or renamed until all conflicts are checked.
    - If there were directories that couldn't be flattened, lists them at the end.
    """
    # TODO: make this recursive
    # check file permissions for final move before starting
    # any other safeguard to stop directories from being half-flattened
    dirlist = sds_listdir(rootpath)
    full_dirlist = [ rootpath + os.sep + fname for fname in dirlist ]
    nested = [ f for f in full_dirlist if sds_isdir(f) ]

    flatten_fail = []
    # main loop over all subdirectories
    for dirpath in nested:
        move_files = []
        dirlist = sds_listdir(dirpath)

        # make sure directory contains only .sds files
        # TODO: if nested subdirectory, recurse
        skip = False
        for fname in dirlist:
            if not sds_endswith(fname):
                skip = True
                break

            elif sds_isdir(dirpath+os.sep+fname):
                skip = True
                break
        if skip:
            warnings.warn(f'{dirpath} contained items that were not .sds files or directories. Could not flatten')
            flatten_fail.append(dirpath)
            continue

        oldroot = None
        # move the _root.sds file to the base directory
        if '_root.sds' in dirlist:
            # might have same name as directory, so use a temp name
            oldroot = dirpath + os.sep + '_root.sds'
            newroot = rootpath + os.sep + '_root.sds_temp'

            # BUG: get the final root path here, or before final save
            if sds_exists(newroot):
                warnings.warn(f'temp file {newroot} already in root container, could not flatten {dirpath}.')
                skip = True
            else:
                # add to final moving list
                move_files.append((oldroot, newroot))
                dirlist.remove('_root.sds')
        if skip:
            flatten_fail.append(dirpath)
            continue

        # strip .sds for renaming
        prefix = dirpath
        if sds_endswith(prefix):
            prefix = dirpath[:-4]
        prefix = prefix + '!'

        # move all .sds files to base directory
        for fname in dirlist:
            old = dirpath + os.sep + fname
            new = prefix + fname
            if sds_exists(new):
                warnings.warn('{new} was already found in base directory. Could not flatten {dirpath}.')
                skip = True
                break
            else:
                move_files.append((old,new))

        if skip:
            flatten_fail.append(dirpath)
            continue

        # move all .sds files to base directory
        for old, new in move_files:
            os.rename(old, new)

        shutil.rmtree(dirpath)
        # rename temp _root if necessary
        if oldroot is not None:
            # strip ! from end of prefix
            finalroot = prefix[:-1]+SDS_EXTENSION
            os.rename(newroot, finalroot)

    if len(flatten_fail)>0:
        print('Failed to flatten subdirectories:')
        for dname in flatten_fail:
            print(dname)


#-----------------------------------------------------------------------------------------
def _sds_path_multi(path, share=None, overwrite=True):
    '''
    Checks for existence of directory for saving multiple .sds files.
    If directory exists, asks user if it should be used (potentially overwriting existing .sds files inside)

    Returns True if okay to proceed with save.
    '''
    # path will never get checked/created if saving to shared memory
    if share is None:
        # prompt user for overwrite
        if sds_exists(path):
            if overwrite is False:
                prompt = f"{path} already exists. Possibly overwrite .sds files in directory? (subdirectories will remain intact) (y/n) "
                overwrite = False
                while(True):
                    choice = input(prompt)
                    if choice in ['Y', 'y']:
                        overwrite = True
                        break
                    elif choice in ['N', 'n']:
                        break
                if overwrite is False:
                    print(f"No file was saved.")
                    return False
                else:
                    pass
                    # don't remove the entire tree by default
                    #shutil.rmtree(path)
        else:
            # possible TODO: call chmod after this so permissions are correct
            # or maybe use os.umask before creating the directory?
            if SDSVerbose: VerbosePrint(f'calling makedirs')
            if SDSMakeDirs:
                os.makedirs(path)
            else:
                os.mkdir(path)
                #raise ValueError(f'Directory {path!r} does not exist.  SDSMakeDirs global variable must be set to auto create sub directories.')

    return True


#-----------------------------------------------------------------------------------------
def _sds_path_single(path, share=None, overwrite=True, name=None, append=None):
    '''
    Checks for existence of a single .sds file and possibly prompts user to overwrite.
    If the directory does not exist, it will be created for the final save.

    Returns full path for final save and status (True if okay to proceed with save)

    NOTE: TJD overwrite changed to True on Aug, 2019
    '''

    # TODO: add this routine to Dataset.save()
    if isinstance(path, bytes):
        path = path.decode()

    # possibly add extension
    if name is None:
        name = os.path.basename(os.path.normpath(path))
    else:
        name = _parse_nested_name(name)
        path = path+os.sep+name

    if sds_endswith(name):
        name = name[:-4]
    else:
        path += SDS_EXTENSION

    # if the user is appending to a file, overwrite is expected
    if append is not None:
        overwrite=True

    # TJD look at this path since it does os check on filepath
    if share is None:
        # if exists, let user know if file or directory
        exists_str = None
        if sds_isfile(path) is False:
            if sds_isdir(path):
                # for now, don't allow overwrite if name.sds is a directory
                exists_str = f'directory'
                raise TypeError(f"{path} already existed and was a {exists_str}.")
        else:
            exists_str = f'file'

        # prompt user for overwrite
        if exists_str is not None:
            prompt = f"{path} already exists. Overwrite? (y/n) "
            if overwrite is False:
                while(True):
                    choice = input(prompt)
                    if choice in 'Yy':
                        overwrite = True
                        break
                    elif choice in 'Nn':
                        break
            if overwrite is False:
                print(f"No file was saved.")
                return path, name, False
            else:
                # overwriting files is allowed, overwriting directories is not
                if sds_isdir(path):
                    shutil.rmtree(path)
                #TJD disabled this (consider flag to re-enable)
                ##print(f"Overwriting {exists_str} with {path}")

        # if the file/directory does not exist, possibly create the nested containing directory
        else:
            dir_end = len(os.path.basename(os.path.normpath(path)))
            if not sds_isdir(path[:-dir_end]):
                # don't make directory if empty string
                if len(path[:-dir_end]) > 0:
                    newpath = path[:-dir_end]
                    if SDSMakeDirs:
                        os.makedirs(newpath)
                    else:
                        os.mkdir(newpath)
                        #raise ValueError(f'Directory {newpath!r} does not exist.  SDSMakeDirs global variable must be set to auto create sub directories.')

    return path, name, True


#-----------------------------------------------------------------------------------------
def _sds_save_single(item, path, share=None, overwrite=True, compress=True, name=None, onefile=False, bandsize=None, append=None, complevel=None):
    '''
    Fast track for saving a single item in an .sds file. This will be called if someone saves
    a single array or FastArray subclass with the main save_sds() wrapper
    '''
    _, name, status = _sds_path_single(path, share=share, overwrite=overwrite, name=name, append=append)
    if status is False:
        return
    # wrap in struct, struct build meta will call item build meta if necessary
    item = TypeRegister.Struct({name:item})
    fileType = SDSFileType.Array
    _write_to_sds(item, path, name=None, compress=compress, sharename=share, fileType=fileType, onefile=onefile, bandsize=bandsize, append=append, complevel=complevel)

#-----------------------------------------------------------------------------------------
def _sds_load_single(meta, arrays, meta_tups, info=False):
    '''
    If an .sds file has a filetype SDSFileType.Array, it will be sent to this routine.
    Extracts the underlying array, and rebuilds any FastArray subclasses.
    '''
    item = TypeRegister.Struct._load_from_sds_meta_data(meta, arrays, meta_tups)
    item = list(item.values())[0]
    return item


#-----------------------------------------------------------------------------------------
def save_sds_uncompressed(
    filepath: Union[str, bytes],
    item: Union[np.ndarray, 'Dataset', 'Struct'],
    overwrite: bool = True,
    name: Optional[str] = None
) -> None:
    """
    Explicitly save an item without using compression.
    Equivalent to ``save_sds(filepath, item, compress=False)``.

    Parameters
    ----------
    filepath: str or bytes
        Path to directory for ``Struct``, path to ``.sds`` file for ``Dataset`` or array
        (where SDS extension will be added if necessary).
    item : Struct, Dataset, ndarray, or ndarray subclass
        The ``Struct``, ``Dataset``, ``ndarray``, or ``ndarray`` subclass to store.
    overwrite : bool
        If ``True``, do not prompt the user when overwriting an existing ``.sds`` file (mainly useful for ``Struct.save()``,
        which may call ``Dataset.save()`` multiple times) (default False).
    name : str, optional
        Name of the sds file (default None).

    Raises
    ------
    TypeError
        If `item` type cannot be saved.

    See Also
    --------
    save_sds: save datasets to the filename.

    """
    save_sds(filepath, item, compress=False, overwrite=overwrite, name=name)

#-----------------------------------------------------------------------------------------
def save_sds(
    filepath: Union[str, bytes],
    item: Union[np.ndarray, 'Dataset', 'Struct'],
    share: Optional[str] = None,
    compress: bool = True,
    overwrite: bool = True,
    name: Optional[str] = None,
    onefile: bool = False,
    bandsize: Optional[int] = None,
    append: Optional[str] = None,
    complevel: Optional[int] = None
) -> None:
    """
    Datasets and arrays will be saved into a single .sds file.
    Structs will create a directory of ``.sds`` files for potential nested structures.

    Parameters
    ----------
    filepath: str or bytes
        Path to directory for Struct, path to ``.sds`` file for Dataset/array (extension will be added if necessary).
    item : Struct, dataset, array, or array subclass
    share
        If the shared memory name is set, `item` will be saved to shared memory and NOT to disk. When shared memory
        is specified, a filename must be included in path. Only this will be used, the rest of the path will be discarded.
        For Windows make sure SE_CREATE_GLOBAL_NAME flag is set.
    compress : bool, default True
        Use compression when saving the file (shared memory is always saved uncompressed)
    overwrite : bool, default False
        If ``True``, do not prompt the user when overwriting an existing ``.sds`` file (mainly useful for ``Struct.save()``,
        which may call ``Dataset.save()`` multiple times)
    name : str, optional
        Name of the sds file.
    onefile : bool, default False
        If True will flatten() a nested struct before saving to make it one file.
    bandsize : int, optional
        If set to an integer greater than 10000 it will compress column datas every `bandsize` rows.
    append : str, optional
        If set to a string it will append to the file with the section name
    complevel : int, optional
        Compression level from 0 to 9. 2 (default) is average. 1 is faster, less compressed, 3 is slower, more compressed.

    Raises
    ------
    TypeError
        If `item` type cannot be saved

    Notes
    -----
    ``save()`` can also be called from a ``Struct`` or ``Dataset`` object.

    Examples
    --------
    Saving a Struct:

    >>> st = Struct({ \
        'a': Struct({ \
            'arr' : arange(10), \
            'a2'  : Dataset({ 'col1': arange(5) }) \
        }), \
        'b': Struct({ \
            'ds1' : Dataset({ 'ds1col': arange(6) }), \
            'ds2' : Dataset({ 'ds2col' : arange(7) }) \
        }), \
    })

    >>> st.tree()
    Struct
        ├──── a (Struct)
        │     ├──── arr int32 (10,) 4
        │     └──── a2 (Dataset)
        │           └──── col1 int32 (5,) 4
        └──── b (Struct)
            ├──── ds1 (Dataset)
            │     └──── ds1col int32 (6,) 4
            └──── ds2 (Dataset)
                    └──── ds2col int32 (7,) 4

    >>> save_sds(r'D:\\junk\\nested', st)
    >>> os.listdir(r'D:\\junk\\nested')
    _root.sds
    a!a2.sds
    a.sds
    b!ds1.sds
    b!ds2.sds

    Saving a Dataset:

    >>> ds = Dataset({'col_'+str(i):arange(5) for i in range(5)})
    >>> save_sds(r'D:\\junk\\test', ds)
    >>> os.listdir(r'D:\\junk')
    test.sds

    Saving an Array:

    >>> a = arange(100)
    >>> save_sds('D:\\junk\\test_arr', a)
    >>> os.listdir('D:\\junk')
    test_arr.sds

    Saving an Array Subclass:

    >>> c = Categorical(np.random.choice(['a','b','c'],500))
    >>> save_sds(r'D:\\junk\\cat', c)
    >>> os.listdir(r'D:\\junk')
    cat.sds
    """
    if isinstance(item, TypeRegister.Dataset):
        # keep name and path as-is, extension added later
        _, _, status = _sds_path_single(filepath, share=share, overwrite=overwrite, name=name, append=append)
        if status is False:
            return
        _write_to_sds(item, filepath, name=name, compress=compress, sharename=share, onefile=onefile, bandsize=bandsize, append=append, complevel=complevel)
        # if it exists, add this dataset to the folder's _root.sds file for future loads
        # maybe stick rebuild in _write_to_sds... this could also take care of single array saves
        _rebuild_rootfile(filepath, sharename=share, bandsize=bandsize, compress=compress, complevel=complevel)

    elif isinstance(item, TypeRegister.Struct):
        save_struct(item, filepath, name=name, sharename=share, overwrite=overwrite, compress=compress, onefile=onefile, bandsize=bandsize, complevel=complevel)

    # pack array into struct for save (will handle subclasses)
    elif isinstance(item, np.ndarray):
        _sds_save_single(item, filepath, share=share, compress=compress, overwrite=overwrite, name=name, complevel=complevel)

    else:
        raise TypeError(f'save_sds() can only save Structs, Datasets, or single arrays. Got {type(item)}')

#-----------------------------------------------------------------------------------------
def _sds_raw_info(filepath, share=None, sections=None, threads=None) -> List[tuple]:
    '''
    Returns
    -------
    a list of sds tuples
    '''
    if isinstance(filepath, bytes):
        filepath = filepath.decode()

    if not sds_endswith(filepath):
        if sds_isdir(filepath):
            # should we return an error also?
            filepath = filepath + os.sep + '_root.sds'
        else:
            filepath = filepath+SDS_EXTENSION
    else:
        if sds_isdir(filepath):
            raise ValueError(f'The filename {filepath} is a directory and ends with .sds so sds_info will not work.  Consider sds_tree(filepath) instead.')

    return decompress_dataset_internal(filepath, info=True, sections=sections, threads=threads)


#-----------------------------------------------------------------------------------------
def sds_dir(filepath: Union[str, bytes], share: Optional[str] = None) -> List[str]:
    """
    Returns list of ``Dataset`` or ``Struct`` item names as strings.
    Only returns top level item names of ``Struct`` directory.

    Parameters
    ----------
    filepath: str or bytes
        Path to directory for Struct, path to ``.sds`` file for Dataset/array (extension will be added if necessary).
    share
        If the shared memory name is set, the item will be saved to shared memory and NOT to disk. When shared memory
        is specified, a filename must be included in path. Only this will be used, the rest of the path will be discarded.

    Returns
    -------
    List of str

    Examples
    --------
    >>> ds = Dataset({'col_'+str(i):arange(5) for i in range(5)})
    >>> ds.save(r'D:\junk\test')
    >>> sds_dir(r'D:\junk\test')
    ['col_0', 'col_1', 'col_2', 'col_3', 'col_4']

    """
    dirlist = []
    firstsds = _sds_raw_info(filepath, share=share)
    meta, info, tups, fileheader = firstsds[0]
    for tup in tups:
        if tup[1] & SDSFlag.OriginalContainer:
            dirlist.append(tup[0].decode())
    return dirlist


#-----------------------------------------------------------------------------------------
def sds_info(filepath:str, share=None, sections=None, threads=None):

    # TODO: match the Matlab output (should it look the same, or print more information from array info?)
    return _sds_raw_info(filepath, share=share, sections=sections, threads=threads)


#-----------------------------------------------------------------------------------------
def sds_tree(filepath: str, threads: Optional[int] = None):
    '''
    Explicitly display a tree of data for .sds file or directory.
    Only loads info, not data.

    Examples:
    ---------

    >>> ds = Dataset({'col_'+str(i):arange(5) for i in range(5)})
    >>> ds.save(r'D:\junk\treeds')
    >>> sds_tree(r'D:\junk\treeds')
    treeds
     ├──── col_0 FA  (5,)  int32  i4
     ├──── col_1 FA  (5,)  int32  i4
     ├──── col_2 FA  (5,)  int32  i4
     ├──── col_3 FA  (5,)  int32  i4
     └──── col_4 FA  (5,)  int32  i4

    '''
    return _load_sds_internal(filepath, info=True)

#-----------------------------------------------------------------------------------------
def load_sds_mem(
    filepath:str,
    share:str,
    include: Optional[List[str]] = None,
    threads: Optional[int] = None,
    filter: Optional[np.ndarray] = None):
    '''
    Explicitly load data from shared memory.

    Parameters
    ----------
    filepath  : name of sds file or directory. if no .sds extension, _load_sds will look for _root.sds
                if no _root.sds is found, extension will be added and shared memory will be checked again.
    share : str
        shared memory name.  For Windows make sure SE_CREATE_GLOBAL_NAME flag is set.
    include : list of str, optional
    threads: int, optional, defaults to None
        how many threads to used
    filter: int array or bool array, optional, defaults to None

    Returns
    -------
    Struct, Dataset or array loaded from shared memory.

    Notes
    -----
    To load a single dataset that belongs to a struct, the extension must be included. Otherwise,
    the path is assumed to be a directory, and the entire Struct is loaded.
    '''
    return _load_sds_internal(filepath, share=share, include=include, threads=threads, filter=filter)

#-----------------------------------------------------------------------------------------
def _sds_dir_from_file_list(filenames, share=None, mustexist=False):
    # files might only be in shared memory
    if share is not None:
        raise NotImplementedError

    names = []
    single_sds = []
    badlist = []
    hasdir = False
    hasfile = False

    for f in filenames:
        if isinstance(f, bytes):
            f = f.decode()

        # TODO: this has been written before... pull the complete version from somewhere else
        if sds_isdir(f):
            names.append(f)
            hasdir = True
        else:
            fnew = sds_endswith(f, add=True)
            if sds_isfile(fnew):
                names.append(None)
                single_sds.append(fnew)
                hasfile = True
            else:
                badlist.append(fnew)
                if mustexist is True:
                    raise ValueError(f'Could not find file named {f} and mustexist is True.')
                else:
                    warnings.warn(f'Could not find file named {f}')

    # names is list of [ None, None, path, None ]
    # where None is a single file
    # single_sds is a list of fullpaths to single sds files
    # also returns flags to save a pass for calling function

    return names, single_sds, badlist, hasdir, hasfile

#-----------------------------------------------------------------------------------------
def _sds_load_from_list(files, single_sds, share=None, info=False, include=None, threads=None,
                       folders=None, filter=None, mustexist=False, sections=None):
    '''
    Called by load_sds(), for loading an explicit list of .sds files or directories.

    Parameters
    ----------
    files      : a list of [ None, None, directory/path, None ] where None is a placeholder for a single load to maintain file order.
    single_sds : a list of single .sds files
    share      : **not implemented
    info       : **not implemented
    include    : skips items in single .sds loads, or files in directory, behaves the same way as include keyword in load_sds()
    filter     : optional: boolean or fancy index filter (only rows in the filter will be added)
    sections   : optional: list of strings with sections to load (file must have been saved with append=)

    loads all loose .sds files in one decompress call
    for each directory, loads all .sds files inside in one decompress call

    For instance, if the list has 3 .sds files, and 2 directories, there will be 3 calls to rc.MultiDecompressFiles
    *future optimization: reduce this to 1 call for all files (gets tricky with nested structures)

    Returns
    -------
    list of datasets/structs/arrays.
    '''

    # load all loose sds files at the same time
    multiload = decompress_dataset_internal(single_sds, sharename=share, info=info, include=include, threads=threads, filter=filter, mustexist=mustexist, folders=folders, sections=sections)

    # check for autodetect on appended to or concat
    if isinstance( multiload, (tuple,list)):
        single_idx = 0
        for idx, f in enumerate(files):
            if f is None:
                files[idx] = _read_sds('', sharename=share, info=info, include=include, multiload=multiload[single_idx], filter=filter, mustexist=mustexist, sections=sections)
                single_idx += 1
            else:
                files[idx] = load_sds(f, share=share, info=info, include=include, threads=threads, filter=filter, mustexist=mustexist, sections=sections)
        return files
    else:
        return multiload
#-----------------------------------------------------------------------------------------
def _multistack_categoricals(spec_name, meta_list, indices, listcats, idx_cutoffs, unique_cutoffs):
    '''
    Call when loading multiple SDS files.
    Assumes meta_list is a list of dictionaries
    Returns a Categorical.
    '''
    if idx_cutoffs is not None and len(idx_cutoffs) > 1:
        # pull the firstkey and check for invalids
        firstindex = indices[idx_cutoffs[1:]-1]
        invalid = INVALID_DICT[firstindex.dtype.num]
        invalidsum = np.sum(firstindex==invalid)
        if invalidsum > 0 or indices[0]==invalid:
            warnings.warn(f'!! {invalidsum} Bad indices in categorical {spec_name}.  May have been gap filled.  Setting invalids to bin 0.')
            finvalidmask = indices == invalid
            indices[finvalidmask] =0

    if SDSVerbose: verbose_start = VerbosePrint(f'start reconstructing categorical {spec_name}')

    mode = { m['instance_vars']['mode'] for m in meta_list }
    if len(mode) != 1:
        raise TypeError(f'Categoricals had different modes! {list(mode)}')
    mode = CategoryMode(list(mode)[0])

    # for other properties, use the ones from the first item
    firstmeta = meta_list[0]
    ordered = firstmeta['instance_vars']['ordered']
    sort_display = firstmeta['instance_vars']['sort_gb']

    #------------------------- start rebuild here
    if mode in (CategoryMode.Dictionary, CategoryMode.IntEnum):
        base_index = None
        indices, listcats = merge_cats(indices, listcats, unique_cutoffs=unique_cutoffs, from_mapping=True, ordered=ordered, verbose=SDSVerbose)
        # TJD added check
        code=listcats[0][0]
        if isinstance(code, (int, np.integer)):
            # EXCPECT first value is string, and second is int
            newcats = dict(zip(listcats[1], listcats[0]))
        else:
            newcats = dict(zip(listcats[0], listcats[1]))

    else:
        base_index = { m['instance_vars']['base_index'] for m in meta_list }
        if len(base_index) != 1:
            raise TypeError(f'Categoricals had different base index {base_index}, cannot be stacked!')
        base_index = CategoryMode(list(base_index)[0])
        indices, newcats = merge_cats(indices, listcats, idx_cutoffs=idx_cutoffs, unique_cutoffs=unique_cutoffs, verbose=SDSVerbose, base_index=base_index, ordered=ordered)

    #newcats = TypeRegister.Grouping(indices, categories=newcats)
    newcats = TypeRegister.Grouping(indices, categories=newcats, _trusted=True, base_index=base_index, ordered=ordered, sort_display=sort_display)
    result = TypeRegister.Categorical(newcats)

    if SDSVerbose: VerbosePrintElapsed(f'finished reconstructing categorical {spec_name}', verbose_start)
    return result

#-----------------------------------------------------------------------------------------
def _multistack_onefile(arrays, nameflag_tup, cutoffs, meta, sep='/'):
    '''
    Interntal routine to stack any FastArray subclasses from a multistacked load.
    Returns dictionary of items.
    '''
    def _build_meta(metastrings):
        # this routine does not get meta for a categorical that appears later
        # we would have to go through all meta data to discover new items
        # get the first meta string (they are stacked)
        beststring = metastrings[0]
        if len(beststring)==0:
            for i in range(1,len(metastrings)):
                beststring = metastrings[i]
                if len(beststring) > 0:
                    break
        #print("buildmeta returning", beststring)
        return beststring

    # first pass, build an array of (colname, value, arrayflag)
    data_cutoffs = {}

    obj_array = np.empty(len(nameflag_tup), dtype='O')
    for i, (colname, flag) in enumerate(nameflag_tup):
        arr = arrays[i]
        colname = colname.decode()
        if flag & SDSFlag.Nested:
            # this is an entry point such as 'data/'
            # the array is a list of meta byte strings
            # now find the first valid meta string
            arr = _build_meta(arr)
        obj_array[i]=(colname, arr, flag)
        # regular item, or underlying array for FastArray subclass
        if flag & SDSFlag.OriginalContainer:
            # find sep char to get name
            pos = colname.rfind(sep)
            if pos >= 0:
                purename = colname[pos+1:]
                data_cutoffs[purename] = cutoffs[i]

    startname = ''
    # the root meta
    meta = _build_meta(meta)
    s = TypeRegister.Struct._flatten_undo(sep, 0, startname, obj_array, meta=meta, cutoffs=cutoffs)
    return s, data_cutoffs

#-----------------------------------------------------------------------------------------
def _multistack_items(arrays, meta_tups, cutoffs, meta):
    '''
    Interntal routine to stack any FastArray subclasses from a multistacked load.
    Returns dictionary of items.
    '''

    data = {}
    data_cutoffs = {}

    spec_items = {}
    spec_cutoffs = {}
    spec_meta = {}

    # list of non-categorical fastarray subclasses
    # all will be rebuilt using metadata from first item

    # loop over all meta data, the first definition gets stored in spec_meta
    for metadata in meta:
        if metadata is None or len(metadata) == 0:
            continue
        item_meta = MetaData(metadata).get('item_meta',[])
        for i_meta in item_meta:
            i_meta = MetaData(i_meta)
            m_list = spec_meta.setdefault(i_meta['name'], [])
            # TJD think this builds a list of all meta items for this column
            m_list.append(i_meta)

    for item_idx, tup in enumerate(meta_tups):
        itemname = tup[0].decode()
        itemenum = tup[1]

        # regular item, or underlying array for FastArray subclass
        if itemenum & SDSFlag.OriginalContainer:
            underlying = arrays[item_idx]
            # TODO: change this loop, or move elsewhere to NOT be hard-coded for categorical
            if itemname in spec_meta:
                metalist = spec_meta[itemname]
                i_meta = metalist[0]

                i_class = i_meta.itemclass
                # we can fix certain classes immediately, categoricals have to wait until all extra arrays loaded
                if not TypeRegister.is_binned_type(i_class):
                    underlying = i_class._load_from_sds_meta_data(itemname, underlying, [], i_meta)
                    del spec_meta[itemname]

            data[itemname] = underlying
            data_cutoffs[itemname] = cutoffs[item_idx]

        # auxilery item (categorical uniques, etc.)
        # python only
        else:
            spec_name = itemname[:itemname.find('!')]

            # each dictionary key in spec_arrays corresponds to an item in the original container
            # spec_items = {itemname: [arr1, arr2, arr3...]}
            spec_list = spec_items.setdefault(spec_name,[])
            spec_list.append(arrays[item_idx])

            # save cutoffs for categorical fixup
            spec_cutoffs_list = spec_cutoffs.setdefault(spec_name,[])
            spec_cutoffs_list.append(cutoffs[item_idx])

    # categoricals only
    for spec_name, meta_list in spec_meta.items():

        underlying = data.get(spec_name, None)
        # only rebuild if the underlying was loaded (may not have been in include list)
        if underlying is not None:
            listcats = spec_items[spec_name]
            idx_cutoffs = data_cutoffs[spec_name]
            unique_cutoffs = spec_cutoffs[spec_name]
            stacked_categorical = _multistack_categoricals(spec_name, meta_list, underlying, listcats, idx_cutoffs, unique_cutoffs)
            data[spec_name] = stacked_categorical

    return data, data_cutoffs

#-----------------------------------------------------------------------------------------
# internal routine to resolve metadata after a stacked load
def _stacked(filenames, result, folders):
    arrays, meta_tups, cutoffs, meta, loadedpaths, fileheader = result

    # check which files were loaded, warn with list of load failures
    found, _ = ismember(filenames, loadedpaths)
    if sum(found) != len(found):
        badlist = [ filenames[idx] for idx, f in enumerate(found) if not f ]
        warnings.warn(f'Error loading files: {badlist}')

    # fix special array subclasses and create a partitioned dataset
    if SDSVerbose: verbose_start = VerbosePrint(f'starting _multistack_items')

    isonefile = False
    # check here for onefile stacking vs normal stacking
    for i, mtup in enumerate(meta_tups):
        # check for the meta flag in a onefile, only onefile has this
        flag = mtup[1]
        if flag & SDSFlag.Meta:
            isonefile = True
            break
        if flag & SDSFlag.Nested and b'/' in mtup[0]:
            isonefile = True
            break

    if isonefile:
        data, allcutoffs = _multistack_onefile(arrays, meta_tups, cutoffs, meta, sep='/')
        # the data might be in a Struct
        if folders is not None and isinstance(data, TypeRegister.Struct):
            sep='/'
            # get the first foldername, has trailing slash
            folder = folders[0]
            while folder.find(sep) >= 0:
                pos = folder.find(sep)
                subfolder = folder[:pos]
                data=data[subfolder]
                folder = folder[pos+1:]

        if not isinstance(data, TypeRegister.Dataset):
            return data

        # fix cutoffs (trim down to just what we have)
        cutoffs = {colname: allcutoffs[colname] for colname in data}
        #print("**final cutoffs", cutoffs)
        #ds = TypeRegister.PDataset(data, cutoffs=cutoffs, filenames=loadedpaths)
        #return data, cutoffs

    else:
        data, cutoffs = _multistack_items(arrays, meta_tups, cutoffs, meta)

    if SDSVerbose: VerbosePrintElapsed(f'finished _multistack_items', verbose_start)
    ds = TypeRegister.PDataset(data, cutoffs=cutoffs, filenames=loadedpaths)

    return ds

#-----------------------------------------------------------------------------------------
def _convert_to_mask(filter):

    if not isinstance(filter, np.ndarray):
        filter = np.atleast_1d(filter)

    if isinstance(filter, np.ndarray) and filter.dtype.char != 'O':
        if filter.dtype.char == '?' and len(filter) != 0:
            # no more bool to fancy
            # filter = bool_to_fancy(filter)
            return filter
        else:
            if filter.dtype.num > 10:
                raise TypeError(f'The filter must be a numpy array of booleans or integers not {filter.dtype}.')
            # convert fancy index to bool
            if len(filter) > 0:
                maxval = np.max(filter)
                mask = zeros(maxval + 1, dtype=np.bool)
                mask[filter] = True
            else:
                mask = zeros(0, dtype=np.bool)
            return mask
    else:
        raise TypeError(f'The filter must be a numpy array of booleans or integers not {type(filter)}.')

#-----------------------------------------------------------------------------------------
def _stack_sds_files(filenames, share=None, info=False, include=None, folders=None, threads=None,
                    filter=None, mustexist=False, sections=None, reserve=0.0):
    '''
    Internal routine for a single list of filenames (datasets or structs) to be stacked.
    Called by stack_sds() and _stack_sds_dirs()
    Only supports datasets (no structs/nesting)
    Returns stacked dataset, will be pdataset when class has been implemented.
    '''

    # files that werent found were not passed in, but may have raised an error during multiload call
    if threads is not None: savethreads = rc.SetThreadWakeUp(threads)
    try:
        if len(filenames)==0:
            raise ValueError(f'MultiStack list was empty. No files existed in original list.')

        if SDSVerbose:
           verbose_start = VerbosePrint(f'calling rc.MultiStackFiles with {len(filenames)} files first: {filenames[0]} last: {filenames[-1]} include: {include}')

        # Always use boolean mask now
        mask = None
        if filter is not None:
            mask = _convert_to_mask(filter)
            filter = None

        result = rc.MultiStackFiles(filenames, include=include, folders=folders, filter=filter, mask=mask, mustexist=mustexist, sections=sections, reserve=reserve)

        if result is None:
            raise ValueError(f'There was a problem when trying to stack the files {filenames}.  No data was returned.')

        if SDSVerbose:
            VerbosePrintElapsed(f'finished rc.MultiStackFiles', verbose_start)
    except Exception as e:
        if threads is not None:
            rc.SetThreadWakeUp(savethreads)
        raise e
    if threads is not None:
        rc.SetThreadWakeUp(savethreads)

    return _stacked(filenames, result, folders)

#-----------------------------------------------------------------------------------------
def _stack_sds_dirs(filenames, share=None, info:bool=False, include=[], folders=[], sections=None, threads=None):
    '''
    Dictionary will be created for final `rc.MultiStackFiles` call.

    >>> dirs = ['D:\junk\foobar\20190201', 'D:\junk\foobar\20190204', 'D:\junk\foobar\20190205']
    >>> include = ['zz', 'qq', 'MM']
    >>> stack_sds( dirs, [] )

    This routine will build the following dictionary:

    include_dict = {
        'zz': ['D:\junk\foobar\20190201\zz.sds',
                'D:\junk\foobar\20190204\zz.sds',
                'D:\junk\foobar\20190205\zz.sds'],

        'qq': ['D:\junk\foobar\20190201\qq.sds',
                'D:\junk\foobar\20190204\qq.sds',
                'D:\junk\foobar\20190205\qq.sds'],

        'MM': ['D:\junk\foobar\20190201\MM.sds',
                'D:\junk\foobar\20190204\MM.sds',
                'D:\junk\foobar\20190205\MM.sds']
    }

    rc.MultiStackFiles will be called 3 times, on each of the dict values.
    A struct will be returned with three stacked Datasets ( or pdatasets when class has been implemented )

    Struct({
        'zz' : rc.MultiStack(include_dict['zz']),
        'qq' : rc.MultiStack(include_dict['qq']),
        'MM' : rc.MultiStack(include_dict['MM']),
    })

    '''
    # folders is the new way (include is really for column names now)
    # onefile will NOT take this path
    if folders is None:
        folders = include

    include_dict = {}
    for path in filenames:
        path = path + os.sep
        # treat include item as name of file within struct directory
        for inc in folders:
            if isinstance(inc, bytes):
                inc = inc.decode()
            inc = sds_endswith(inc, add=True)
            # don't put .sds in result item name
            name = inc[:-4]
            # don't check for existence of file, CPP loader will just skip it
            # pull ref to list, or create a new one
            inc_list = include_dict.setdefault(name, [])
            inc_list.append(path+inc)

    # build single dataset for each include item
    for inc, files in include_dict.items():
        include_dict[inc] = _stack_sds_files(files, share=share, info=info, sections=sections)

    # if just one item pop it
    #if len(include_dict) == 1:
    #    return include_dict.popitem()[1]

    # return all items in struct container
    return TypeRegister.Struct(include_dict)


#-----------------------------------------------------------------------------------------
def _load_sds_internal(
    filepath: str,
    share: Optional[str] = None,
    info: bool = False,
    include_all_sds: bool = False,
    include: Optional[List[str]] = None,
    stack: Optional[bool] = None,
    name: Optional[str] = None,
    threads: Optional[int] = None,
    folders: Optional[List[str]] = None,
    filter: Optional[np.ndarray] = None,
    mustexist: bool = False,
    sections: Optional[List[str]] = None,
    reserve: float = 0.0
):
    '''
    All explicity load_sds calls will be funneled into this routine.
    See docstrings for load_sds(), load_sds_mem(), sds_tree(), sds_info()
    '''
    if isinstance(include, (str, bytes)):
        include=[include]

    # All folder names have to end in /
    if folders is not None:
        if isinstance(folders, (str, bytes)):
            folders = [folders]
        if not isinstance(folders, list):
            raise ValueError(f'The folders kwarg must be a list of strings of dataset or struct names to include. {folders}')

    if stack is True:
        if isinstance(filepath, (str, bytes)):
            filepath=[filepath]

        files, sds_filelist, badlist, hasdir, hasfile = _sds_dir_from_file_list(filepath, share=share, mustexist=mustexist)

        if hasdir:
            if hasfile:
                raise TypeError(f'List of files must contain only directories or only .sds files. {filepath}')
            else:
                # only directories
                if include is None and folders is None:
                    raise ValueError(f'SDS stacking only implemented for Datasets. Must provide folders list if loading from multiple Struct directories.')
                return _stack_sds_dirs(files, share=share, info=info, include=include, folders=folders, sections=sections, threads=threads)
        else:
            # only files
            # TODO: Check if stacking with onefile (have to read file type of first file??)
            # TODO folders= must be preserved
            if folders is not None:
                # make sure all folders end with slash
                newfolders=[]
                for f in folders:
                    if not f.endswith('/'):
                        f = f + '/'
                    newfolders.append(f)
                folders=newfolders

                # assume onefile mode
                include_dict={}
                for f in folders:
                    fname = f[:-1]
                    include_dict[fname] = _stack_sds_files(sds_filelist, share=share, info=info, include=include, folders=[f], sections=sections, threads=threads, filter=filter, reserve=reserve)
                return TypeRegister.Struct(include_dict)

            return _stack_sds_files(sds_filelist, share=share, info=info, include=include, folders=folders, sections=sections, threads=threads,
                                   filter=filter, mustexist=mustexist, reserve=reserve)

    # not stacked
    # string-only operations until final load
    if isinstance(filepath, bytes):
        filepath = filepath.decode()

    # list of full filepaths provided
    elif isinstance(filepath, list):
        files, single_sds, _, _, _ = _sds_dir_from_file_list(filepath, mustexist=mustexist)
        return _sds_load_from_list(files, single_sds, share=share, info=info, include=include, threads=threads,
                                  filter=filter, mustexist=mustexist, folders=folders, sections=sections)

    if sds_endswith(filepath) or share is not None:
        # do not have a try
        result = _load_sds(filepath, sharename=share, info=info, include_all_sds=include_all_sds, include=include, name=name, stack=stack, threads=threads,
                          filter=filter, mustexist=mustexist, folders=folders, sections=sections)
    else:
        # change so only one routine (_load_sds) attempts to fix file
        # do this when shared memory load gets forked

        # try to load with extension and without (due to people naming directories with .sds extensions)
        try:
            result = _load_sds(filepath, sharename=share, info=info, include_all_sds=include_all_sds, include=include, name=name, stack=stack, threads=threads,
                               filter=filter, folders=folders, sections=sections)
            origerror = None
        except Exception:
            origerror = sys.exc_info()[1]

        if origerror is not None:

            # try again with extension
            filepath = filepath+SDS_EXTENSION

            try:
                result = _load_sds(filepath, sharename=share, info=info, include_all_sds=include_all_sds, include=include, name=name, stack=stack, threads=threads,
                                  filter=filter, folders=folders, sections=sections)
            except Exception:
                raise ValueError(f'Could not load item with filepath {filepath!r} and shared name {share!r}.  First error: {origerror!r}.  Second error {sys.exc_info()[1]}')

    if info:
        # tree from struct, otherwise single string from array
        if isinstance(result, TypeRegister.Struct):
            result = TypeRegister.Struct._info_tree(filepath, result)

    return result

#-----------------------------------------------------------------------------------------
def load_sds(
    filepath: Union[str, bytes],
    share: Optional[str] = None,
    info: bool = False,
    include_all_sds: bool = False,
    include: Optional[List[str]] = None,
    name: Optional[str] = None,
    threads: Optional[int] = None,
    stack: Optional[bool] = None,
    folders: Optional[List[str]] = None,
    sections: Optional[List[str]] = None,
    filter: Optional[np.ndarray] = None,
    mustexist: bool = False,
    verbose: bool = False,
    reserve: float = 0.0
) -> 'Struct':
    """
    Load a dataset from single ``.sds`` file or struct from directory of ``.sds`` files.

    When ``stack=True``, generic loader for a single ``.sds`` file or directory of multiple ``.sds`` files.

    Parameters
    ----------
    filepath : str or bytes
        Full path to file or directory.
        When `stack` is ``True`` can be list of ``.sds`` files to stack
        When `stack` is ``True`` list of directories containing ``.sds`` files to stack (must also use kwarg `include`)
    share : str, optional
        The shared memory name. loader will check for dataset in shared memory first and if it's not there, the
        data (if the filepath is found on disk) will be loaded into the user's workspace AND shared memory.
        A sharename must be accompanied by a file name. The rest of a full path will be trimmed off internally.
        Defaults to None.  For Windows make sure SE_CREATE_GLOBAL_NAME flag is set.
    info : bool
        No item data will be loaded, the hierarchy will be displayed in a tree (defaults to False).
    include_all_sds : bool
        If ``True``, any extra files in saved struct's directory will be loaded into final struct (skips user prompt) (defaults to False).
    include : list of str, optional
        A list of strings of which columns to load, e.g. ``['Ask','Bid']``.
        When `stack` is ``True`` and directories passed, list of filenames to stack across each directory (defaults to None).
    name : str, optional
        Optionally specify the name of the struct being loaded. This might be different than directory (defaults to None).
    threads : int, optional
        How many threads to read, stack, and decompress with (defaults to None).
    stack : bool, optional
        Set to ``True`` to stack array data before loading into python (see docstring for `stack_sds`).
        Set to ``False`` when appending many files into one and want columns flattening.
        This parameter is not compatible with the `share` or `info` parameters (defaults to None).
    folders : list of str, optional
        A list of strings on which folders to include e.g., ``['zz/','xtra/']`` (must be saved with ``onefile=True``) (defaults to None).
    sections : list of str, optional
        A list of strings on which sections to include (must be saved with ``append="name"``) (defaults to None).
    filter : ndarray, optional
        Optional fancy index or boolean array. Does not work with ``stack=True``.
        Designed to read in contiguous sections; for example, ``filter=arange(10)`` to read first 10 elements (defaults to None).
    mustexist : bool
        Set to True to ensure that all files exist or raise an exception (defaults to False).
    verbose : bool
        Prints time related data to stdout (defaults to False).
    reserve : float
        When set greater than 0.0 and less than 1.0, this is how much extra room is reserved when stacking.
        If set to 0.10, it will allocate 10% more memory for future partitions.
        Defaults to 0.0.

    Returns
    -------
    Struct

    Notes
    -----
    When `stack` is ``True``:
    - columns with the same name must have matching types or upcastable types
    - bytestring widths will be fixed internally
    - numeric types will be upcast appropriately
    - missing columns will be filled with the invalid value for the column type

    Examples
    --------

    Stacking multiple files together while loading:

    >>> files = [ r'D:\dir1\ds1.sds' r'D:\dir2\ds1.sds' ]
    >>> load_sds(files, stack=True)
    #   col_0   col_1   col_2   col_3   col_4
    -   -----   -----   -----   -----   -----
    0    0.71    0.86    0.44    0.97    0.47
    1    0.89    0.40    0.10    0.94    0.66
    2    0.03    0.56    0.80    0.85    0.30

    Stacking multiple files together while loading, explicitly specifying the
    list of columns to be loaded.

    >>> files = [ r'D:\dir1\ds1.sds' r'D:\dir2\ds1.sds' ]
    >>> include = ['col_0', 'col_1', 'col_4']
    >>> load_sds(files, include=include, stack=True)
    #   col_0   col_1   col_4
    -   -----   -----   -----
    0    0.71    0.86    0.47
    1    0.89    0.40    0.66
    2    0.03    0.56    0.30

    Stacking multiple directories together while loading, explicitly specifying
    the list of `Dataset`s to load (from each directory, then stack together).

    >>> files = [ r'D:\dir1', r'D:\dir2' ]
    >>> include = [ 'ds1', 'ds2', 'ds3' ]
    >>> load_sds(files, include=include, stack=True)
    #   Name   Type      Size                0   1   2
    -   ----   -------   -----------------   -   -   -
    0   ds1    Dataset   20 rows x 10 cols
    1   ds2    Dataset   20 rows x 10 cols
    2   ds3    Dataset   20 rows x 10 cols

    See Also
    --------
    sds_tree
    sds_info
    """
    if verbose:
        SDSVerboseOn()
    else:
        SDSVerboseOff()

    if stack is True:
        if info:
            raise ValueError('sds: info cannot be set when stack=True')

        if share is not None:
            raise ValueError('sds: share cannot be set when stack =True')

    if stack is False:
        if reserve != 0.0:
            raise ValueError('sds: reserve cannot be set when stack=False')

    return _load_sds_internal(
        filepath,
        share=share,
        info=info,
        include_all_sds=include_all_sds,
        include=include,
        folders=folders,
        sections=sections,
        stack=stack,
        name=name,
        threads=threads,
        filter=filter,
        mustexist=mustexist,
        reserve=reserve)

#-----------------------------------------------------------------------------------------
def _make_zero_length(sdsresult):
    '''
    Internal routine that walks each array returned and creates a 0 length version
    '''
    # we just read in first row and now we have to return arrays of 0 length in the first dim
    newlist=[]

    # SDS file returns 4 tuples
    # meta, arrays, (name, flags), infodict
    for sds in sdsresult:
        filetype = sds[3]['FileType']
        if filetype == SDSFileType.Dataset or filetype == SDSFileType.Array:
            arr=sds[1]
            tups=sds[2]
            newarr=[]
            for a, t in zip(arr,tups):
                # names with a bang are for categoricals (we dont touch them)
                if t[0].find(b'!') == -1:
                    # make zero length array for first dim
                    l=[*a.shape]
                    l[0]=0
                    a=empty(tuple(l), dtype=a.dtype)
                newarr.append(a)
            arr=tuple(newarr)
            # rebuild sds
            sds = (sds[0], arr, sds[2], sds[3])
        newlist.append(sds)
    return newlist


#-----------------------------------------------------------------------------------------
# TODO - PEP484 What type does sds_concat return?
def sds_concat(filenames: List[str], output: Optional[str] = None, include: List[str] = None):
    """
    Parameters
    ----------
    filenames : list of str
        List of fully qualified pathnames
    output : str, optional
        Single string of the filename to create (defaults to None).
    include : list of str, optional
        A list of strings indicating which columns to include in the load (currently not supported).
        Defaults to None.

    Returns
    -------
    A new file created with the name in `output`.  This output file has all the filenames appended.

    Raises
    ------
    ValueError
        If output filename is not specified.

    Notes
    -----
    The `include` parameter is not currently implemented.

    Examples
    --------
    >>> flist=['/nfs/file1.sds', '/nfs/file2.sds', '/nfs/file3.sds']
    >>> sds_concat(flist, output='/nfs/mydata/concattest.sds')
    >>> sds_load('/nfs/mydata/concattest.sds', stack=True)

    """
    if output is None:
        raise ValueError(f'The output kwarg must be specified and be a valid filename to create.')

    result = rc.MultiConcatFiles(filenames, output=output, include=include)


#-----------------------------------------------------------------------------------------
def decompress_dataset_internal(
    filename: Union[bytes, str, List[str]],
    mode: CompressionMode = CompressionMode.DecompressFile,
    sharename: Optional[Union[bytes, str]] = None,
    info: bool = False,
    include: Optional[Union[bytes, str, List[str]]] = None,
    stack: Optional[bool] = None,
    threads: Optional[int] = None,
    folders: Optional[List[str]] = None,
    sections: Optional[List[str]] = None,
    filter: Optional[np.ndarray] = None,
    mustexist: bool = False,
    goodfiles: Optional[List[str]] = None
) -> List[Tuple[bytes, List[np.ndarray], List[tuple]]]:
    """
    Parameters
    ----------
    filename : str, bytes, or list of str
        A string (or list of strings) of fully qualified path name, or shared memory location (e.g., ``Global\...``)
    mode : CompressionMode
        When set to `CompressionMode.Info`, tup2 is replaced with a tuple of numpy attributes (shape, dtype,
        flags, itemsize) (default CompressionMode).
    sharename : str, or bytes, optional
        Unique bytestring for shared memory location. Prevents mistakenly overwriting data in shared memory (defaults to None).
    include : str, bytes, or list of str
        Which items to include in the load. If items were omitted, tuples will still appear, but None will
        be loaded as their corresponding data (defaults to None).
    stack : bool, optional
        Set to ``True`` to stack array data before loading into python (see docstring for `stack_sds`).
        Set to ``False`` when appending many files into one and want columns flattening.
        Defaults to None.
    threads : int, optional
        How many threads to read, stack, and decompress with (defaults to None).
    info : boolean
        Instead of decompressing numpy arrays, return a summary of each one's contents (shape/dtype/itemsize/etc.)
    folders : str, bytes, or list of strings, optional
        When saving with ``onefile=True`` (will filter out only those subfolders) list of strings (defaults to None)
    filter : ndarray, optional
        A boolean or fancy index filter (only rows in the filter will be added) (defaults to None).
    mustexist : bool
        When true will raise exception if any file is missing.
    sections : list of str, optional
        List of strings with sections to load (file must have been saved with ``append=``) (defaults to None).
    goodfiles : list of str, optional
        Tuples of two objects (list of filenames, path the files came from) -- often from ``os.walk`` (defaults to None).

    Returns
    -------
    list of tuples, optional
        tup1: json metadata in a bytestring
        tup2: list of numpy arrays or tuple of (shape, dtype, flags, itemsize) if info mode
        tup3: list of tuples containing (itemname, SDSFlags bitmask) for all items in container (might not correspond with 2nd item's arrays)
        tup4: dictionary of file header meta data

    Raises
    ------
    ValueError
        If `include` is not a list of column names.
        If the result doesn't contain any data.
    """
    #-------------------------------------------
    def _add_sds_ext(filename):
        """If a filename does not exist or is not a file, and has no extension, add extension
        """
        try_add = False
        if sds_exists(filename):
            pass
        else:
            root, ext = os.path.splitext(filename)
            if len(ext) == 0:
                try_add = True
        if try_add:
            filename = sds_endswith(filename, add=True)
        return filename
    #-----------------------------------------------------------------------------------------
    def _include_as_dict(include):
        '''
        If include list is specified, converts to dictionary of names->None
        '''
        if include is None or isinstance(include,dict):
            pass
        else:
            include = {item:None for item in include}
        return include
    #-------------------------------------------

    # even if one string, convert to a list of one string
    if isinstance(filename, (str, bytes)):
        filename = [filename]

    # sharename still passed as bytes
    if isinstance(sharename, str):
        sharename = sharename.encode()

    if include is not None:
        if isinstance(include, (str, bytes)):
            include = [include]
        if not isinstance(include, list):
            raise ValueError(f'The include kwarg must be a list of column names to include. {include}')

    if info:
        mode = CompressionMode.Info

    #print('***filename',filename)
    #print('mode',mode)
    #print('sharename',sharename)
    #print('include',include)

    # until the low-level routine does this, put the final extension check here
    # all SDS loads will hit this block
    # user can also pass in a list of known good filenames
    if goodfiles is not None:
        flist = goodfiles[0]
        fullpath = goodfiles[1]

        # build a good dictionary to avoid checking for file existence
        gooddict={}
        for file in flist:
            gooddict[os.path.join(fullpath, file)]=True

        for pfname, fname in enumerate(filename):
            if gooddict.get(fname, False):
                filename[pfname] = fname
            else:
                filename[pfname] = _add_sds_ext(fname)
    else:
        for pfname, fname in enumerate(filename):
            filename[pfname] = _add_sds_ext(fname)

    # check mask
    #if mask is not None:
    #    if filter is not None:
    #        raise ValueError('Both "mask" and "filter" are set.  Only one can be set.')
    #    if not isinstance(mask, np.ndarray):
    #        mask = np.atleast_1d(mask)
    #    if isinstance(mask, np.ndarray) and mask.dtype.char != '?':
    #        pass
    zerofilter=False
    mask =  None
    # check filter
    if filter is not None:
        # Always use boolean mask now
        mask = _convert_to_mask(filter)
        filter = None
        if len(mask) == 0:
            # special handling of zero filters
            zerofilter=True
            warnings.warn(f'Zero length filter for sds detected.')

    if threads is not None: savethreads = rc.SetThreadWakeUp(threads)
    if sharename is None:
        try:
            # TODO:
            # When len(filename) > 1000 we need to loop over MultiDecompressFiles and read 1000 files at a time
            # to avoid hitting open file limits on various operating systems
            if SDSVerbose: verbose_start = VerbosePrint(f'calling rc.MultiDecompressFiles first: {filename[0]} last: {filename[-1]}')
            if stack is not False and info is False and len(filename)==1:
                # this path checks for just one file.  this file may have been appended to and so contains multple files inside
                # if we detect the appended file ('StackType'==1), we default to stacking unless a forced stack=False
                result = rc.MultiPossiblyStackFiles(filename, include=include, folders=folders, filter=filter, mask=mask, mustexist=mustexist, sections=sections)

                if result is None:
                    raise ValueError(f'No data was found in the file {filename} with the specified parameters.')

                # the length of the return result will tell use what type of file this was
                if len(result) > 0 and len(result[0]) != 4:
                    # this is an sds_concat file and we assume it is stacked
                    return _stacked(filename, result, folders)

                # TO BE DELETED WHEN PASSES TESTS
                ## arrays, meta_tups, cutoffs, meta, loadedpaths, fileheader_dict
                #try:
                #    # TODO: Future improvement for split style loading
                #    # reader checks the header and calls one of the two style loads
                #    result = rc.MultiStackFiles(filename, include=include, folders=folders, filter=filter, mustexist=mustexist, sections=sections)
                #    arrs, meta_tups, cutoffs, meta, loadedpaths, fileheader_dict = result
                #    if fileheader_dict['StackType'] == 1:
                #        # this is an appended file
                #        return _stacked(filename, result, folders)
                #    # else convert back to as if normal MultiDecomp was called
                #    result = ((meta[0], arrs, meta_tups, fileheader_dict),)
                #except Exception:
                #    # fallback to old style read
                #    result = rc.MultiDecompressFiles(filename, mode, include=include, folders=folders, sections=sections, filter=filter, mustexist=mustexist)

            else:
                # meta, arrs, meta_tups, fileheader_dict
                result = rc.MultiDecompressFiles(filename, mode, include=include, folders=folders, sections=sections, filter=filter, mask=mask, mustexist=mustexist)

            if SDSVerbose: VerbosePrintElapsed(f'finished rc.MultiDecompressFiles', verbose_start)
        except Exception as e:
            if threads is not None: rc.SetThreadWakeUp(savethreads)
            raise e
        if threads is not None: rc.SetThreadWakeUp(savethreads)
    else:
        # call it the old way for now
        filename = filename[0]
        if isinstance(filename, str):
            filename = filename.encode()
        if include is not None:
            include = _include_as_dict(include)

        try:
            if SDSVerbose: verbose_start = VerbosePrint(f'calling rc.DecompressFiles filename: {filename} sharename: {sharename}')
            # return a list of one to normalize return values
            result = [rc.DecompressFile(filename, mode, sharename, include=include, folders=folders, sections=sections, filter=filter, mustexist=mustexist)]
            if SDSVerbose: VerbosePrintElapsed(f'calling rc.MultiDecompressFiles filename: {filename} sharename: {sharename}', verbose_start)
        except Exception as e:
            if threads is not None: rc.SetThreadWakeUp(savethreads)
            raise e
        if threads is not None: rc.SetThreadWakeUp(savethreads)

    if zerofilter:
        # we just read in first row and now we have to return arrays of 0 length in the first dim
        result = _make_zero_length(result)

    return result


#------------------------------------------------------------------------------------
def compress_dataset_internal(
    filename: Union[str, bytes],
    metadata: bytes,
    listarrays: List[np.ndarray],
    meta_tups: Optional[List[Tuple[str, SDSFlag]]] = None,
    comptype: CompressionType = CompressionType.ZStd,
    complevel: Optional[int] = 2,
    fileType = 0,
    sharename=None,
    bandsize=None,
    append=None
) -> None:
    '''
    All SDS saves will hit this routine before the final call to ``riptable_cpp.CompressFile()``

    Parameters
    ----------
    filename : str or bytes
        Fully qualified filename (path has already been checked by save_sds wrapper)
    metadata  : bytes
        JSON metadata as a bytestring
    listarrays : list of numpy arrays
    meta_tups : Tuples of (itemname, SDSFlag) - see SDSFlag enum in rt_enum.py
    comptype  : CompressionType
        Specify the type of compression to use when saving the Dataset.
    complevel : int
        Compression level. 2 (default) is average. 1 is faster, less compressed, 3 is slower, more compressed.
    fileType : SDSFileType
        See SDSFileType in rt_enum.py - distinguishes between Struct, Dataset, Single item, or Matlab Table
    sharename : If provided, data will be saved (uncompressed) into shared memory. No file will be saved to disk.

    Returns
    -------
    None
    '''
    if complevel is None:
        complevel = 2
    if meta_tups is None:
        meta_tups = []

    if not isinstance(filename, bytes):
        filename = filename.encode()

    # until the low-level routine does this, put the final extension check here
    # all SDS saves will hit this block
    if not filename.endswith(SDS_EXTENSION.encode()):
        filename+=SDS_EXTENSION.encode()

    if not isinstance(metadata, bytes):
        metadata = metadata.encode()

    if not isinstance(listarrays, list):
        raise TypeError(f"Input must be list of numpy arrays. Got {type(listarrays)}")

    # TODO - a gateway check could go here

    if sharename is None:
        if SDSVerbose: print(f'calling rc.CompressFile {filename}')
        rc.CompressFile(filename, metadata, listarrays, meta_tups, comptype, complevel, fileType, bandsize=bandsize, section=append)
        if SDSVerbose: print(f'finished rc.CompressFile')
    else:
        if isinstance(sharename, str):
            sharename=sharename.encode()

        rc.CompressFile(filename, metadata, listarrays, meta_tups, comptype, complevel, fileType, sharename, bandsize=bandsize, section=append)


#------------------------------------------------------------------------------------
def container_from_filetype(filetype: SDSFileType) -> SDSFileType:
    """
    Returns the appropriate container class based on the ``SDSFileType`` enum saved in the SDS file header.
    Older files where the file type is not set will default to 0, and container will default to ``Struct``.

    Parameters
    ----------
    filetype: SDSFileType

    Returns
    -------
    SDSFileType
    """
    if filetype in (SDSFileType.Dataset, SDSFileType.Table):
        container_type = TypeRegister.Dataset
    else:
        container_type = TypeRegister.Struct
    return container_type

#------------------------------------------------------------------------------------
def _rebuild_rootfile(path, sharename=None, compress=True, bandsize=None, complevel=None):
    """If a dataset is saved to an existing directory that was part of a previous struct save,
    _root.sds file will be resaved to include the added dataset for future loads.
    """
    if not SDSRebuildRoot:
        return

    if isinstance(path, bytes):
        path = path.decode()

    # check for root file in directory
    rootpath = os.path.join(os.path.dirname(path),'_root.sds')
    if not os.path.isfile(rootpath):
        return

    if path.endswith('.sds'):
        path = path[:-4]

    # decompress only the _root.sds file, not the full schema
    meta, arrays, sdsflags, fileheader = decompress_dataset_internal(rootpath, sharename=sharename)[0]

    # check if the container already exists in the root file
    # if the new save has the same name as an array in root, array will be kept
    dsname = os.path.basename(path).encode()
    itemnames = { t[0]:True for t in sdsflags }
    exists = itemnames.get(dsname, False)

    # no rewrite
    if exists:
        return

    # add tuple with name, container flag to sds tuples
    flag = SDSFlag.Nested + SDSFlag.OriginalContainer
    sdsflags.append( tuple((dsname, flag)) )

    # containers save None as a placeholder
    arrays = list(arrays)
    arrays.append(None)

    comptype = CompressionType.ZStd if compress else CompressionType.Uncompressed

    # send raw data back to compressor in same format as original save
    try:
        compress_dataset_internal(rootpath, meta, arrays, meta_tups=sdsflags, comptype=comptype, complevel=complevel, fileType = fileheader['FileType'], sharename=sharename, bandsize=bandsize)
    except:
        warnings.warn(f'Could not add {dsname} to {rootpath}.')

#------------------------------------------------------------------------------------
def skeleton_from_meta_data(container_type, filepath, meta, arrays, meta_tups, file_header):

    # bug when nested struct doesn't have its own file (only has containers)
    # need to find a general way to address this for all SDS loads
    # general loader should also look for files that start with end of path
    # if a folder is:
    # st1 /
    #   st2!ds1.sds
    #   st2!ds2.sds
    #   ds3.sds
    # should be able to call load_sds(r'st1/st2')

    # also not sure if this will get hit when onefile = True
    # looks like meta data doesn't work for this
    # need to turn onefile info load into same meta as multiple file info load

    if not isinstance(meta, MetaData):
        meta = MetaData(meta)

    data = {}
    specitems = {}
    arr_idx = 0

    # store all sds container info in a separate struct
    # BUG: the columns are sorted later... these meta items are not in the meta data
    # sort appears to be checking number of columns first
    data['Name_'] = meta['name']
    data['Type_'] = container_type.__name__
    data['FilePath_'] = filepath
    data['MetaData_'] = meta
    # placeholder for shape
    data['Shape_'] = None

    num_infofields = len(data)

    def get_dtype(dtypenum, itemsize):
        # bytestrings
        if dtypenum == 18:
            return np.dtype('S'+str(itemsize))
        # unicode
        elif dtypenum == 19:
            return np.dtype('U'+str(itemsize // 4))
        else:
            return np.dtype(np.typeDict[dtypenum])

    def info_to_struct(f):
        def wrapper(item_tup, sds_tup, filepath):
            iteminfo = {}
            iteminfo['Name_'] = sds_tup[0].decode()
            iteminfo = f(iteminfo, item_tup, sds_tup, filepath)
            iteminfo['SDSFlags_'] = sds_tup[1]
            iteminfo = TypeRegister.Struct(iteminfo)
            #iteminfo.sds_info_on()
            return iteminfo
        return wrapper

    @info_to_struct
    def arrinfo_to_struct(arrinfo, array_tup, sds_tup, filepath):
        # also store container filepath so this array can be sniped
        # load_sds() can be sent the item name in 'include' keyword
        arrinfo['Type_'] = 'FastArray'
        arrinfo['Shape_'] = TypeRegister.FastArray(array_tup[0])
        arrinfo['Dtype_'] = get_dtype(array_tup[1], array_tup[3])
        arrinfo['NumpyFlags_'] = array_tup[2]
        arrinfo['Itemsize_'] = array_tup[3]
        arrinfo['FilePath_'] = filepath
        return arrinfo

    @info_to_struct
    def scalarinfo_to_struct(scinfo, scalar_tup, sds_tup, filepath):
        dtype = get_dtype(scalar_tup[1], scalar_tup[3])
        scinfo['Type_'] = dtype.type.__name__
        scinfo['FilePath_'] = filepath
        return scinfo

    # store to individually load this
    for idx, tup in enumerate(meta_tups):
        name = tup[0].decode()
        itemenum = tup[1]

        if itemenum & SDSFlag.OriginalContainer:
            if itemenum & SDSFlag.Scalar:
                #data['nScalars'] += 1
                data[name] = scalarinfo_to_struct(arrays[idx], tup, filepath)

            # nested container will add its own info
            elif itemenum & SDSFlag.Nested:
                pass
                #data['nContainers'] += 1
                # do this to maintain order
            else:
                #data['nArrays'] += 1
                # store array info tuple
                data[name] = arrinfo_to_struct(arrays[idx], tup, filepath)

        else:
            # don't add to main dict of items
            # later on, if has extra columns, check item meta data
            #data['nExtra'] += 1

            # save to extra dict?
            # add to this items array info?
            specname = name[:name.find('!')]
            # add tuple of extra array info to spec items dict
            # each special item will have a list of extra array info
            specitem = specitems.setdefault(specname,[])
            specitem.append( tuple((arrays[idx], tup)) )

    # add more info for special items

    try:
        for imeta in meta['item_meta']:
            imeta = MetaData(imeta)
            name = imeta['name']
            arrinfo = data[name]

            # change type
            # add info for extra arrays if present
            arrinfo['Type_'] = imeta['classname']
            arrinfo['MetaData_'] = imeta

            # special item types should probably have their own routine to do repair their info
            # similar to _load_from_sds_meta_data
            extra_arrays = specitems.get(name, None)
            if extra_arrays is not None:
                extrastart = len(name)+1
                # these are tuples of (array tup, sds tup)
                for extra in extra_arrays:
                    # trim prefix off
                    extraname = extra[1][0][extrastart:]
                    sdstup = tuple((extraname, extra[1][1]))
                    # generate same info for extra array, just as if it were in a container
                    arrinfo[extraname] = arrinfo_to_struct(extra[0], sdstup, filepath)
    except:
        # Tjd should print a warning or something here?  This happens with matlab saves?
        pass

    # get container shape
    if data['Type_'] == 'Dataset':
        # get nrows from first column (same for all in dataset)
        item = list(data.values())[num_infofields]
        nrows = item['Shape_'][0]
    else:
        nrows = 0
    ncols = len(data) - num_infofields
    data['Shape_'] = tuple((nrows, ncols))

    #data = TypeRegister.Struct(data)
    #data.sds_info_on()

    return data



#------------------------------------------------------------------------------------
def _init_root_container(path, name, sharename=None, info=False, include=None, threads=None):
    fullpath = path + os.path.sep + name + SDS_EXTENSION
    firstsds = decompress_dataset_internal(fullpath, sharename=sharename, info=info, include=include, threads=threads)
    meta, arrays, meta_tups, fileheader_dict = firstsds[0]
    container_type = container_from_filetype(fileheader_dict['FileType'])

    # possibly non-json meta data (Matlab, older SDS format)
    try:
        final_meta = MetaData(meta)
    except:
        final_meta = None

    if info:
        #root_struct = container_type._tree_from_sds_meta_data( meta, arrays, meta_tups, fileheader_dict)
        # pass in the path as path to the folder
        root_struct = skeleton_from_meta_data(container_type, path, meta, arrays, meta_tups, fileheader_dict )
    else:
        root_struct = container_type._load_from_sds_meta_data(meta, arrays, meta_tups, fileheader_dict)
    return root_struct, final_meta, meta_tups

#------------------------------------------------------------------------------------
def _include_extra_sds_files(schema, final_sort, include_all_sds=False, include=None):
    '''
    If additional .sds files are found in a directory, the user can optionally load all or individually.
    If additional files are loaded, the user can choose to rewrite the _root.sds file to always include these.

    Parameters
    ----------
    schema     : nested dictionary build based on the files in a saved Struct's directory
    final_sort : if a _root.sds file was found, a list of its item names
    '''
    include_extra = False
    extra_items = []
    # keep track of all items not in root struct
    for k, v in schema.items():
        if k not in final_sort:
            extra_items.append(k)

    # extra items found in root struct
    if len(extra_items) > 0:
        # skip prompt if flag is set
        if include_all_sds:
            include_extra = True
        else:
            # Change for Blair April 2019
            warnings.warn(f"Found extra .sds information for items {extra_items}.  They will not be included")
            #prompt = f"Found extra .sds information for items {extra_items}. Would you like to include any? (y/n/a) "
            #while(True):
            #    choice = input(prompt)
            #    if choice in 'Yy':
            #        include_extra = True
            #        break
            #    elif choice in 'Nn':
            #        print("No extra items will be included.")
            #        break
            #    elif choice in 'Aa':
            #        include_extra = True
            #        include_all_sds = True
            #        break

        # extra items will be appended to sortlist
        if include_extra:
            for item in extra_items:
                # fast track to include all extra files
                if include_all_sds:
                    print(f"Including {item}")
                    final_sort.append(item)
                # ask about each item until they set all flag or finish list
                else:
                    prompt = f"Include {item}? (y/n/a) "
                    while(True):
                        choice = input(prompt)
                        if choice in 'Yy':
                            final_sort.append(item)
                            break
                        elif choice in 'Nn':
                            del schema[item]
                            break
                        elif choice in 'Aa':
                            final_sort.append(item)
                            include_all_sds = True
                            break
        # exclude all extra files
        else:
            for item in extra_items:
                del schema[item]

    return include_extra

#------------------------------------------------------------------------------------
def _build_schema_shared(root_tups, sharename, prefix='', dirlist=[], threads=None):
    '''
    Returns nested schema and directory list for shared memory.
    Because shared memory has no directory call, .sds file info needs to be expanded to check for nested structures.

    Parameters
    ----------
    root_tups : list of tuples (itemname, SDSFlag)
    sharename : shared memory name
    prefix    : used to recursively build .sds file names for nested structures
    dirlist   : list gets passed through recursion to avoid generating in a separate pass
                after recursion, will be identical to directory list in non-sharedmemory load
    '''
    schema = {}

    for item in root_tups:
        itemname = item[0].decode()
        itemenum = item[1]
        if itemenum & SDSFlag.OriginalContainer and itemenum & SDSFlag.Nested:
            filename = prefix+itemname+SDS_EXTENSION
            # check for file for nested container
            try:
                # only pull info to check for nested containers
                # replace with some sort of shared memory specific dir() call
                firstsds = decompress_dataset_internal(filename, sharename=sharename, info=True, threads=threads)
                meta, arrays, meta_tups, fileheader = firstsds[0]

                # only structs can have nested containers
                if fileheader['FileType'] == SDSFileType.Struct:

                    # chain off prefix
                    prefix = prefix+itemname+'!'
                    schema[itemname], dirlist = _build_schema_shared(meta_tups, sharename, prefix, dirlist, threads)

                # wasn't a struct, stop the schema here
                else:
                    schema[itemname] = {}

                dirlist.append(filename)
            except:
                warnings.warn(f'Could not find {filename} in sharename {sharename}.')

    return schema, dirlist

#------------------------------------------------------------------------------------
def _load_sds_mem(path, name=None, sharename=None, info=False, include_all_sds=False, include=None, threads=None):
    '''
    Shared memory checks directory differently for different operating systems.
    In Linux, a regular directory listing is used.
    In Windows, a different mechanism needs to be written **not implemented.
    Split this into a separate routine for readability in main _load_sds routine.
    '''

    checkdir = False
    root_struct = None
    meta = None
    meta_tups = None

    if sys.platform != 'win32':
        # strip shared memory prefix
        dir = TypeRegister.SharedMemory.listdir(sharename)
        schema = _build_schema(path, dir)
        checkdir = True
    else:
        # for now, can only load struct from shared memory if a _root.sds file exists
        root_struct, meta, meta_tups = _init_root_container(path, '_root', sharename=sharename, info=info, include=include, threads=threads)
        schema, dir = _build_schema_shared(meta_tups, sharename, threads=threads)

    return dir, schema, checkdir, root_struct, meta, meta_tups


#------------------------------------------------------------------------------------
def _load_sds(
    path:str,
    name: Optional[str] = None,
    sharename: Optional[str] = None,
    info: bool = False,
    stack: bool = None,
    include_all_sds: bool = False,
    include: Optional[List[str]] = None,
    threads: Optional[int] = None,
    folders: Optional[List[str]] = None,
    mustexist: bool = False,
    sections: Optional[List[str]] = None,
    filter: Optional[np.ndarray] = None
) -> Union['Struct', 'Dataset']:
    '''
    Build a tree (nested dictionaries) using the SDS file names in the provided directory.
    If path is a file, it will be loaded directly.

    Rebuild containers and numpy arrays after possibly decompressing SDS files. Any nodes without
    SDS files are assumed to be Structs.

    Parameters
    ----------
    path : str
        Full path to root directory.
    name : str, optional
        Optionally specify the name of the struct being loaded. This might be different than directory,
        however the _build_schema routine will be able to pull it generically.

    Returns
    ------
    Struct or Dataset
        When a `Struct` is returned, it may have nested data from all SDS files.
    '''
    has_ext = sds_endswith(path, add=False)
    goodfiles = None

    # TODO: only use this routine for non-shared memory loads
    if sharename is None:
        checkdir = False
        # check file
        if sds_isfile(path):
            loadpath = path
        # check directory
        elif sds_isdir(path):
            checkdir = True
        else:
            # if has extension, remove and check for dir again
            if has_ext:
                loadpath = path[:-len(SDS_EXTENSION)]
                if sds_isdir(loadpath):
                    path = loadpath
                    checkdir = True
                else:
                    raise ValueError(f'Failed to load. {path} was not a file or directory and {loadpath} was not a directory.')
            # if no extension, add and check file again
            else:
                loadpath = path + SDS_EXTENSION
                if not sds_isfile(loadpath):
                    raise ValueError(f'Failed to load. {path} was not a file or directory and {loadpath} was not a file.')
        if not checkdir:
            return _read_sds(loadpath, sharename=sharename, info=info, include=include, stack=stack, threads=threads, folders=folders, sections=sections, filter=filter)
        # only directories will hit this
        # we can speed this up with os.walk
        # remember the file list, because we know all these files exist
        dir=[f for root, dirs, files in os.walk(path) for f in files]
        # old code --> sds_listdir(path)
        schema = _build_schema(path, dir, nodirs=True)

        # remember the file list because it can be large
        goodfiles = (dir, path)

        root_struct = None
        meta = None
        meta_tups = None

    else:
        # shared memory path
        if sys.platform != 'win32':
            # TJD this path needs to be tested more
            if has_ext:
                return _read_sds(path, sharename=sharename, info=info, include=include, stack=stack, sections=sections, threads=threads, filter=filter)
            dir, schema, checkdir, root_struct, meta, meta_tups = _load_sds_mem(path, name=name, sharename=sharename, info=info, include_all_sds=include_all_sds, include=include, threads=threads)
        else:
            # NOTE: windows shared memory does not support dataset nesting via a struct currently..
            # but it could with a little more work
            return _read_sds(path, sharename=sharename, info=info, include=include, stack=stack, threads=threads, folders=folders, sections=sections, filter=filter)

    # root struct still needs to be initialized - windows sharedmemory load has root struct already
    # linux has a normal directory listing from the file system
    if checkdir:
        if name is None:
            name = f'_root.sds'
            if name not in dir:
                # directories with SDS and no _root are pretty common, killing this warning
                #warnings.warn(f'Could not find _root.sds file. Loading files in {dir} into container struct.')
                root_struct = TypeRegister.Struct({})
                meta = None
            else:
                # build the initial struct from root sds
                del schema['_root']
                root_struct, meta, meta_tups = _init_root_container(path, '_root', sharename=sharename, info=info, include=include, threads=threads)
            file_prefix=None
        else:
            # tiers can be separated by /, but files will be named with !
            name = name.replace('/','!')
            if sds_endswith(name, add=False):
                name = name[:-4]
            # use name keyword to snipe one dataset or struct
            if name + SDS_EXTENSION in dir:
                root_struct, meta, meta_tups = _init_root_container(path, name, sharename=sharename, info=info, include=include, threads=threads)
                file_prefix = name
                name = name.split('!')
                # climb down tiers
                for tier in name:
                    schema = schema[tier]
            else:
                raise ValueError(f"Could not find .sds file for {name} in {path}")

    # TODO: write something to handle name keyword in shared memory
    else:
        file_prefix = None

    final_sort = None
    root_file_found = (meta is not None)

    # possibly load from extra files in directory
    include_extra = False
    if root_file_found:
        final_sort = _order_from_meta(root_struct, meta, meta_tups)
        # all items will be included if no root file was found
        if final_sort is not None:
            # check for extra files, see if user wants to include
            include_extra = _include_extra_sds_files(schema, final_sort, include_all_sds, include=include)

    # choose the correct recursive function (full load or just info)
    # the recursive function will crawl other structures, or dictionaries from tree
    if info:
        #build_func = _summary_from_schema
        nocrawl = str
    else:
        #build_func = _struct_from_schema
        nocrawl = np.ndarray

    #multiload = None
    multiload = []

    # load individual files
    # not supported for shared memory
    # include keyword behaves differently than with an individual file load, so take the less common path for that too
    if multiload is None or sharename is not None or include is not None:
        # ---- main load for entire directory
        for k, v in schema.items():
            if include is not None:
                if k not in include:
                    continue
            try:
                item = root_struct[k]
                # none indicates that the structure was initialized, but data hasn't been loaded from file
                # this helps preserve item order in struct
                if item is None:
                    #root_struct[k] = build_func(schema, path, dir, filename=file_prefix, root=k, sharename=sharename, include=include)
                    root_struct[k] = _sds_load_from_schema(schema, path, dir, filename=file_prefix, root=k, sharename=sharename, include=None, info=info, nocrawl=nocrawl, threads=threads)
                    #root_struct[k] = _sds_load_from_schema(schema, path, dir, filename=file_prefix, root=k, sharename=sharename, include=include, info=info, nocrawl=nocrawl, threads=threads)
                else:
                    warnings.warn(f"Found .sds file for item {k}, but was already in struct as {root_struct[k]}. Skipping .sds load.", stacklevel=2)
            except:
                #root_struct[k] = build_func(schema, path, dir, filename=file_prefix, root=k, sharename=sharename)
                root_struct[k] = _sds_load_from_schema(schema, path, dir, filename=file_prefix, root=k, sharename=sharename, include=None, info=info, nocrawl=nocrawl, threads=threads)
                #root_struct[k] = _sds_load_from_schema(schema, path, dir, filename=file_prefix, root=k, sharename=sharename, include=include, info=info, nocrawl=nocrawl, threads=threads)

    # in this branch, flip to multi-file load
    else:
        # first pass, collect all filepaths
        # TODO: fold this into one pass, store return index in some kind of nested dictionary?
        for k, v in schema.items():
            if include is not None:
                if k not in include:
                    continue
            try:
                item = root_struct[k]
                if item is None:
                    _ = _sds_load_from_schema(schema, path, dir, filename=file_prefix, root=k, sharename=sharename, include=None, info=info, nocrawl=nocrawl, multiload=multiload)
                else:
                    pass
            except:
                _ = _sds_load_from_schema(schema, path, dir, filename=file_prefix, root=k, sharename=sharename, include=None, info=info, nocrawl=nocrawl, multiload=multiload)
        # call multiload, loads all into list
        # NEW: pass in list of known good files
        multiload = decompress_dataset_internal(multiload, sharename=sharename, info=info, include=include, stack=stack, threads=threads, filter=filter, goodfiles=goodfiles)

        #if isinstance(multiload, tuple):
        #    multiload = [multiload]
        # second pass, build nested containers
        # fake python int pointer to index the order of loaded files, restore correct hierarchy
        multiload_idx = [0]
        for k, v in schema.items():
            if include is not None:
                if k not in include:
                    continue
            try:
                item = root_struct[k]
                if item is None:
                    root_struct[k] = _sds_load_from_schema(schema, path, dir, filename=file_prefix, root=k, sharename=sharename, include=None, info=info, nocrawl=nocrawl, multiload=multiload, multiload_idx=multiload_idx)
                else:
                    pass
            except:
                root_struct[k] = _sds_load_from_schema(schema, path, dir, filename=file_prefix, root=k, sharename=sharename, include=None, info=info, nocrawl=nocrawl, multiload=multiload, multiload_idx=multiload_idx)

    # if root file found for metadata, sort items in root struct
    # if no root file found, order will be same as directory order
    if root_file_found and final_sort is not None and include is None:
        # if any files from original list were not included in final struct, list them, remove from sort list
        missing = []
        for item in final_sort:
            rm_item = False
            try:
                v = root_struct[item]
            except:
                rm_item = True
            else:
                # initialized from root info, but no .sds file found - value will be None
                if v is None:
                    rm_item = True
            if rm_item:
                warn_missing = True
                if include is not None:
                    if rm_item not in include:
                        warn_missing = False
                if warn_missing:
                    warnings.warn(f"Could not load data for item {item}, file for this item may be missing.")
                missing.append(item)
        for item in missing:
            final_sort.remove(item)

        root_struct = root_struct[final_sort]
        # if extra files were added to root struct, optionally rebuild _root.sds
        # if all extra files were included, skip the prompt, but don't rewrite the _root.sds file
        if include_extra and not include_all_sds:
            prompt = f"Include extra items in root struct for future loads? (_root.sds will be rebuilt) (y/n) "
            while(True):
                choice = input(prompt)
                if choice in 'Yy':
                    _write_to_sds(root_struct, path, name='_root', compress=True, sharename=sharename)
                    break
                elif choice in 'Nn':
                    break

    return root_struct

#------------------------------------------------------------------------------------
def _sds_load_from_schema(schema, path, dir, filename=None, root=None, sharename=None, include=None, info=False, nocrawl=np.ndarray, multiload=None, multiload_idx=None, multiload_schema=None, threads=None):
    '''
    Recursive function for loading data or info from .sds directory.
    Nested structures are stored:

    Example:
    --------

    >>> st = Struct({ 'a': Struct({ 'arr' : arange(10),
                                    'a2'  : Dataset({ 'col1': arange(5) }) }),

                        'b': Struct({ 'ds1' : Dataset({ 'ds1col': arange(6) }),
                                    'ds2' : Dataset({ 'ds2col' : arange(7) }) }),
        })
    >>> st.tree()
    Struct
        ├──── a (Struct)
        │     ├──── arr int32 (10,) 4
        │     └──── a2 (Dataset)
        │           └──── col1 int32 (5,) 4
        └──── b (Struct)
            ├──── ds1 (Dataset)
            │     └──── ds1col int32 (6,) 4
            └──── ds2 (Dataset)
                    └──── ds2col int32 (7,) 4

    >>> st.save(r'D:\junk\morejunk')
    >>> os.listdir(r'D:\junk\morejunk')
    _root.sds
    a!a2.sds
    a.sds
    b!ds1.sds
    b!ds2.sds

    '''
    multiload_schema = {}
    schema = schema[root]
    if filename is not None:
        filename = filename + '!' + root
    else:
        # for root level items
        filename = root

    # set default container in case nested .sds file doesn't exist
    default_container = TypeRegister.Struct
    data = {}
    sds_file = filename+SDS_EXTENSION

    # check for file in directory list
    if sds_file in dir:
        fullpath = path + os.path.sep + sds_file
        if multiload is None:
            # load container or array
            data = _read_sds(fullpath, sharename=sharename, include=include, info=info)
        else:
            if multiload_idx is None:
                # add full path for final multiload call
                multiload.append(fullpath)
                # maybe add to a different schema so the second pass for final load can be reduced
                # will this save any time? it would reduce the amount of calls to 'in', but not much else
            else:
                # pass the preloaded data to final constructor
                data = _read_sds(fullpath, sharename=sharename, include=include, info=info, multiload=multiload[multiload_idx[0]])
                multiload_idx[0] += 1

    # only recurse/restore order for containers
    if not isinstance(data, nocrawl):
        # TJD Feb 2020 - this code is slow when many files in the directory > 10000+
        # TODO improve the speed of this
        for k in schema.keys():
            data[k] = _sds_load_from_schema(schema, path, dir, filename=filename, root=k, sharename=sharename, include=include, info=info, nocrawl=nocrawl, multiload=multiload, multiload_idx=multiload_idx, threads=threads)

        # nested structures might not have .sds files, flip to default container type (Struct)
        if multiload is None or multiload_idx is not None:
            if not isinstance(data, default_container):
                data = default_container(data)

    return data

#------------------------------------------------------------------------------------
def _order_from_meta(data, meta, meta_tups):
    '''
    Restore the order of container items based on meta data.
    Meta tuples with (itemname, SDSFlag) will be checked first.
    If there is a mismatch (possibly older SDS version), json meta data will be used.
    If meta data is not valid, order will not change.
    '''
    if meta_tups is None:
        return None

    numitems = len(data)
    success = False

    # first try with tuples only
    order = []
    for t in meta_tups:
        if t[1] & SDSFlag.OriginalContainer:
            order.append( t[0].decode() )

    # OLD: sds is no longer dependent on item_names in python meta data
    if len(order) != numitems:
        if isinstance(meta, MetaData):
            # items still might be missing
            try:
                order = meta['item_names']
            except:
                order = None
        else:
            order = None

    return order

#------------------------------------------------------------------------------------
def _build_schema(path, dir, share_prefix='', nodirs=False):
    '''
    :param path: Full path to root directory.
    ::

    Build a tree with nested dictionaries using SDS files names in provided directory. Leaf
    nodes will be empty dictionaries.

    :return schema: Nested dictionary.
    '''
    plen= len(share_prefix)

    schema = {}

    #fnames=[f for root, dirs, files in os.walk(f1) for f in files]

    # TJD check if caller stripped the directories already
    if nodirs:
        files = [ f[plen:-4] for f in dir if sds_endswith(f, add=False)]
    else:
        # also check if a directory
        # NOTE this runs too slow when there are many files because of all the isdir checks
        files = [ f[plen:-4] for f in dir if sds_endswith(f, add=False) and not sds_isdir(os.path.join(path,f))]

    for f in files:
        schema_p = schema

        while(True):
            sep_idx = f.find('!')
            if sep_idx != -1:
                node = f[:sep_idx]
                schema_p = schema_p.setdefault(node, {})
            else:
                node = f
                schema_p = schema_p.setdefault(node, {})
                break

            f = f[sep_idx+1:]

    return schema

def _parse_nested_name(name):
    '''
    For Struct/Dataset save when a name is provided. Turns the '/' into '!' for the correct
    file prefix.
    '''
    if isinstance(name, bytes):
        name = name.decode()
    name = name.replace('/','!')
    return name

#------------------------------------------------------------------------------------
def save_struct(data=None, path=None, sharename=None, name=None, overwrite=True, compress=True, onefile=False, bandsize=None, complevel=None):
    if isinstance(path, bytes):
        path = path.decode()

    has_nested_containers= data.has_nested_containers

    # only create a directory if the struct has nested containers
    if has_nested_containers and not onefile:

        # add this change when matlab save changes
        # to mirror the save, strip .sds if the user added the extension
        # .sds should only be on files, not directories
        #if path.endswith(SDS_EXTENSION):
        #    path = path[:-4]
        if _sds_path_multi(path, share=sharename, overwrite=overwrite):
            # all structs with nested containers will get a _root.sds (helps maintain order)
            rootname = '_root' if name is None else name.split('!')[0]
            _write_to_sds(data, path, name=rootname, compress=compress, sharename=sharename, onefile=onefile, bandsize=bandsize, complevel=complevel)
            _sds_from_tree(data, path, name=name, sharename=sharename, compress=compress)
        else:
            return

    # otherwise, save as a single .sds file, possibly overwrite
    else:
        path, name, status = _sds_path_single(path, share=sharename, overwrite=overwrite)
        if status is False:
            return

        if has_nested_containers and onefile:
            # we have nesting and onefile is true
            flatstruct = data.flatten()
            meta = flatstruct.metastring
            del flatstruct.metastring
            arrayflags = flatstruct.arrayflags
            del flatstruct.arrayflags
            arrays = [*flatstruct.values()]
            meta_tups = [(name.encode(), arrayflag) for name, arrayflag in zip(flatstruct.keys(), arrayflags)]
            filetype=SDSFileType.OneFile

        else:
            meta, arrays, meta_tups = data._build_sds_meta_data(name)
            meta = meta.string
            filetype=SDSFileType.Struct

        comptype = COMPRESSION_TYPE_ZSTD if compress else COMPRESSION_TYPE_NONE
        compress_dataset_internal(path, meta, arrays, sharename=sharename, meta_tups=meta_tups, comptype=comptype, fileType=filetype, bandsize=bandsize, complevel=complevel)

    # most structs will create a new directory when they save
    # possibly append to existing _root.sds file if this struct was appended to another
    if name is not None or onefile:
        if not onefile:
            # get first name in sequence like 'st1!nested1!ds2'
            rootname = name.split('!')[0]
            path = os.path.join(path, rootname)
        _rebuild_rootfile(path, sharename=sharename, compress=compress, bandsize=bandsize, complevel=complevel)

#------------------------------------------------------------------------------------
def _escape_filename(name: str, dname: str) -> Tuple[str, str]:
    '''
    Raises an error if invalid characters are found in dname.

    Not fully implemented.
    TODO: replace with escape string

    Parameters
    ----------
    name : str
        Full name of individual file - includes ! to separate tiers of riptable container classes.
    dname : str
        Leaf of file new container name to be checked/possibly escaped.

    Returns
    -------
    name : str
        Full name may be escaped.
    dname : str
        Name of container may be escaped.
    '''

    for invalid in INVALID_FILE_CHARS:
        if invalid in dname:
            raise ValueError(f"Invalid character {invalid} found in file name {name}.")

    return name, dname

#------------------------------------------------------------------------------------
def _read_sds(
    path:str,
    sharename: Optional[str] = None,
    info: bool = False,
    include: Optional[List[str]] = None,
    stack: bool = None,
    multiload=None,
    threads: Optional[int] = None,
    folders: Optional[List[str]] = None,
    sections=None,
    filter: Optional[np.ndarray] = None,
    mustexist: bool = False):
    """
    Wrapper around a single .sds file load. Will return the appropriate item type based on the file header.

    Parameters
    -----------
    path : str
        full path to .sds file or filename in shared memory (will always be trimmed to name.sds in shared)
    sharename : str, optional
        specify a shared memory name instead of loading from disk
    info : bool
        When True, array header information will be returned, stored in a struct.
    include : list of str, optional
        if not None, list of column names to selectively load
    folders : list of str, optional
        list of strings containing folder names. Only valid when file was saved with ``onefile=True``.
    """
    # even if one string, convert to a list of one string
    if isinstance(path, (str, bytes)):
        path = [path]

    specialappend = -1
    # firstsds is normalized - it is what SDS decompress return (always a list of tuples)
    if multiload is None:
        firstsds = decompress_dataset_internal(path, sharename=sharename, info=info, include=include, stack=stack, threads=threads, folders=folders, sections=sections, filter=filter, mustexist=mustexist)

        # check if file was concat
        if not isinstance(firstsds, (list,tuple)):
            return firstsds

        meta, arrs, meta_tups, fileheader_dict = firstsds[0]

        if len(firstsds) > 1:
            # check for a file that was appended that has a stack type
            # currently if sds_concat was called, StackType will be 1
            # if it was manually appended, then it will be 0
            # to be used in the future
            specialappend = fileheader_dict.get('StackType', 0)
    else:
        meta, arrs, meta_tups, fileheader_dict = multiload

    ftype = fileheader_dict.get('FileType', None)

    if ftype == SDSFileType.Array:
        # TODO: determine what to do with include list for single item - need to know filetype before entire load
        #if include is not None:
        #    warnings.warn(f'Found single item in .sds file, but include was not None: {include}. Ignoring include keyword, Loading single item.')
        #    include = None
        result = _sds_load_single(meta, arrs, meta_tups, info=info)

    elif ftype == SDSFileType.OneFile:
        if info:
            raise NotImplementedError(f"SDS info struct not yet supported for onefile saves.")
        return TypeRegister.Struct._from_sds_onefile(arrs, meta_tups, meta=meta, folders=folders)

    else:
        container_type = container_from_filetype(ftype)
        if info:
            loader = getattr(container_type, '_tree_from_sds_meta_data')
            # test skeleton struct
            return skeleton_from_meta_data(container_type, path, meta, arrs, meta_tups, fileheader_dict)
        else:
            loader = getattr(container_type, '_load_from_sds_meta_data')

        section_count = 1
        if multiload is None: section_count = len(firstsds)

        # since ver 4.5, sds can be appended to
        if len(path) == 1 and section_count > 1 :
            result = np.empty(section_count, dtype='O')
            offsets = np.empty(section_count, dtype=np.int64)
            sections = None
            # walk all possible datasets, structs
            for i, sds in enumerate(firstsds):
                meta, arrays, meta_tups, fileheader_dict = sds
                result[i] = loader(meta, arrays, meta_tups, fileheader_dict)
                # find which one has the Sections (this is random due to threading)
                offsets[i] = sds[3].get('SectionOffset', None)
                temp = sds[3].get('Sections', None)
                if temp:
                    sections = temp

            # sort by the order appended by
            sortorder = np.lexsort([offsets])
            result = result[sortorder]

            # we should always find the section names
            if sections:
                set_type = set()
                len_type = set()

                # now try a named Struct
                # this will work for one file that was appended with sections
                # if we have multiple files with sections appended we need another routine
                resultStruct = TypeRegister.Struct({})
                for s,r in zip(sections, result):
                    resultStruct[s] = r
                    set_type.add(type(r))
                    try:
                        len_type.add(len(r))
                    except Exception:
                        # col not found and is likely None
                        len_type.add(0)


                # check if all the section were the same type and same length
                if len(set_type) ==1 and len(len_type)==1:
                    only_type = set_type.pop()
                    if only_type == TypeRegister.Dataset:
                        # try to make one dataset
                        arr_dict = {}
                        counter = 0
                        fail = False
                        for ds in result:
                            for k,v in ds.items():
                                # try (small effort) for no name clashes
                                if k in arr_dict:
                                    k = k + '_'+ str(counter)
                                    # give up if still not unique
                                    if k in arr_dict:
                                        fail = True
                                        break
                                    counter =counter + 1
                                arr_dict[k]=v
                        # return a Dataset if we can
                        if not fail: return TypeRegister.Dataset(arr_dict)

                # try to return a Struct if no information loss
                if len(resultStruct) == len(result):
                    return resultStruct
        else:
            result = loader(meta, arrs, meta_tups, fileheader_dict)

    return result

#------------------------------------------------------------------------------------
def _write_to_sds(data, path, name=None, compress=True, sharename=None, fileType=None, onefile=False, bandsize=None, append=None, complevel=None):
    '''
    :param data: Struct/Dataset/Multiset (must have _build_sds_meta_data() method)
    :param name: Name of SDS file. Hierarchy separated by !
    :param path: Full path to directory where all SDS files are stored.

    Writes a data structure's metadata, python objects, and numpy arrays to an SDS file
    with optional compression.

    :return None:
    '''
    # what do we do about other containers/subclasses?
    if fileType is None:
        if isinstance(data, TypeRegister.Dataset):
            fileType = SDSFileType.Dataset
        else:
            fileType = SDSFileType.Struct

    # path IS the single file
    if name is None:
        dname = name
        fullpath = path
    else:
        # chop off end of string or use full string
        dname_idx = name.rfind('!') + 1
        dname = name[dname_idx:]
        name, dname = _escape_filename(name, dname)
        fullpath = path+os.path.sep+name

    comptype = CompressionType.ZStd if compress else CompressionType.Uncompressed

    if onefile:
        # NOTE: this routine is similar to savestruct
        # TODO: Reduce to one routine
        flatstruct = data.flatten()
        meta = flatstruct.metastring
        del flatstruct.metastring
        arrayflags = flatstruct.arrayflags
        del flatstruct.arrayflags

        arrays = [*flatstruct.values()]
        meta_tups = [(name.encode(), arrayflag) for name, arrayflag in zip(flatstruct.keys(), arrayflags)]
        fileType=SDSFileType.OneFile
        compress_dataset_internal(fullpath, meta, arrays, meta_tups=meta_tups, comptype=comptype, sharename=sharename, fileType=fileType, bandsize=bandsize, append=append, complevel=complevel)
    else:
        meta, arrays, meta_tups = data._build_sds_meta_data(name=dname)
        compress_dataset_internal(fullpath, meta.string, arrays, meta_tups=meta_tups, comptype=comptype, sharename=sharename, fileType=fileType, bandsize=bandsize, append=append, complevel=complevel)

#------------------------------------------------------------------------------------
def _sds_from_tree(data, path, name=None, compress=True, sharename=None):
    '''
    :param data: Struct/Dataset/Multiset
    :param path: Full path to directory where SDS files will be stored
    :param name: Name of possible SDS file. Hierarchy separated by !

    Recursively crawls through containers within a Struct, generating SDS files as necessary.
    If a container only holds other containers, no SDS file will be generated. When loaded, it
    it will be reconstructed as a Struct.

    :return None:
    '''

    # in shared memory, always generate .sds file, even for struct that only has containers
    needs_sds = (sharename is not None)
    if len(data) == 0:
        # empty container, like empty struct will also be written
        needs_sds = True

    for k, v in data.items():
        # flip hdf5.io objects to riptable containers
        try:
            if v.__module__ == 'hdf5.io':
                v = h5io_to_struct(v)
        except:
            pass

        # if the item is an riptable container class, crawl it
        if hasattr(v, 'items') and hasattr(v, '_build_sds_meta_data'):
            if name is None:
                new_name = k
            else:
                new_name = name+'!'+k

            _sds_from_tree(v, path, name=new_name, compress=compress, sharename=sharename)
            #needs_sds = True
        # if the item contains things other than containers (arrays, python objects), it needs an sds file
        else:
            needs_sds = True

    if needs_sds:
        if name is not None:
            _write_to_sds(data, path, name=name, compress=compress, sharename=sharename)
