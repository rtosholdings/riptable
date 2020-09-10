__all__ = ['SharedMemory'] 

import os
import sys
import warnings
from .rt_enum import TypeRegister


class SharedMemoryMeta(type):
    SM_FOLDER = '/dev/shm/'
    GLOBAL_NAME = 'Global_'
    SM_LIMIT = 2.5e10

    #------------------------------------------------------------------------------------
    def listdir(cls, sharename=None):
        '''
        (Linux only)
        returns list of all riptable shared memory files, or those with a specific sharename.
        '''
        if sys.platform != 'win32':
            prefix = cls.GLOBAL_NAME
            if sharename is not None:
                prefix += sharename + '!'
            plen = len(prefix)
            try:
                dir = os.listdir(cls.SM_FOLDER)
                return [ f[plen:] for f in dir if f.startswith(prefix) ]
            except Exception as e:
                return []
    #------------------------------------------------------------------------------------
    def clear(cls, sharename=None):
        '''
        (Linux only)
        :param sharename: (optional) specify a specific sharename

        Will remove all shared memory in /dev/shm, or all shared memory in the same directory with a specific share name.
        '''
        if sys.platform != 'win32':
            dir = cls._get_full_paths(sharename)
            for f in dir:
                os.remove(f)
        else:
            raise NotImplementedError(f"This feature is for Linux only. To clear shared memory in Windows, exit  your python session.")

    #------------------------------------------------------------------------------------
    def view(cls, sharename=''):
        '''
        (Linux only)
        TODO: add option for file info
        '''
        return cls.listdir(sharename)

    #------------------------------------------------------------------------------------
    def __repr__(cls):
        if sys.platform != 'win32':
            dir = cls.listdir()
            if len(dir) == 0:
                return "Shared memory is empty."

            dir = "\n".join([f'Shared memory {cls.size}:']+dir)
            return dir
        else:
            return 'Shared memory functions not currently supported on Windows.'
    
    #------------------------------------------------------------------------------------
    def _get_full_paths(cls, sharename=None):
        '''
        Get full file paths on linux shared memory. Necessary for deleting, checking file sizes.
        '''
        fullpaths = []
        try:
            dir = os.listdir(cls.SM_FOLDER)
            prefix = cls.GLOBAL_NAME
            if sharename is not None:
                prefix += sharename + '!'
            for f in dir:
                if f.startswith(prefix):
                    fullpaths.append(cls.SM_FOLDER+f)
        except Exception as e:
            pass
        return fullpaths

    #------------------------------------------------------------------------------------
    def _convert_bytes(cls, num):
        """
        this function will convert bytes to MB.... GB... etc
        """
        for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
            if num < 1024.0:
                return "%3.1f %s" % (num, x)
            num /= 1024.0

    
    #------------------------------------------------------------------------------------
    def _get_file_size(cls):
        dir = cls._get_full_paths()
        total_size = 0
        try:
            for f in dir:
                file_info = os.stat(f)
                total_size += file_info.st_size
        except Exception as e:
            pass
        return total_size

    #------------------------------------------------------------------------------------
    def check_shared_memory_limit(cls):
        if sys.platform != 'win32':
            size = cls._get_file_size()

            if size > cls.SM_LIMIT:
                readable = cls._convert_bytes(size)
                warnings.warn(f"!!!Shared memory is using {readable}. Consider using sm.clear() to remove.")

    #------------------------------------------------------------------------------------
    @property
    def size(cls):
        s = cls._get_file_size()
        return cls._convert_bytes(s)

class SharedMemory(metaclass=SharedMemoryMeta):
    pass


TypeRegister.SharedMemory = SharedMemory
