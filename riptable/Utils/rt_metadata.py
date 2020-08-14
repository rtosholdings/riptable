__all__ = ['MetaData', 'meta_from_version', 'META_VERSION']

import json
from ..rt_enum import TypeId, TypeRegister, DisplayLength, CategoryMode

# global meta version, saved starting 12/17/2018 - anything prior will default to 0
META_VERSION = 0

version_dict = {
    # this dictionary will be checked whwnever a riptable class is being rebuilt during an SDS load
    # top level keys are the version number
    # next keys are the class's type id - see TypeId in enum
    0: {
        TypeId.Categorical: {
            # vars for container loader
            'name': 'Categorical',
            'typeid': TypeId.Categorical,
            'version': 0,
            # vars for additional arrays
            'colnames': [],
            'ncols': 0,
            # vars to rebuild the same categorical
            'instance_vars': {
                'mode': CategoryMode.StringArray,
                'base_index': 1,
                'ordered': False,
                'sorted': False,
            },
        },
        TypeId.Dataset: None,
        TypeId.Struct: None,
        TypeId.DateTimeNano: {
            'name': 'DateTimeNano',
            'typeid': TypeId.DateTimeNano,
            'ncols': 0,
            'version': 0,
            'instance_vars': {
                '_display_length': DisplayLength.Long,
                '_timezone_str': 'America/New York',
                '_to_tz': 'NYC',
            },
        },
        TypeId.TimeSpan: {
            'name': 'TimeSpan',
            'typeid': TypeId.TimeSpan,
            'ncols': 0,
            'version': 0,
            'instance_vars': {'_display_length': DisplayLength.Long},
        },
    }
}


def meta_from_version(cls, vnum):
    '''
    Returns a dictionary of meta data defaults.
    '''
    id = getattr(TypeId, cls.__name__)
    return version_dict[vnum][id]


class MetaData:

    default_dict = {'name': "", 'typeid': TypeId.Default}

    def __init__(self, metadict={}):
        self._dict = self.default_dict.copy()

        if isinstance(metadict, MetaData):
            self._dict = metadict._dict

        else:
            if isinstance(metadict, (bytes, str)):
                metadict = json.loads(metadict)

            for k, v in metadict.items():
                self._dict[k] = v

    @property
    def string(self):
        return json.dumps(self._dict)

    @property
    def dict(self):
        return self._dict

    @property
    def name(self):
        return self['name']

    @property
    def typeid(self):
        return self['typeid']

    @property
    def itemclass(self):
        """Starting 4/29/2019 item classes will be saved as strings in json meta data in `classname`.
        For backwards compatibility, will also check `typeid`. The TypeId class is an enum of ItemClass -> typeid.
        Both will lookup the items class in the TypeRegister, which holds classname -> itemclass.
        """
        try:
            classname = self['classname']
        except:
            classname = TypeId(self['typeid']).name
        return getattr(TypeRegister, classname)

    # pass these to dict
    # ------------------------------------------------------------
    def __getitem__(self, idx):
        return self._dict[idx]

    def __setitem__(self, idx, value):
        self._dict[idx] = value

    def get(self, key, default):
        return self._dict.get(key, default)

    def setdefault(self, k, v):
        self._dict.setdefault(k, v)

    def __repr__(self):
        return self._dict.__repr__()

    def __str__(self):
        return self._dict.__str__()

    # ------------------------------------------------------------
