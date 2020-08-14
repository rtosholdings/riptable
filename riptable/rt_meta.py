__all__ = ['Item', 'Info', 'Doc', 'apply_schema', 'info', 'doc']

from typing import Optional, List

from .rt_struct import Struct
from .rt_fastarray import FastArray
from .rt_display import DisplayText

META_DICT = '_meta'
DOC_KEY = 'Doc'
DESCRIPTION_KEY = 'Description'
STEWARD_KEY = 'Steward'
TYPE_KEY = 'Type'
DETAIL_KEY = 'Detail'
CONTENTS_KEY = 'Contents'

NO_DESCRIPTION = '<no description>'
NO_STEWARD = '<no steward>'
NO_TYPE = '<none>'

NAME_DEFAULT_WIDTH = 4
DESCRIPTION_DEFAULT_WIDTH = 50
STEWARD_DEFAULT_WIDTH = 12
TYPE_STR_DEFAULT_WIDTH = 4

# ERROR KEYS
TYPE_MISMATCH = 'Type Mismatch'
EXTRA_COLUMN = 'Extra Column'
MISSING_COLUMN = 'Missing Column'

class Item:
    """Descriptive information for a data object.

    Parameters
    ----------
    name : str
        The name of the data object.
    type : str
        The type of the data object.
    description : str
        A description of the data object.
    steward : str
        The steward of the data object.
    """
    name : str
    """str: The name of the data object."""
    type : str
    """str: The type of the data object."""
    description : str
    """str: A description of the data object."""
    steward : str
    """steward: The steward of the data object."""

    def __init__(self, name: str, type: str, description: str, steward: str):
        self.name = name
        self.type = type
        self.description = description
        self.steward = steward


class Info:
    """A hierarchically structured container of descriptive information
    for a data object.
    """
    title = []
    """list: The title of the data object"""
    description : Optional[str] = None
    """str: The description of the data object."""
    steward : Optional[str] = None
    """str: The steward of the data object."""
    type : Optional[str] = None
    """str: The type of the data object."""
    detail = None
    """str: Detail about the data object."""
    items : Optional[List[Item]] = None
    """list of `Item`: For a :class:`~.rt_struct.Struct` or :class:`~.rt_dataset.Dataset`, the items contained within it."""
    def __init__(self):
        pass

    def _make_text(self):
        title_format = DisplayText.title_format
        header_format = DisplayText.header_format

        rows = []
        if self.title:
            rows += [title_format('{}'.format(self.title))]
            rows += [title_format('='*len(self.title))]
        if self.description:
            rows += [header_format('Description: ') + self.description]
        if self.steward:
            rows += [header_format('Steward: ') + self.steward]
        if self.type:
            rows += [header_format('Type: ') + self.type]
        if self.detail:
            rows += [header_format('Detail: ') + self.detail]
        if self.items:
            rows += [header_format('Contents:'), '']

            # Set column widths
            name_width = max(NAME_DEFAULT_WIDTH, max(len(item.name) for item in self.items))
            descrip_width = DESCRIPTION_DEFAULT_WIDTH
            steward_width = STEWARD_DEFAULT_WIDTH
            stype_width = max(TYPE_STR_DEFAULT_WIDTH, max(len(item.type) for item in self.items))

            # Add list header
            rows += [header_format("{: <{}}  {: <{}}  {: <{}}  {: <{}}".format(
                "Type", stype_width, "Name", name_width,
                "Description", descrip_width, "Steward", steward_width))]
            rows += [header_format("{}  {}  {}  {}".format(
                "-" * stype_width, "-" * name_width, "-" * descrip_width, "-" * steward_width))]

            # Add item rows
            for item in self.items:
                rows += ["{: <{}}  {}  {: <{}}  {: <{}}".format(
                    item.type, stype_width, title_format('{: <{}}'.format(item.name, name_width)),
                    item.description, descrip_width, item.steward, steward_width)]

        # Add a newline at the end if there is a title on top
        if self.title:
            rows += ['']

        return "\n".join(rows)

    def __str__(self):
        return DisplayText(self._make_text()).__str__()

    def __repr__(self):
        return DisplayText(self._make_text()).__repr__()

    def _repr_html_(self):
        return DisplayText(self._make_text())._repr_html_()


class Doc(Struct):
    """A document object containing metadata about a data object.

    Parameters
    ----------
    schema : dict
        See :meth:`apply_schema` for more information on the format of the
        dictionary.
    """

    _type = NO_TYPE
    _descrip = NO_DESCRIPTION
    _steward = NO_STEWARD
    _detail = None

    def __init__(self, schema):
        super().__init__()
        self._type = schema.get(TYPE_KEY)
        self._descrip = schema.get(DESCRIPTION_KEY, NO_DESCRIPTION)
        self._steward = schema.get(STEWARD_KEY, NO_STEWARD)
        self._detail = schema.get(DETAIL_KEY, None)
        schema_contents = schema.get(CONTENTS_KEY)
        if schema_contents:
            for key in schema_contents.keys():
                if self.is_valid_colname(key):
                    self[key] = Doc(schema_contents[key])

    def _as_info(self):
        info = Info()
        info.title = None
        info.description = self._descrip
        info.steward = self._steward
        info.type = self._type
        info.detail = self._detail
        info.items = []
        for name in self.keys():
            elem = self[name]
            info.items.append(Item(name, elem._type, elem._descrip,
                                   elem._steward))
        return info

    def __str__(self):
        return self._as_info().__str__()

    def __repr__(self):
        return self._as_info().__repr__()

    def _repr_html_(self):
        return self._as_info()._repr_html_()


def apply_schema(obj, schema: dict, doc: bool=True):
    """
    Apply a schema containing descriptive information recursively to the
    input data object.

    The schema should be in the form of a hierarchical dictionary, where
    for the data object, and recursively for each element it may contain,
    there is a descriptive dictionary with the following keys and values:
        * Type: 'Struct', 'Dataset', 'Multiset', 'FastArray', etc.
        * Description: a brief description of the data object
        * Steward: the name of the steward for that data object
        * Detail: any additional descriptive information
        * Contents: if the data object is a :class:`~.rt_struct.Struct`,
          :class:`~.rt_dataset.Dataset`, or :class:`~.rt_multiset.Multiset`, a
          recursively formed dictionary where there is a descriptive
          dictionary of this form associated with the name of each element
          contained by the data object.

    When the schema is applied to the data object, key/value pairs are set
    within the ``_meta`` dictionary attribute of the object and all of
    its elements, to enable subsequent retrieval of the descriptive
    information using the :meth:`.rt_struct.Struct.info` method or
    :meth:`.rt_struct.Struct.doc` property.

    In addition, during the schema application process, the contents and type
    of each data object is compared to the expectation of the schema, with
    any differences returned in the form of a dictionary.

    Parameters
    ----------
    obj : Struct or FastArray
        The data object to apply the schema information to.
    schema : dict
        A descriptive dictionary defining the schema that should apply to the
        data object and any elements it may contain.
    doc : bool
        Indicates whether to create and attach a :class:`Doc` to the object,
        so that the :meth:`doc` method may be run on the object.

    Returns
    -------
    res : dict
        Dictionary of deviations from the schema

    See Also
    --------
    :meth:`.rt_struct.Struct.apply_schema`
    """
    res = {}
    if isinstance(obj, (Struct, FastArray)):
        if not hasattr(obj, META_DICT):
            obj._meta = {}
        if doc:
            obj._meta[DOC_KEY] = Doc(schema)
        obj._meta[DESCRIPTION_KEY] = schema.get(DESCRIPTION_KEY, NO_DESCRIPTION)
        obj._meta[STEWARD_KEY] = schema.get(STEWARD_KEY, NO_STEWARD)
        obj._meta[DETAIL_KEY] = schema.get(DETAIL_KEY, None)
        stype = schema.get(TYPE_KEY)
        if stype and _type_str(obj) != stype:
            res[TYPE_MISMATCH] = "Type {} does not match schema type {}".\
                format(_type_str(obj), stype)
        schema_contents = schema.get(CONTENTS_KEY)
        if schema_contents:
            for key in obj.keys():
                elem_schema = schema_contents.get(key)
                if elem_schema:
                    elem_res = apply_schema(obj[key], elem_schema, False)
                    if elem_res:
                        res[key] = elem_res
                else:
                    res[EXTRA_COLUMN] = key
            for key in schema_contents.keys():
                if key not in obj.keys():
                    res[MISSING_COLUMN] = key
    return res

def _type_str(obj) -> str:
    """
    Return the string representation of an object's type.

    Parameters
    ----------
    obj : Any
        An object

    Returns
    -------
    str : str
        String representation of an object's type.
    """
    if isinstance(obj, FastArray):
        stype = obj.dtype.name
    else:
        stype = type(obj).__name__
    return stype

def info(obj, title=None) -> Info:
    """
    Return the :class:`Info` for the object, describing its contents.

    Parameters
    ----------
    obj : Any
        The object
    title : str
        The title to give the object, defaults to None.

    Returns
    -------
    info : Info
        Information about `obj`.
    """
    info = Info()
    info.title = title
    info.description = NO_DESCRIPTION
    info.steward = NO_STEWARD
    info.detail = None
    info.type = _type_str(obj)
    if hasattr(obj, META_DICT):
        info.description = obj._meta.get(DESCRIPTION_KEY, info.description)
        info.steward = obj._meta.get(STEWARD_KEY, info.steward)
        info.detail = obj._meta.get(DETAIL_KEY, None)
    if isinstance(obj, Struct):
        info.items = []
        for name in obj.keys():
            descrip = NO_DESCRIPTION
            steward = NO_STEWARD
            if hasattr(obj[name], META_DICT):
                descrip = obj[name]._meta.get(DESCRIPTION_KEY, descrip)
                steward = obj[name]._meta.get(STEWARD_KEY, steward)
            info.items.append(Item(name, _type_str(obj[name]), descrip, steward))
    return info

def doc(obj) -> Optional[Doc]:
    """
    Return the :class:`Doc` for the object, describing its contents.

    Parameters
    ----------
    obj : Any
        The object.

    Returns
    -------
    doc : Doc
        Returns a :class:`Doc` instance if  the object contains documentation
        metadata, otherwise None.
    """
    if hasattr(obj, META_DICT):
        if DOC_KEY in obj._meta:
            return obj._meta[DOC_KEY]
    return None
