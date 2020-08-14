__all__ = [ 'ItemContainer', ]
import numpy as np
import warnings
import re
from riptable.rt_enum import ColumnAttribute

ATTRIBUTE_LABEL = "Label"
ATTRIBUTE_SUMMARY = "Right"
ATTRIBUTE_FOOTER = "Footer"
ATTRIBUTE_MARGIN_COLUMN = "MarginColumn"
ATTRIBUTE_NUMBER_OF_FOOTER_ROWS = "NumberOfFooterRows"

class ItemAttribute():
    '''
    An attribute about an item which, in turn, contains attributes in the
    form of Python attributes, set and retrieved using setattr() and getattr()
    '''
    ATTRIB_EXCLUSION_LIST = 'copy'

    def __repr__(self, indent=2):
        result = self.__class__.__name__ + '\n'
        for k,v in self._attribs():
            result += ' '*indent + k + ': ' + str(v)
        result += '\n'
        return result

    def _attribs(self):
        '''
        Returns all attributes dynamically set for this ItemAttribute..

        NOTE: Add to the ATTRIB_EXCLUSION_LIST all method or property names statically
        added to ItemAttribute that don't begin with '_'.
        :return:
        '''
        return [(k, getattr(self, k)) for k in dir(self) if
                (not k.startswith('_') and k not in ItemAttribute.ATTRIB_EXCLUSION_LIST)]

    def copy(self):
        '''
        Performs a deep copy of the ItemAttribute, including all values
        of any dynamically added attributes.
        :return:
        '''
        attrib = ItemAttribute()
        for k, v in self._attribs():
            setattr(attrib, k, v.copy() if hasattr(v, 'copy') else v)
        return attrib


class ItemContainer():
    'Container for items in Struct -- all values are tuples with an attribute'

    def __init__(self, *args, **kwds):
        '''Initialize an IC

        '''
        self._items={}
        self._items.update(*args, **kwds)

    def __getitem__(self, key):
        return self._items[key]

    def __setitem__(self, key, value):
        'ic.__setitem__(i, y) <==> ic[i]=y'
        self._items[key] = value

    def __delitem__(self, key):
        'ic.__delitem__(y) <==> del ic[y]'
        # Deleting an existing item uses self.__map to find the link which is
        # then removed by updating the links in the predecessor and successor nodes.
        del self._items[key]

    #def __iter__(self):
    #    'od.__iter__() <==> iter(od)'
    #    # Traverse the linked list in order.
    #    root = self.__root
    #    curr = root[NEXT]
    #    while curr is not root:
    #        yield curr[KEY]
    #        curr = curr[NEXT]

    #def __reversed__(self):
    #    'od.__reversed__() <==> reversed(od)'
    #    # Traverse the linked list in reverse order.
    #    root = self.__root
    #    curr = root[PREV]
    #    while curr is not root:
    #        yield curr[KEY]
    #        curr = curr[PREV]

    #def __reduce__(self):
    #    'Return state information for pickling'
    #    items = [[k, self[k]] for k in self]
    #    tmp = self.__map, self.__root
    #    del self.__map, self.__root
    #    inst_dict = vars(self).copy()
    #    self.__map, self.__root = tmp
    #    if inst_dict:
    #        return (self.__class__, (items,), inst_dict)
    #    return self.__class__, (items,)

    def clear(self):
        self._items.clear()

    def __contains__(self,*args):
        return self._items.__contains__(*args)

    def __next__(self):
        return self._items.__next__()

    def __len__(self):
        return self._items.__len__()

    def __iter__(self):
        #return self._items.__iter__()
        return iter(self._items)

    def items(self):
        return self._items.items()

    def values(self):
        return self._items.values()

    def keys(self):
        # how to best do this?
        return list(self._items.keys())

    def setdefault(self, *args):
        return self._items.setdefault(*args)

    def update(self, *args):
        return self._items.update(*args)

    def pop(self, *args):
        return self._items.pop(*args)

    #setdefault = MutableMapping.setdefault
    #update = MutableMapping.update
    #pop = MutableMapping.pop
    #keys = MutableMapping.keys
    #values = MutableMapping.values
    #items = MutableMapping.items
    #__ne__ = MutableMapping.__ne__

    #def popitem(self, last=True):
    #    '''od.popitem() -> (k, v), return and remove a (key, value) pair.
    #    Pairs are returned in LIFO order if last is true or FIFO order if false.

    #    '''
    #    if not self:
    #        raise KeyError('dictionary is empty')
    #    key = next(reversed(self) if last else iter(self))
    #    value = self.pop(key)
    #    return key, value

    #-----------------------------------------
    def __repr__(self):
        'ic.__repr__() <==> repr(ic)'
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, self._items.items())

    #-----------------------------------------
    def copy_inplace(self, rowmask):
        '''
        inplace rowmask applied
        '''
        for v in self._items.values():
            # first item in tuple is the array
            arr = v[0]
            # preserve name when copying inplace
            name = arr.get_name()
            arr=arr[rowmask]
            arr.set_name(name)
            v[0] = arr

    #-----------------------------------------
    def copy(self, cols=None, deep=False):
        '''
        Returns a shallow copy of the item container.
        cols list can be provided for specific selection.
        '''

        newcontainer = ItemContainer()
        if cols is None:
            newcontainer._items = self._items.copy()
            for k,v in newcontainer._items.items():
                newcontainer._items[k] = v.copy()
        else:
            for k in cols:
                newcontainer._items[k] = self._items[k].copy()
        return newcontainer

    #-----------------------------------------
    def copy_apply(self, func, *args, cols=None):
        '''
        Returns a copy of the itemcontainer, applying a function to items before swapping them out in the new ItemContainer object.
        Used in Dataset row masking.
        '''
        
        newcontainer = ItemContainer()

        if cols is None:
            for k, v in self._items.items():
                # tuple copy
                v = v.copy()
                newcontainer._items[k] = v
                v[0] = func(v[0], *args)
        else:
            for k in cols:
                # tuple copy
                v= self._items[k].copy()
                newcontainer._items[k] = v
                v[0] = func(v[0], *args)

        return newcontainer

    #-----------------------------------------
    def apply(self, func, *args, cols=None):
        '''
        Performs a possibly inplace operation on items in the itemcontainer
        '''
        if cols is None:
            for v in self._items.values():
                func(v[0], *args)
        else:
            for k in cols:
                v = self._items[k]
                func(v[0], *args)

    #-----------------------------------------
    def __eq__(self, other):
        if isinstance(other, ItemContainer):
            return self._items == other._items
        return self._items == other

    def __ne__(self, other):
        if isinstance(other, ItemContainer):
            return self._items != other._items
        return self._items != other

    #def __del__(self):
    #    self._items.clear()                # eliminate cyclical references

    #-----------------------------------------
    def items_as_dict(self):
        '''
        Return dictionary of items without attributes.
        '''
        return {k:v[0] for k,v in self._items.items()}

    #-----------------------------------------
    def items_tolist(self):
        return [v[0] for v in self._items.values()]

    #-----------------------------------------
    def item_delete(self, key):
        del self._items[key]

    # -------------------------------------------------------
    def item_get_dict(self):
        ''' 
        return the underlying dict

        values are stored in the first tuple, attributes in the second tuple
        '''
        return self._items

    # -------------------------------------------------------
    def iter_values(self):
        '''
        This will yield the full values in _items dict (lists with item, attribute)
        '''
        for v in self._items.values():
            yield v

    # -------------------------------------------------------
    def item_get_value(self, key):
        ''' 
        return the value for the given key 

        NOTE: a good spot to put a counter for debugging
        '''
        return self._items[key][0]
    # -------------------------------------------------------
    def item_get_values(self, keylist):
        '''
        return list of value for the given key
        used for fast dataset slicing/copy with column selection
        '''
        return [ self.item_get_value(i) for i in keylist ]

    # -------------------------------------------------------
    def item_set_value(self, key, value, attr=None):
        # check if already exists...
        temp = [value, attr]
        v = self._items.setdefault(key, temp)
        if v is not temp:
            v[0] = value

    def item_set_value_internal(self, key, value):
        # no checks, go to dict
        self._items[key] = value

    # -------------------------------------------------------
    def item_get_attribute(self, key, attrib_name, default=None):
        '''
        Params
        ------
        Arg1: key: name of the item
        Arg2: attrib_name: name of the attribute

        Retrieves the value of the attribute previously assigned with item_set_attribute
        '''
        item = self._items.get(key, None)
        if item is None:
            return None
        attrib = item[1]
        if attrib is None:
            return None
        return getattr(attrib, attrib_name, default)


    # -------------------------------------------------------
    def _set_attribute(self, item, name, value):
        attrib = item[1]
        if attrib is None:
            attrib = ItemAttribute()
        setattr(attrib, name, value)
        item[1]=attrib

    # -------------------------------------------------------
    def item_set_attribute(self, key, attrib_name, attrib_value):
        '''
        Params
        ------
        Arg1: key: name of the item
        Arg2: attrib_name: name of the attribute
        Arg3: attrib_value: value of the attribute

        Attaches an attribute (name,value) pair to the item
        Any valid dictionary name and any object can be assigned.

        Note: see item_get_attribute to retrieve
        '''

        # check if already exists...
        if self.item_exists(key):
            self._set_attribute(self._items[key], attrib_name, attrib_value)
        else:
            raise KeyError(f"{key!r} does not already exist, thus cannot add attribute")

    # -------------------------------------------------------
    def item_get_len(self):
        return len(self._items)

    # -------------------------------------------------------
    def item_exists(self, item):
        return item in self._items

    # -------------------------------------------------------
    def get_dict_values(self):
        '''
        Returns a tuple of items in the item dict. Each item is a list.
        '''
        return tuple(self._items.values())

    # -------------------------------------------------------
    def item_replace_all(self, newdict, check_exists=True):
        '''
        Replace the data for each item in the item dict. Original attributes 
        will be retained.
        
        Parameters
        ----------
        newdict : dictionary of item names -> new item data (can also be a dataset)

        check_exists : if True, all newdict keys and old item keys will be compared to ensure a match
        '''
        # for intenal routines, an existance check can often be skipped
        if check_exists:
            for k in newdict:
                if self.item_exists(k) is False:
                    raise ValueError(f"Item {k} not found in original item dictionary.")
            for k in self._items:
                if k not in newdict:
                    raise ValueError(f"Item {k} in original item dictionary not found in new items.")

        # replace the data, keep any attributes if set
        for k, v in newdict.items():
            self._items[k][0] = v

    # -------------------------------------------------------
    def item_rename(self, old, new):
        """
        Rename a single column.

        :param old: Current column name.
        :param new: New column name.
        :return: value portion of item that was renamed
        """
        if old == new:
            return None
        if old not in self._items:
            raise ValueError(f'Invalid column to rename: {old!r} cannot rename column that does not exit in itemcontainer.')
        if new in self._items:
            raise ValueError(f'Invalid column name: {new!r}; already exists in itemcontainer, cannot rename to it.')

        newdict = self._items.copy()

        return_val = None

        self._items.clear()
        for k,v in newdict.items():
            if k == old:
                k = new

                # return the value portion
                return_val = v[0]

            self._items[k]=v

        return return_val

    # -------------------------------------------------------
    def _get_move_cols(self, cols):
        '''
        Possibly convert list/array/dictionary/string/index of items to move
        for item_move_to_front(), item_move_to_back()
        '''
        if isinstance(cols, (str, bytes)):
            cols = [cols]
        elif isinstance(cols, (int, np.integer)):
            try:
                cols = [list(self._items.keys())[cols]]
            except:
                raise ValueError(f"Items could not be indexed by {cols}")

        if not isinstance(cols, (np.ndarray, list, tuple, dict)):
            raise TypeError(f"Item(s) to move must be list, tuple, ndarray, dictionary (keys), single unicode or byte string, or single index. Got {type(cols)}")
        else:
            if len(cols) > len(self._items):
                raise ValueError(f"Found {len(cols)} items to move to front - more than {len(self._items)} in container.")

        return cols

    # -------------------------------------------------------
    def item_move_to_front(self, cols):
        """
        Move single column or group of columns to front of list for iteration/indexing/display.
        Values of columns will remain unchanged.

        :param cols: list of column names to move.
        :return: None
        """
        cols = self._get_move_cols(cols)

        new_all_items = {}
        for cn in cols:
            if cn in self._items:
                new_all_items[cn] = self._items[cn]
            else:
                warnings.warn(f"Column {cn} not found. Could not move to front.")
        for cn in self._items:
            if cn not in new_all_items:
                new_all_items[cn] = self[cn]
        self._items = new_all_items

    # -------------------------------------------------------
    def item_move_to_back(self, cols):
        """
        Move single column or group of columns to front of list for iteration/indexing/display.
        Values of columns will remain unchanged.

        :param cols: list of column names to move.
        :return: None
        """
        cols = self._get_move_cols(cols)

        all_names = self._items
        for cn in cols:
            if cn in all_names:
                all_names[cn] = all_names.pop(cn)
            else:
                warnings.warn(f"Column {cn} not found. Could not move to back.")


    # -------------------------------------------------------
    def item_add_prefix(self, prefix):
        '''
        inplace operation.
        adds prefix in front of existing item name

        faster than calling rename
        '''
        newdict = self._items.copy()
        self._items.clear()
        for k,v in newdict.items():
            self._items[f'{prefix}{k}']=v

    # --------------------------------------------------------
    def item_str_match(self, expression, flags=0):
        """
        Create a boolean mask vector for items whose names match the regex.
        NB Uses re.match(), not re.search().

        :param expression: regular expression
        :param flags: regex flags (from re module).
        :return: list array of bools (len ncols) which is true for columns which match the regex.
        """
        match_fun = re.compile(expression, flags=flags).match
        return [bool(match_fun(x)) for x in self._items.keys()]

    def item_str_replace(self, old, new, maxr=-1):
        '''
        :param old: string to look for within individual names of columns
        :param new: string to replace old string in column names

        If an item name contains the old string, the old string will be replaced with the new one.
        If replacing the string conflicts with an existing item name, an error will be raised.

        returns True if column names were replaced
        '''
        new_names = []
        replace_count = 0
        for item in self._items:
            r = item.replace(old, new, maxr)
            if r != item:
                replace_count += 1
                # prevent name conflict from overwriting existing column
                if r in self._items:
                    raise ValueError(f"Item {r} already existed, cannot make replacement in item name {item}.")
            new_names.append(r)

        # only do this if necessary
        if replace_count != 0:
            newdict = self._items.copy()
            self._items.clear()
            for i, v in enumerate(newdict.values()):
                self._items[new_names[i]] = v
            return True

        return False

    # -------------------------------------------------------
    def footer_get_value(self, key):
        return self.item_get_attribute(key, ATTRIBUTE_FOOTER)
    def footer_set_value(self, key, value):
        self.item_set_attribute(key, ATTRIBUTE_FOOTER, value)

    # --LABEL (LEFT FOR DISPLAY)-----------------------------
    # -------------------------------------------------------
    def label_as_dict(self):
        return self._tagged_as_dict(ATTRIBUTE_LABEL)

    def label_get_names(self):
        return self._tagged_get_names(ATTRIBUTE_LABEL)

    def label_set_names(self, listnames):
        self._tagged_set_names(listnames, ATTRIBUTE_LABEL)

    def label_remove(self):
        self._tagged_remove(ATTRIBUTE_LABEL)

    # --RIGHT FOR DISPLAY------------------------------------
    # -------------------------------------------------------
    def summary_as_dict(self):
        return self._tagged_as_dict(ATTRIBUTE_SUMMARY)

    def summary_get_names(self):
        return self._tagged_get_names(ATTRIBUTE_SUMMARY)

    def summary_set_names(self, listnames):
        self._tagged_set_names(listnames, ATTRIBUTE_SUMMARY)

    def summary_remove(self):
        self._tagged_remove(ATTRIBUTE_SUMMARY)

    # --GENERAL ATTRIBUTE FUNCTIONS--------------------------
    # -------------------------------------------------------
    def _tagged_get_names(self, attrname):
        '''
        Returns a list of item names tagged with attrname in order.
        '''
        tagged_names=[]
        max, tagged_dict = self._tagged_get_dict_max(attrname)
        if max >= 0:
            for i in range(max+1):
                if i in tagged_dict:
                    tagged_names.append(tagged_dict[i])
        return tagged_names

    def _tagged_set_names(self, listnames, attrname):
        '''
        Removes existing items tagged with attrname.
        If items in listnames exist, they will be tagged with attrname.
        '''
        if not isinstance(listnames, list):
            listnames = [listnames]
        self._tagged_remove(attrname)
        for i, tagged in enumerate(listnames):
            if self.item_exists(tagged):
                self._set_attribute(self._items[tagged], attrname, i)

    def _tagged_remove(self, attrname):
        '''
        Removes existing items tagged with attrname.
        '''
        for v in self._items.values():
            # get the attribute tuple
            attr = v[1]
            if attr is not None and hasattr(attr, attrname):
                delattr(attr, attrname)

    def _tagged_as_dict(self, attrname):
        '''
        Returns dictionary of columns tagged with attrname.
        '''
        return_dict = {}
        max, tagged_dict = self._tagged_get_dict_max(attrname)
        if max >= 0:
            for i in range(max+1):
                if i in tagged_dict:
                    name = tagged_dict[i]
                    return_dict[name] = self.item_get_value(name)
        if len(return_dict) > 0:
            return return_dict
        return None

    def _tagged_get_dict_max(self, attrname):
        '''
        Returns unordered dictionary of columns tagged with attrname, max value for order.
        '''
        tagged_dict={}
        max = -1
        for k, v in self._items.items():
            # get the attribute tuple
            attr = v[1]
            if attr is not None and hasattr(attr, attrname):
                val = getattr(attr, attrname)
                if val > max: max = val
                tagged_dict[val] = k
        return max, tagged_dict
