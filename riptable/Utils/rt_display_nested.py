import numpy as np
import os
from ..rt_enum import (
    TypeRegister,
    gAnsiColors,
    DisplayDetectModes,
    TypeId,
    CategoryMode,
    SDS_EXTENSION_BYTES,
)
from ..rt_display import DisplayString, DisplayDetect
from ..rt_sds import _build_schema, decompress_dataset_internal

BOX_LIGHT = {
    'UP_AND_RIGHT': u'\u2514',
    'HORIZONTAL': u'\u2500',
    'VERTICAL': u'\u2502',
    'VERTICAL_AND_RIGHT': u'\u251C',
}  #: Unicode box-drawing glyphs, light style


class KeyArgsConstructor(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class DictTraversal(KeyArgsConstructor):
    """Traversal suitable for a dictionary. Keys are tree labels, all values
    must be dictionaries as well."""

    def get_children(self, node):
        return list(node[1].items())

    def get_root(self, tree):
        return list(tree.items())[0]

    def get_text(self, node):
        return node[0]


class AttributeTraversal(KeyArgsConstructor):
    """Attribute traversal.

    Uses an attribute of a node as its list of children.
    """

    attribute = 'children'  #: Attribute to use.

    def get_children(self, node):
        return getattr(node, self.attribute)


class Style(KeyArgsConstructor):
    """Rendering style for trees."""

    label_format = u'{}'  #: Format for labels.

    def node_label(self, text):
        """Render a node text into a label."""
        return self.label_format.format(text)

    def child_head(self, label):
        """Render a node label into final output."""
        return label

    def child_tail(self, line):
        """Render a node line that is not a label into final output."""
        return line

    def last_child_head(self, label):
        """Like :func:`~asciitree.drawing.Style.child_head` but only called
        for the last child."""
        return label

    def last_child_tail(self, line):
        """Like :func:`~asciitree.drawing.Style.child_tail` but only called
        for the last child."""
        return line


class BoxStyle(Style):
    """A rendering style that uses box draw characters and a common layout."""

    gfx = BOX_LIGHT  #: Glyhps to use.
    label_space = 1  #: Space between glyphs and label.
    horiz_len = 4  #: Length of horizontal lines
    indent = 1  #: Indent for subtrees

    def child_head(self, label):
        return (
            ' ' * self.indent
            + self.gfx['VERTICAL_AND_RIGHT']
            + self.gfx['HORIZONTAL'] * self.horiz_len
            + ' ' * self.label_space
            + label
        )

    def child_tail(self, line):
        return ' ' * self.indent + self.gfx['VERTICAL'] + ' ' * self.horiz_len + line

    def last_child_head(self, label):
        return (
            ' ' * self.indent
            + self.gfx['UP_AND_RIGHT']
            + self.gfx['HORIZONTAL'] * self.horiz_len
            + ' ' * self.label_space
            + label
        )

    def last_child_tail(self, line):
        return (
            ' ' * self.indent
            + ' ' * len(self.gfx['VERTICAL'])
            + ' ' * self.horiz_len
            + line
        )


class LeftAligned(KeyArgsConstructor):
    """Creates a renderer for a left-aligned tree.

    Any attributes of the resulting class instances can be set using
    constructor arguments."""

    draw = BoxStyle()
    "The draw style used. See :class:`~asciitree.drawing.Style`."
    traverse = DictTraversal()
    "Traversal method. See :class:`~asciitree.traversal.Traversal`."

    def render(self, node):
        """Renders a node. This function is used internally, as it returns
        a list of lines. Use :func:`~asciitree.LeftAligned.__call__` instead.
        """
        lines = []

        children = self.traverse.get_children(node)
        lines.append(self.draw.node_label(self.traverse.get_text(node)))

        for n, child in enumerate(children):
            child_tree = self.render(child)

            if n == len(children) - 1:
                # last child does not get the line drawn
                lines.append(self.draw.last_child_head(child_tree.pop(0)))
                lines.extend(self.draw.last_child_tail(l) for l in child_tree)
            else:
                lines.append(self.draw.child_head(child_tree.pop(0)))
                lines.extend(self.draw.child_tail(l) for l in child_tree)

        return lines

    def __call__(self, tree):
        """Render the tree into string suitable for console output.

        :param tree: A tree."""
        return '\n'.join(self.render(self.traverse.get_root(tree)))


class DisplayNested:

    inline_svg = {
        "array": '<svg xmlns="http://www.w3.org/2000/svg" width="25" height="25" viewBox="0 0 50 50" enable-background="new 0 0 50 50"><path d="M 7 2 L 7 3 L 7 47 L 7 48 L 8 48 L 42 48 L 43 48 L 43 47 L 43 15 L 43 14.59375 L 42.71875 14.28125 L 30.71875 2.28125 L 30.40625 2 L 30 2 L 8 2 L 7 2 z M 9 4 L 29 4 L 29 15 L 29 16 L 30 16 L 41 16 L 41 46 L 9 46 L 9 4 z M 31 5.4375 L 39.5625 14 L 31 14 L 31 5.4375 z"/></svg>',
        "object": '<svg xmlns="http://www.w3.org/2000/svg" width="25" height="25" viewBox="2 2 28 28"><path d="M 16 4 C 9.3844276 4 4 9.3844276 4 16 C 4 22.615572 9.3844276 28 16 28 C 22.615572 28 28 22.615572 28 16 C 28 9.3844276 22.615572 4 16 4 z M 16 6 C 21.534692 6 26 10.465308 26 16 C 26 21.534692 21.534692 26 16 26 C 10.465308 26 6 21.534692 6 16 C 6 10.465308 10.465308 6 16 6 z M 22 16 C 22 19.325562 19.325562 22 16 22 L 16 24 C 20.406438 24 24 20.406438 24 16 L 22 16 z"/></svg>',
        "container": '<svg xmlns="http://www.w3.org/2000/svg" width="25" height="25" viewBox="0 0 50 50"><path d="M 25 0 C 19.3545 0 14.230156 1.0122381 10.4375 2.71875 C 8.5411719 3.5720059 6.9801672 4.6031002 5.84375 5.8125 C 4.7073328 7.0218998 4 8.4567299 4 10 L 4 19.8125 A 1.0001 1.0001 0 0 0 4 20 L 4 29.8125 A 1.0001 1.0001 0 0 0 4 30 L 4 39 C 4 40.54327 4.7073328 41.9781 5.84375 43.1875 C 6.9801672 44.3969 8.5411719 45.427994 10.4375 46.28125 C 14.230156 47.987762 19.3545 49 25 49 C 30.6455 49 35.769844 47.987762 39.5625 46.28125 C 41.458828 45.427994 43.019833 44.3969 44.15625 43.1875 C 45.292667 41.9781 46 40.54327 46 39 L 46 20 L 46 19.90625 L 46 10 C 46 8.4567299 45.292667 7.0218998 44.15625 5.8125 C 43.019833 4.6031002 41.458828 3.5720059 39.5625 2.71875 C 35.769844 1.0122381 30.6455 0 25 0 z M 25 2 C 30.4025 2 35.273406 3.0122619 38.71875 4.5625 C 40.441422 5.3376191 41.800167 6.2431811 42.6875 7.1875 C 43.574833 8.1318189 44 9.0567701 44 10 C 44 10.94323 43.574833 11.868181 42.6875 12.8125 C 41.800167 13.756819 40.441422 14.662381 38.71875 15.4375 C 35.273406 16.987738 30.4025 18 25 18 C 19.5975 18 14.726594 16.987738 11.28125 15.4375 C 9.5585781 14.662381 8.1998328 13.756819 7.3125 12.8125 C 6.4251672 11.868181 6 10.94323 6 10 C 6 9.0567701 6.4251672 8.1318189 7.3125 7.1875 C 8.1998328 6.2431811 9.5585781 5.3376191 11.28125 4.5625 C 14.726594 3.0122619 19.5975 2 25 2 z M 6 14.34375 C 7.1223185 15.486391 8.6228131 16.464729 10.4375 17.28125 C 14.230156 18.987762 19.3545 20 25 20 C 30.6455 20 35.769844 18.987762 39.5625 17.28125 C 41.377187 16.464729 42.877682 15.486391 44 14.34375 L 44 19.84375 A 1.0001 1.0001 0 0 0 44 19.9375 L 44 20 C 44 20.94323 43.574833 21.868181 42.6875 22.8125 C 41.800167 23.756819 40.441422 24.662381 38.71875 25.4375 C 35.273406 26.987738 30.4025 28 25 28 C 19.5975 28 14.726594 26.987738 11.28125 25.4375 C 9.5585781 24.662381 8.1998328 23.756819 7.3125 22.8125 C 6.4251672 21.868181 6 20.94323 6 20 A 1.0001 1.0001 0 0 0 6 19.8125 L 6 14.34375 z M 6 24.34375 C 7.1223185 25.486391 8.6228131 26.464729 10.4375 27.28125 C 14.230156 28.987762 19.3545 30 25 30 C 30.6455 30 35.769844 28.987762 39.5625 27.28125 C 41.377187 26.464729 42.877682 25.486391 44 24.34375 L 44 29.84375 A 1.0001 1.0001 0 0 0 44 29.9375 L 44 30 C 44 30.94323 43.574833 31.868181 42.6875 32.8125 C 41.800167 33.756819 40.441422 34.662381 38.71875 35.4375 C 35.273406 36.987738 30.4025 38 25 38 C 19.5975 38 14.726594 36.987738 11.28125 35.4375 C 9.5585781 34.662381 8.1998328 33.756819 7.3125 32.8125 C 6.4251672 31.868181 6 30.94323 6 30 A 1.0001 1.0001 0 0 0 6 29.8125 L 6 24.34375 z M 6 34.34375 C 7.1223185 35.486391 8.6228131 36.464729 10.4375 37.28125 C 14.230156 38.987762 19.3545 40 25 40 C 30.6455 40 35.769844 38.987762 39.5625 37.28125 C 41.377187 36.464729 42.877682 35.486391 44 34.34375 L 44 39 C 44 39.94323 43.574833 40.868181 42.6875 41.8125 C 41.800167 42.756819 40.441422 43.662381 38.71875 44.4375 C 35.273406 45.987738 30.4025 47 25 47 C 19.5975 47 14.726594 45.987738 11.28125 44.4375 C 9.5585781 43.662381 8.1998328 42.756819 7.3125 41.8125 C 6.4251672 40.868181 6 39.94323 6 39 L 6 34.34375 z"/></svg>',
    }

    def __init__(self):
        self._fmtstart = None
        self._fmtend = None

    # ---------------------------------------------------------------
    # TODO: borrow from global color pallette - probably needs to move out of display
    @property
    def fmtstart(self):
        if self._fmtstart is None:
            self._fmtstart = (
                ""
                if DisplayDetect.Mode == DisplayDetectModes.Console
                else gAnsiColors['LightCyan']
            )
        return self._fmtstart

    # ---------------------------------------------------------------
    @property
    def fmtend(self):
        if self._fmtend is None:
            self._fmtend = (
                ""
                if DisplayDetect.Mode == DisplayDetectModes.Console
                else gAnsiColors['Normal']
            )
        return self._fmtend

    # ---------------------------------------------------------------
    def _map_asciitree(self, data, structure, name=None, info=False):
        structure[self.fmtstart + name + self.fmtend] = {}
        structure = structure[self.fmtstart + name + self.fmtend]

        if not len(data):
            return

        maxlen = len(max(list(data.keys()), key=len))

        for k, v in data.items():

            info_indent = " " * (maxlen - len(k) + 1)

            if hasattr(v, 'items'):
                name = f"{k}" if info else f"{k} ({type(v).__name__})"
                self._map_asciitree(v, structure, name=name, info=info)

            else:
                item_formatter = _default_info
                if type(v) == TypeRegister.FastArray or type(v) == np.ndarray:
                    item_formatter = _arr_info

                elif type(v) == TypeRegister.Categorical:
                    item_formatter = _cat_info

                # TODO: Use np.isscalar() here instead?
                elif isinstance(
                    v, (int, float, bool, np.bool_, np.integer, np.floating, str, np.str_, bytes, np.bytes_)
                ):
                    item_formatter = _scalar_info

                item_name = k + f"{info_indent}{item_formatter(v)}"

                structure[item_name] = {}

    def _map_full_paths(self, data, structure, name=None, prefix=""):
        pass
        # structure[prefix+self.fmtstart+name+self.fmtend] = {}
        # structure = structure[prefix+self.fmtstart+name+self.fmtend]

        # maxlen = len(max(list(data.keys()), key=len))

        # for k, v in data.items():

        #    info_indent = " "*(maxlen - len(k)+1)

        #    if hasattr(v, 'items'):
        #        prefix = prefix+name+"."
        #        self._map_full_paths(data, structure, name=f"{k}", prefix=prefix)

        #    else:
        #        structure[prefix+k] = {}

    # ---------------------------------------------------------------
    def _map_htmltree(self, data, structure, name=None, html_str=[], showicon=True):
        structure[name] = {}
        structure = structure[name]

        menu_str = "<li class='menu-header'><a>"
        if showicon:
            menu_str += self.inline_svg['container']
            # menu_str += "<img src='dbicon.png' />&nbsp;&nbsp;"
        menu_str += f"{name} ({type(data).__name__})</a></li><ul class='sfw-ul'>"
        html_str.append(menu_str)

        for k, v in data.items():
            t = type(v)

            if hasattr(v, 'items'):
                self._map_htmltree(
                    v, structure, name=k, html_str=html_str, showicon=showicon
                )

                html_str.append("</ul></li>")
            else:
                file_str = "<li>"

                if type(v) == TypeRegister.FastArray or type(v) == np.ndarray:
                    if showicon:
                        file_str += self.inline_svg['array']
                    file_str += f"<p>{k}"
                    file_str += f" {type(v).__name__} {_arr_info(v)}"  # {v.dtype}"

                # TODO: Use np.isscalar() here instead?
                elif isinstance(
                    v, (int, float, bool, np.bool_, np.integer, np.floating, str, np.str_, bytes, np.bytes_)
                ):
                    if showicon:
                        file_str += self.inline_svg['object']
                    file_str += f"<p>{k}"
                    file_str += f" {v}"
                else:
                    if showicon:
                        file_str += self.inline_svg['object']
                    file_str += f"<p>{k} {type(v).__name__}"

                file_str += "</p></li>"
                html_str.append(file_str)
                structure[k] = {}

    # ---------------------------------------------------------------
    def _build_nested_html(self, data, name=None):
        html_str = []
        # STYLE
        html_str.append("<html><head><style>")
        html_str.append(
            "a{ padding-top: 15px; line-height: 20px; vertical-align: top; font-weight: bold; cursor: pointer;}"
        )
        html_str.append(".hidelist{ display: none;}")
        html_str.append("li{ margin: 3px auto auto auto; }")
        html_str.append("li.menu-header{ margin: 10px auto auto auto;}")
        html_str.append(
            "li p{ line-height:25px; vertical-align:top; display:inline-block; margin: 0 0 0 5px !important;}"
        )
        html_str.append("svg{display:inline-block;}")
        html_str.append(
            "ul.rt-ul{ list-style-type: none !important; list-style: none !important;}"
        )
        # html_str.append("ul.rt-ul img{ float:left; }")
        # html_str.append("li.listitem-wrap{ width: 100%; height: 20px; line-height: 20px; }")
        html_str.append("</style>")

        # JAVASCRIPT
        html_str.append(
            "<script src='https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js'></script>"
        )
        html_str.append(
            "<script>$(document).ready(function(){$('a').click(function(){$(this).parent().next().toggleClass('hidelist');return false;});});"
        )
        html_str.append("</script></head><body>")

        if name is None:
            name = type(data).__name__
        # NESTED LISTS
        structure = {}
        html_str.append("<ul class='sfw-ul'>")
        self._map_htmltree(data, structure, name=name, html_str=html_str)
        html_str.append('</ul></body></html>')
        html_str = "".join(html_str)

        return DisplayString(html_str)

    # ---------------------------------------------------------------
    def _build_nested_ascii(self, data, name=None, showpaths=False, info=False):
        if name is None:
            name = type(data).__name__

        structure = {}
        if showpaths:
            self._map_full_paths(data, structure, name=name)
        else:
            self._map_asciitree(data, structure, name=name, info=info)

        ascii_string = LeftAligned()
        return DisplayString(ascii_string(structure))

    def build_nested_html(self, data={}, name=None):
        return self._build_nested_html(data=data, name=name)

    def build_nested_string(self, data={}, name=None, showpaths=False, info=False):
        return self._build_nested_ascii(
            data=data, name=name, showpaths=showpaths, info=info
        )


# ---------------------------------------------------------------
def _default_info(item):
    pass


def _scalar_info(item):
    return item


def _arr_info(arr):
    dt = arr.dtype
    typename = dt.name
    sh = str(arr.shape)
    itemsize = str(arr.itemsize)
    return " ".join([typename, sh, itemsize])


def _cat_info(arr):
    if arr.category_mode == CategoryMode.StringArray:
        str_type = arr.category_array.dtype.char
    else:
        str_type = ""

    item_name = (
        f"Categorical {arr.dtype} {arr.shape} {arr.category_mode.name} ({str_type})"
    )
    return item_name


# ---------------------------------------------------------------
def treedir(path, name=None):
    # should we use the struct routine to do this, or just the file names?
    # return TypeRegister.Struct.sload(path, info=True)

    if isinstance(path, bytes):
        path = path.decode()

    if not os.path.isdir(path):
        raise OSError(f"{path} doesn't exist.")

    dirlist = os.listdir(path)
    schema = _build_schema(path, dirlist)

    if name is not None:
        if name in schema:
            schema = schema[name]
        else:
            raise ValueError(f"Could not find {name} in directory {path}")

    if '_root' in schema:
        del schema['_root']
    dirname = os.path.basename(os.path.normpath(path))

    full_schema = {dirname: schema}
    ascii_string = LeftAligned()
    return DisplayString(ascii_string(full_schema))


# ---------------------------------------------------------------
def _mask_flags(flagnum):
    flagdict = {
        "C_CONTIGUOUS": 0x0001,
        "F_CONTIGUOUS": 0x0002,
        "OWNDATA": 0x0004,
        "FORCECAST": 0x0010,
        "ENSURECOPY": 0x0020,
        "ENSUREARRAY": 0x0040,
        "ELEMENTSTRIDES": 0x0080,
        "ALIGNED": 0x0100,
        "NOTSWAPPED": 0x0200,
        "WRITEABLE": 0x0400,
        "UPDATEIFCOPY": 0x1000,
    }
    for flagname, bitcode in flagdict.items():
        flagdict[flagname] = (flagnum & bitcode) > 0

    return flagdict
