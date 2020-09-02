__all__ = ['build_header_tuples',  'output_cache_flush',
           'output_cache_none', 'output_cache_setsize', 'parse_header_tuples',
           'profile_func', 'sub2ind','autocomplete','jedi_completions']

import warnings
import numpy as np

from .rt_enum import TypeRegister, ColHeader
#
#
# MATLAB
# sub2ind([10 10], 3,3)
#ans = 23.00
#
#------------------------
#python
#In [10]: np.ravel_multi_index((3,3),(10,10))
#Out[10]: 33
#
#In [6]: np.unravel_index(23,(10,10))
#Out[6]: (2, 3)
#
#----------------------------------------------
# MATLAB
# sub2ind([7 13], 6, 12)
# ans =  83.00
#
# sub2ind([7 13], 5, 12)
# ans =  82.00
#
# sub2ind([7 15], 5, 12)
# ans =  82.00
#
# np.ravel_multi_index((11,5),(13,7))
# 82
#
# np.ravel_multi_index((5,11),(7, 13), order='F')
# 82
# ALSO
# np.ravel_multi_index((5,11),(7, 15), order='F')
# 82

# MATH for this.. where is 5,11 in the array (7,13) if the array (7,13) was flattened
# 5, 11 in Fortran order means
# each row is 11 items and there are 5 columns
#
# so each row 11*7 = 77 + 5 = 82
# 


def sub2ind(aSize,aPosition):
    """
    MATLAB
    ---------------------------------
    sub2ind Linear index from multiple subscripts.
    sub2ind is used to determine the equivalent single index
    corresponding to a given set of subscript values.

    IND = sub2ind(SIZ,I,J) returns the linear index equivalent to the
    row and column subscripts in the arrays I and J for a matrix of
    size SIZ.

    IND = sub2ind(SIZ,I1,I2,...,IN) returns the linear index
    equivalent to the N subscripts in the arrays I1,I2,...,IN for an
    array of size SIZ.

    I1,I2,...,IN must have the same size, and IND will have the same size
    as I1,I2,...,IN. For an array A, if IND = sub2ind(SIZE(A),I1,...,IN)),
    then A(IND(k))=A(I1(k),...,IN(k)) for all k.

    PYTHON
    ----------------------------
    ravel_multi_index(...)
    ravel_multi_index(multi_index, dims, mode='raise', order='C')

    Converts a tuple of index arrays into an array of flat
    indices, applying boundary modes to the multi-index.

    Parameters
    ----------
    multi_index : tuple of array_like
        A tuple of integer arrays, one array for each dimension.
    dims : tuple of ints
        The shape of array into which the indices from ``multi_index`` apply.
    mode : {'raise', 'wrap', 'clip'}, optional
        Specifies how out-of-bounds indices are handled.  Can specify
        either one mode or a tuple of modes, one mode per index.

        * 'raise' -- raise an error (default)
        * 'wrap' -- wrap around
        * 'clip' -- clip to the range

        In 'clip' mode, a negative index which would normally
        wrap will clip to 0 instead.
    order : {'C', 'F'}, optional
        Determines whether the multi-index should be viewed as
        indexing in row-major (C-style) or column-major
        (Fortran-style) order.

    Returns
    -------
    raveled_indices : ndarray
        An array of indices into the flattened version of an array
        of dimensions ``dims``.

    See Also
    --------
    unravel_index

    Notes
    -----
    .. versionadded:: 1.6.0

    Examples
    --------
    >>> arr = np.array([[3,6,6],[4,5,1]])
    >>> np.ravel_multi_index(arr, (7,6))
    array([22, 41, 37])
    >>> np.ravel_multi_index(arr, (7,6), order='F')
    array([31, 41, 13])
    >>> np.ravel_multi_index(arr, (4,6), mode='clip')
    array([22, 23, 19])
    >>> np.ravel_multi_index(arr, (4,4), mode=('clip','wrap'))
    array([12, 13, 13])

    >>> np.ravel_multi_index((3,1,4,1), (6,7,8,9))
    1621
    """
    return np.ravel_multi_index(aPosition, aSize, order='F')




#---------------------------------------------------------------------------
def build_header_tuples(headers, span, group):
    if headers is None: return []
    return [ColHeader(name, span, group) for name in headers]

#---------------------------------------------------------------------------
def parse_header_tuples(header_tups):
    return [h.col_name for h in header_tups]

#----------------------------------------------
def jedi_completions(text, offset):
    '''
    autocomplete() must be called first.
    Not used yet. Returns the same completions jedi would.
    Examples
    --------
    from riptable.rt_misc import jedi_completions
    st = Struct({'a': 5})
    jedi_completions('st', 2)
    '''

    def position_to_cursor(text:str, offset:int):
        before = text[:offset]
        blines = before.split('\n')
        line = before.count('\n')
        col = len(blines[-1])
        return line, col

    def cursor_to_position(text:str, line:int, column:int)->int:
        lines = text.split('\n')
        return sum(len(l) + 1 for l in lines[:line]) + column

    try:
        ipc = Hooker._ipcompleter
        cursor_line, cursor_column = position_to_cursor(text, offset)

        namespaces = [ipc.namespace]
        if ipc.global_namespace is not None:
            namespaces.append(ipc.global_namespace)

        completion_filter = lambda x:x
        offset = cursor_to_position(text, cursor_line, cursor_column)
        # filter output if we are completing for object members
        if offset:
            pre = text[offset-1]
#            if pre == '.':
#                if self.omit__names == 2:
#                    completion_filter = lambda c:not c.name.startswith('_')
#                elif self.omit__names == 1:
#                    completion_filter = lambda c:not (c.name.startswith('__') and c.name.endswith('__'))
#                elif self.omit__names == 0:
#                    completion_filter = lambda x:x
#                else:
#                    raise ValueError("Don't understand self.omit__names == {}".format(self.omit__names))

        import jedi
        interpreter = jedi.Interpreter(
            text[:offset], namespaces, column=cursor_column, line=cursor_line + 1)
        return interpreter.completions()

    except Exception:
        return []



class CacheWarning(UserWarning):
    pass

# ------------------------------------------------------------------------
def output_cache_none():
    '''
    used in ipython, jupyter, or spyder
    sets the terminalInteractiveShell output cache size to none
    Out[#] will no longer work
    the Out dictionary will be empty
    _# will no longer work
    '''
    try:
        import IPython
        t=IPython.terminal.interactiveshell.TerminalInteractiveShell()
        # stop the interactiveshell from caching (this is the Out[])
        t.cache_size=0
        # note: do we set the use_ns Out back to something to allow to kick back in?
        # stop the displayhook from caching
        ipython = IPython.get_ipython()
        if ipython:
            ipython.displayhook.do_full_cache = False
    except:
        pass
        #warnings.warn("Failed to set output_cache_none.", CacheWarning)

# ------------------------------------------------------------------------
def output_cache_setsize(cache_size=100):
    '''
    used in ipython, jupyter, or spyder
    sets the terminalInteractiveShell output cache size to cache_size (100 is the default)
    '''
    try:
        import IPython
        t=IPython.terminal.interactiveshell.TerminalInteractiveShell()
        # stop the interactiveshell from caching (this is the Out[])
        t.cache_size=cache_size
        # stop the displayhook from caching
        ipython = IPython.get_ipython()
        if ipython:
            ipython.displayhook.do_full_cache = True
        # note: consider setting cull_fraction or cache_size
    except:
        warnings.warn("Failed to set output_cache_setsize.", CacheWarning)


# ------------------------------------------------------------------------
def output_cache_flush():
    '''
    used in ipython, jupyter, or spyder
    calling output_cache_flush() will remove object reference in the output cache
    it is recommended this is called when there are memory concerns
    '''
    try:
        from IPython import get_ipython
        # from IPython import InteractiveShell
        ipython = get_ipython()
        if not ipython:
            return
        tempcache=ipython.displayhook.do_full_cache
        ipython.displayhook.do_full_cache = True
        ipython.displayhook.flush()
        # ipython.displayhook.cache_size =0
        del ipython.displayhook.shell.user_ns['_']
        shell = ipython
        # ns_refs = [m.__dict__ for m in shell._main_mod_cache.values()]
        ## Also check in output history
        # ns_refs.append(shell.history_manager.output_hist)
        # for ns in ns_refs:
        #    to_delete = [n for n, o in ns.items()]
        #    for name in to_delete:
        #        if (ns[name] is obj):
        #            print("**deleting", name)
        #            del ns[name]

        ns_refs = [shell.user_ns, shell.user_global_ns, shell.user_ns_hidden]
        for ns in ns_refs:
            to_delete = [n for n, o in ns.items()]
            for name in to_delete:
                if name in ('_', '__', '___'):
                    print("**deleting", name, type(name))
                    del ns[name]

                # if ns[name] is obj:
                #    print("**extra deleting", name)
                #    del ns[name]

                # search for variables named _# such as _4 or _37
                elif name.startswith('_') and name[1] >= '0' and name[1] <= '9':
                    temp = ns[name]

                    # TODO: consider looking for more than just Dataset
                    if isinstance(temp, TypeRegister.Dataset):
                        print("**extra deleting Dataset", name, "size of ", temp._last_row_stats())
                    elif isinstance(temp, np.ndarray):
                        print("**extra deleting array", name, "size of ", (temp.itemsize * temp.size) / 1e6, "MB")
                    else:
                        print("**extra deleting array", name, "type ", type(temp))
                    del ns[name]

        # Ensure it is removed from the last execution result
        shell.last_execution_result = None

        # displayhook keeps extra references, but not in a dictionary
        for name in ('_', '__', '___'):
            setattr(shell.displayhook, name, None)

        # set it back to what it was
        ipython.displayhook.do_full_cache = tempcache

    except:
        warnings.warn("Failed to set output_cache_flush.", CacheWarning)


# ------------------------------------------------------------------------
def profile_func(func, sortby='time'):
    '''
    Used to profile a function that has no arguments

    Examples
    --------
    This will time how long the __repr__ function to print out a dataset

    >>> import riptable as rt
    >>> import riptable_docdata as rtd
    >>> trips = rt.Dataset(rtd.get_bike_trips_data('trips'))
    >>> profile_func(trips.__repr__)
    '''

    try:
        import cProfile
        pr = cProfile.Profile()
        pr.enable()

        func()

        pr.disable()
        pr.print_stats(sort=sortby)
    except:
        print(f"cProfile could not be imported or could not run {func}")

# simple class to hold what we hooked
class Hooker:
    _ipcompleter = None
    _orig_do_complete = None
    _orig_deduplicate = None
    _putils = None
    babydict={c:None for c in list('_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.')}


# The Completion class is copied
class Completion:
    """
    Completion object used and return by IPython completers.

    .. warning:: Unstable

        This function is unstable, API may change without warning.
        It will also raise unless use in proper context manager.

    This act as a middle ground :any:`Completion` object between the
    :any:`jedi.api.classes.Completion` object and the Prompt Toolkit completion
    object. While Jedi need a lot of information about evaluator and how the
    code should be ran/inspected, PromptToolkit (and other frontend) mostly
    need user facing information.

    - Which range should be replaced replaced by what.
    - Some metadata (like completion type), or meta information to displayed to
      the use user.

    For debugging purpose we can also store the origin of the completion (``jedi``,
    ``IPython.python_matches``, ``IPython.magics_matches``...).
    """

    __slots__ = ['start', 'end', 'text', 'type', 'signature', '_origin']

    def __init__(self, start: int, end: int, text: str, *, type: str=None, _origin='', signature='') -> None:

        self.start = start
        self.end = end
        self.text = text
        self.type = type
        self.signature = signature
        self._origin = _origin

    def __repr__(self):
        return '<Completion start=%s end=%s text=%r type=%r, signature=%r,>' % \
                (self.start, self.end, self.text, self.type or '?', self.signature or '?')

    def __eq__(self, other):
        """
        Equality and hash do not hash the type (as some completer may not be
        able to infer the type), but are use to (partially) de-duplicate
        completion.

        Completely de-duplicating completion is a bit tricker that just
        comparing as it depends on surrounding text, which Completions are not
        aware of.
        """
        return self.start == other.start and \
            self.end == other.end and \
            self.text == other.text

    def __hash__(self):
        return hash((self.start, self.end, self.text))



def autocomplete(hook:bool=True, jedi:bool=None, greedy:bool=None):
    '''
    Call rt.autocomplete() to specialize jupyter lab autcomplete output.
    arrays, categoricals, datetime, struct, and datasets will be detected.
    array will be array followed by the dtype.
    
    Parameters
    ----------
    hook: bool, default True
        set to False to unhook riptable autocomplete
    jedi: bool, default None
        set to True to set use_jedi in IPython
    greedy: bool, default None
        set to True to set greedy in IPython for [' autocomplete

    Examples
    --------
    >>> rt.autocomplete(); ds=Dataset({'test':arange(5), 'another':arange(5.0), 'mycat':rt.Cat(arange(5)), 'mystr': arange(5).astype('S')})
    Now in jupyter lab type 'ds.<tab>'

    '''

    def gettype(element) -> str:
        result: str = ''
        if hasattr(element, '_autocomplete'):
            result = element._autocomplete()
        elif isinstance(element, np.ndarray):
            # how to display array
            bstring = 'Array '
            dnum = element.dtype.num
            if dnum ==0:
                # bool check
                extra='b'
            elif dnum <= 10:
                # integer check
                if dnum & 1 ==1:
                    extra='i'
                else:
                    extra='u'
            elif dnum <= 13:
                extra='f'
            else:
                extra = element.dtype.char+str(element.itemsize)

            if dnum <=13:
                result = bstring+extra+str(element.itemsize*8)
            else:
                result = bstring+extra

        else:
            if callable(element):
                result='function'
            else:
                try:
                    result=element.__class__.__name__
                except Exception:
                    result = 'unknown'
        return result


    #----------------------------------------------
    def gettype_foritem(acobj, name, oclass, oinstance) -> str:
        t='<unknown>'
        try:
            item=None
            if name in oinstance:
                item = oinstance[name]
            elif name in oclass:
                item = oclass[name]
            elif hasattr(acobj, name):
                item = getattr(acobj, name)
            t=gettype(item)
        except Exception:
            pass
        return t


    # ---------------------------------------------
    def _evaluate_text(code, dotpos, careful:bool=False):
        '''
        careful: when True indicates that the entire line should NOT be evaluated

        NOTE: Internal function that calls eval() which may not be acceptable for some applications.
        '''
        def call_eval(text):
            try:
                # try two diff namespaces
                acobj = eval(mainpart, _ipcompleter.namespace)
            except:
                try:
                    acobj = eval(mainpart, _ipcompleter.global_namespace)
                except:
                    acobj = None
            return acobj

        mainpart = code[:dotpos]
        acobj=None
        startpos = dotpos
        endpos = dotpos+1
        startwith = None

        if careful:
            # check for ()
            pass

        if not careful:
            acobj=call_eval(mainpart)

        if acobj is None:
            # trickier... now scan backwards from dot
            while startpos >= 0:
                # search back until hit non-naming character
                if code[startpos] not in Hooker.babydict:
                    startpos += 1
                    break
                startpos -= 1

            #careful did not check entire line
            if careful and startpos < 0:
                startpos=0
            if startpos >= 0:
                mainpart=code[startpos:dotpos]
                acobj = call_eval(mainpart)


            # if we still have not found it
            if acobj is None:
                laststartpos=startpos
                startpos = dotpos
                while startpos >= 0:
                    # search back until hit paren() or comma
                    if code[startpos] in '(),':
                        startpos += 1
                        break
                    startpos -= 1

                #careful did not check entire line
                if careful and startpos < 0:
                    startpos=0

                if startpos != laststartpos:
                    if startpos >= 0:
                        mainpart=code[startpos:dotpos]
                        acobj = call_eval(mainpart)

        if acobj is not None:
            # calc the subsection
            while endpos < len(code):
                # search back until hit non-naming character
                if code[endpos] not in Hooker.babydict:
                    break
                endpos += 1

            if endpos > (dotpos +1):
                startwith = code[dotpos+1:endpos]
            #print("startpos", startpos, endpos, mainpart, startwith)
            startpos =dotpos
        
        return acobj, startpos, mainpart, endpos, startwith

    # ---------------------------------------------
    def _riptable_deduplicate_completions(text, completions):
        # This is the hook for use_jedi=True and console.
        #[<Completion start=3 end=3 text='as_struct' type='function', signature='(self)',>,  ..]
        # there is also special code to detect a 'function' in IPython\terminal\ptutils.py
        found = False

        # turn the enumerator into a list
        completions=list(completions)

        # look for apply_schema as only riptable will have this marker up front
        for comp in completions:
            if comp.text == 'apply_schema' or comp.text == 'apply_cols':
                found = True  # looks like our Struct
                break

        # if jedi completed something that is not a Struct, stop trying to autocomplete
        if not found and len(completions) > 0:
            return Hooker._orig_deduplicate(text, completions)

        # we only autocomplete on dots
        dotpos=text.rfind('.')

        if dotpos > 0:
            acobj, startpos, mainpart, endpos, startwith = _evaluate_text(text, dotpos, careful=not found)

            if acobj is not None:      
                
                # calc the subsection
                endpos = dotpos+1
                while endpos < len(text):
                    # search back until hit non-naming character
                    if text[endpos] not in Hooker.babydict:
                        break
                    endpos += 1

                # is this a container class we own
                if isinstance(acobj, (TypeRegister.Struct, TypeRegister.FastArray)):
                    oclass=acobj.__class__.__dict__
                    oinstance = acobj.__dict__

                    # TODO: if jedi has a mistake, do we correct it here?
                    if len(completions) ==0:
                        #redo completions
                        completions=[]
                        ldir = dir(acobj)
                        for c in ldir:
                            if not c.startswith('_'):
                                # check if we have a startswith
                                if startwith is not None:
                                    if not c.startswith(startwith):
                                        continue

                                t=gettype_foritem(acobj, c, oclass, oinstance)
                                completions.append(Completion(start=startpos+1, end=dotpos, text=c, type=t, signature='(self)'))

                    if isinstance(acobj, TypeRegister.Struct):
                        # get the columns
                        keys = acobj.keys()
                        keys.sort()
                        movetotop={}

                        # first put the columns in
                        for k in keys:
                            # check if we have a startswith
                            if startwith is not None:
                                if not k.startswith(startwith):
                                    continue

                            # Struct or Dataset getitem call
                            element = acobj[k]
                            movetotop[k] = Completion(start=startpos+1, end=dotpos, text=k, type=gettype(acobj[k]), signature='(self)')

                        # then add anything else (note if completions is empty we could call dir)
                        for comp in completions:
                            text = comp.text
                            if text is not None and text not in movetotop:
                                movetotop[text] = comp

                        completions=list(movetotop.values())

        return Hooker._orig_deduplicate(text, completions)

    def _riptable_do_complete(self, code, cursor_pos):
        '''
        Hooked from ipythonkernel.do_complete.  Hit in jupyter lab.
        Calls the original do_complete, then possibly rearranges the list.
        As of Dec 2019, this is the use_jedi=True hook in jupyter lab.
        '''

        # self is ipkernel.ipythonkernel
        # code is what text the user typed
        # call original first (usually kicks in jedi)
        result =Hooker._orig_do_complete(self, code, cursor_pos)        

        # we only autocomplete on dots
        dotpos=code.rfind('.')

        if dotpos > 0:
            mainpart = code[:dotpos]
            acobj, startpos, mainpart,endpos, startwith = _evaluate_text(code, dotpos)

            if acobj is not None:
                try:
                    if isinstance(acobj, TypeRegister.Struct):

                        oclass=acobj.__class__.__dict__
                        oinstance = acobj.__dict__

                        # add a dot to complete mainpart for string matching later
                        mainpart += '.'
                        lenmainpart = len(mainpart)

                        # get the jedi completions
                        rmatches=result['matches']

                        # check if there are any jedi completions
                        if len(rmatches) ==0:

                            # jedi failed, so we will attempt
                            matches = []
                            completions=[]
                            ldir = dir(acobj)
                            for c in ldir:
                                if not c.startswith('_'):
                                    # check if we have a startswith
                                    if startwith is not None:
                                        if not c.startswith(startwith):
                                            continue
                                    matches.append(c)
                                    t=gettype_foritem(acobj, c, oclass, oinstance)
                                    completions.append({'start': startpos, 'end': dotpos, 'text': c, 'type': t})

                            rmatches=matches

                            result={'matches': matches, 'cursor_end': endpos, 'cursor_start': dotpos+1, 
                                    'metadata': {'_jupyter_types_experimental': completions},
                                    'status':'ok'}

                        meta = result['metadata']
                        keys = acobj.keys()
                        keys.sort()

                        # result will look similar to:
                        # {'matches': ['list of strings'], 'cursor_end': 5, 'cursor_start': 5, 'metadata': {'_jupyter_types_experimental':
                        #     [{'start': 5, 'end': 5, 'text': 'add_traits', 'type': 'function'},
                        #      {'start': 5, 'end': 5, 'text': 'update_config', 'type': 'function'}]}, 'status':'ok'}
                        # jupyter notebook
                        jtypes = meta.get('_jupyter_types_experimental', None)                        
                        if jtypes is not None:
                            toptext=[]
                            bottomtext=[]
                            
                            topmatch=[]
                            bottommatch=[]

                            # jtypes is a list of dicts
                            for jdict in jtypes:
                                top=False
                                text = jdict.get('text', None)
                                if text is not None:
                                    subtext=text
                                    if text.startswith(mainpart):
                                        subtext = text[lenmainpart:]
                                    # jedi matches after the dot
                                    if subtext in keys:
                                        # Struct or Dataset getitem call                                    
                                        element = acobj[subtext]
                                        jdict['type'] = gettype(element)
                                        top=True
                                if top:
                                    topmatch.append(jdict)
                                    toptext.append(text)
                                else:
                                    bottommatch.append(jdict)
                                    bottomtext.append(text)

                            # change the order
                            jtypes = topmatch + bottommatch
                            rmatches = toptext + bottomtext
                            
                            # insert our order
                            meta['_jupyter_types_experimental']=jtypes
                            result['matches']=rmatches

                            if len(toptext) > 0 and dotpos < len(code):
                                # indicate end of list in some instances to force regen
                                # to move out list back to the top during partial completions like ds.a<tab>
                                msg = '---'
                                jtypes.append({'start': 0, 'end': 0, 'text': msg, 'type': 'endlist'})
                                rmatches.append(msg)

                except Exception as e:
                    # indicate we crashed so user can report
                    jtypes = meta.get('_jupyter_types_experimental', None)
                    msg = 'CRASHRIPTABLE'
                    jtypes.insert(0, {'start': 0, 'end': 0, 'text': msg, 'type': f'{e}'})
                    result['matches'].insert(0, msg)

        return result

    # ---- start of main code for autocomplete --------
    import IPython
    from ipykernel import ipkernel
    _ipcompleter = IPython.get_ipython().Completer
    Hooker._ipcompleter = _ipcompleter

    # check version, we only support 7
    version = IPython.version_info[0]
    if version != 7:
        return

    # caller may optionally set jedi or greedy
    if jedi is True or jedi is False: _ipcompleter.use_jedi=jedi
    if greedy is True or greedy is False: _ipcompleter.greedy=greedy

    if hook:
        # for jupyter lab
        if Hooker._orig_do_complete is None:
            Hooker._orig_do_complete = ipkernel.IPythonKernel.do_complete            
            setattr(ipkernel.IPythonKernel, 'do_complete', _riptable_do_complete)

        # for console text (not jupyter lab)
        if Hooker._orig_deduplicate is None:
            Hooker._putils=IPython.terminal.ptutils
            Hooker._orig_deduplicate = Hooker._putils._deduplicate_completions
            setattr(Hooker._putils, '_deduplicate_completions', _riptable_deduplicate_completions)
    else:
        # unhook ------
        if Hooker._orig_do_complete is not None:            
            setattr(ipkernel.IPythonKernel, 'do_complete', Hooker._orig_do_complete)
            Hooker._orig_do_complete = None

        if Hooker._orig_deduplicate is not None:            
            setattr(Hooker._putils, '_deduplicate_completions', Hooker._orig_deduplicate)
            Hooker._orig_deduplicate = None
