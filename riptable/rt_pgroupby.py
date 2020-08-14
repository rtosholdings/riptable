from .rt_groupbyops import GroupByOps
from .rt_groupby import GroupBy

#=====================================================================================================
#=====================================================================================================
class PGroupBy(GroupBy):
    """
    Parameters
    ----------
    dataset: Dataset
        The dataset object

    keys: list
        List of column names to groupby

    filter: array of bools
        Boolean mask array applied as filter before grouping

    return_all: bool
        Default to False. When set to True will return all the dataset columns for every operation.

    hintSize: int
        Hint size for the hash

    sort: bool
        Default to True.  Indicates 

    Notes
    -----
    None at this time.

    Properties
    ----------
    gbkeys:  dictionary of numpy arrays binned from
    isortrows: sorted index or None

    """
    DebugMode=False
    ShowEmpty =True

    TestCatGb = True

    def __init__(self, *args, **kwargs):
        super()._init(self, *args, **kwargs)

    #---------------------------------------------------------------
    def copy(self, deep = True):
        pass