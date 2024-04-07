from __future__ import absolute_import
import warnings
from .grid import IncrementalSamples4grid
from .prid import IncrementalSamples4prid
from .ilids import IncrementalSamples4ilids
from .viper import IncrementalSamples4viper
from .cuhk01 import IncrementalSamples4cuhk01
from .cuhk02 import IncrementalSamples4cuhk02
from .cuhk03 import IncrementalSamples4cuhk03
from .msmt17 import IncrementalSamples4msmt17
from .sensereid import IncrementalSamples4sensereid
from .market1501 import IncrementalSamples4market
from .dukemtmcreid import IncrementalSamples4duke
from .cuhksysu import IncrementalSamples4subcuhksysu
# from .mix import IncrementalSamples4mix


# from .dukemtmc import DukeMTMC
# from .market1501 import Market1501
# from .msmt17 import MSMT17
# from .cuhk03 import CUHK03
# from .cuhk01 import CUHK01
# from .cuhk_sysu import CUHK_SYSU
# from .grid import GRID
# from .sensereid import SenseReID
# from .uav_test import UAV_test
# from .uavhuman import UAVhuman
__factory = {
    'market1501': IncrementalSamples4market,
    'dukemtmc': IncrementalSamples4duke,
    'msmt17': IncrementalSamples4msmt17,
    'cuhk_sysu': IncrementalSamples4subcuhksysu,
    'cuhk03': IncrementalSamples4cuhk03,
    'cuhk01': IncrementalSamples4cuhk01,
    'grid': IncrementalSamples4grid,
    'sense': IncrementalSamples4sensereid,
    # 'mix': IncrementalSamples4mix,
    'ilids':IncrementalSamples4ilids,
    'viper':IncrementalSamples4viper,
    "prid":IncrementalSamples4prid,
    "cuhk02":IncrementalSamples4cuhk02
    # 'uav_test':UAV_test,
    # 'uavhuman':UAVhuman
}


def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. Can be one of 'viper', 'cuhk01', 'cuhk03',
        'market1501', and 'dukemtmc'.
    root : str
        The path to the dataset directory.
    split_id : int, optional
        The index of data split. Default: 0
    num_val : int or float, optional
        When int, it means the number of validation identities. When float,
        it means the proportion of validation to all the trainval. Default: 100
    download : bool, optional
        If True, will download the dataset. Default: False
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)


def get_dataset(name, root, *args, **kwargs):
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name, root, *args, **kwargs)
