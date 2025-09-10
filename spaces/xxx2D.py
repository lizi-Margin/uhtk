"""
    Author: sunhaocheng@bupt.edu.cn
    Description: Anything to Discrete
        1. Box2D
        2. MultiDiscete2D
        3. Discrete to Discrete (of course)
        4. list of Box, MultiDiscte, Discrete to Discrete
"""

import numpy as np
import gymnasium as gym
import gymnasium.spaces as spaces
from typing import Union, List, Tuple
from uhtk.print_pack import *
from uhtk.siri.utils.lprint import lprint
from uhtk.siri.utils.iterable_tools import iterable_prod
from .xxx2D_api import ActionDiscretizer

# def XXX2D(space: gym.Space) -> ActionDiscretizer:
#     if isinstance(space, gym.spaces.Discrete):
#         return D2D(space)
#     elif isinstance(space, gym.spaces.Box):
#         raise NotImplementedError("Box (non-discrete) spaces need a 'n_bins' param to cut, go use Box2D.")
#     elif isinstance(space, gym.spaces.MultiDiscrete):
#         # eg. MultiDiscrete [1, 2, 3, 4, 5] -> n_action: 1*2*3*4*5
#         return MD2D(space)
#     else:
#         raise ValueError(f"Unsupported action space type: {type(space)}")

# def XXX2D(space: gym.Space):
#     if isinstance(space, gym.spaces.Discrete):
#         return D2D
#     elif isinstance(space, gym.spaces.Box):
#         return Box2D
#     elif isinstance(space, gym.spaces.MultiDiscrete):
#         # eg. MultiDiscrete [1, 2, 3, 4, 5] -> n_action: 1*2*3*4*5
#         return MD2D
#     else:
#         raise ValueError(f"Unsupported action space type: {type(space)}")

def is_Box(space: gym.Space) -> bool:
    if isinstance(space, gym.spaces.Box):
        return True
    elif 'Box' in space.__class__.__name__:
        return True
    return False

def is_Discrete(space: gym.Space) -> bool:
    if isinstance(space, gym.spaces.Discrete):
        return True
    elif 'Discrete' in space.__class__.__name__:
        return True
    return False

def is_MultiDiscrete(space: gym.Space) -> bool:
    if isinstance(space, gym.spaces.MultiDiscrete):
        return True
    elif 'MultiDiscrete' in space.__class__.__name__:
        return True
    return False


class Anything2D(ActionDiscretizer):
    def __init__(self,spaces: list):
        super().__init__()
        if isinstance(spaces, gym.Space): spaces = [spaces]
        assert isinstance(spaces, list)

        self.adapters = []
        self.nvec = []
        # self.dtype = self.spaces[0].dtype
        # self.dtype = np.float32
        for space in spaces:
            if is_Discrete(space):
                self.adapters.append(D2D(space))
            elif is_Box(space):
                self.adapters.append(Box2D.auto_init(space))
            elif is_MultiDiscrete(space):
                self.adapters.append(MD2D(space))
            else:
                raise ValueError(f"Unsupported action space type: {type(space)}")
            self.nvec.append(self.adapters[-1].n_actions)
        self.n_actions = iterable_prod(self.nvec)
    
    def index_to_action(self, index):
        indices = np.array(np.unravel_index(int(index), self.nvec), dtype=np.float32)
        length = len(indices)
        assert length == len(self.adapters)
        # return np.array([self.adapters[i](indices[i]) for i in range(length)])
        return [self.adapters[i].index_to_action(indices[i]) for i in range(length)]


class DL2D(ActionDiscretizer):
    @staticmethod
    def check_MDL_space(spc: list):
        assert isinstance(spc, list), str(spc)
        n_list = []
        for disc in spc:
            if not is_Discrete(disc):
                raise ValueError(f"{disc} is a Discrete space?")
            assert disc.dtype == spc[0].dtype, f"{disc.dtype} {spc[0].dtype}"
            n_list.append(disc.n)
        return n_list
    @staticmethod
    def is_MDL_space(spc):
        try:
            DL2D.check_MDL_space(spc)
        except:
            return False
        return True
    def __init__(self, raw_space: list):
        super().__init__()
        self.nvec = np.array(self.check_MDL_space(raw_space))
        self.space_list = raw_space
        self.n_actions = iterable_prod(self.nvec)
        self.dtype = self.space_list[0].dtype
    
    def index_to_action(self, index):
        indices = np.array(np.unravel_index(int(index), self.nvec), dtype=self.dtype)
        return indices


class MD2D(ActionDiscretizer): 
    def get_n_actions(space: gym.Space) -> int:
        assert is_MultiDiscrete(space)
        # eg. MultiDiscrete [1, 2, 3, 4, 5] -> n_action: 1*2*3*4*5
        return np.prod(space.nvec)

    def __init__(self, md: gym.spaces.MultiDiscrete):
        self.md = md
        self.nvec = md.nvec
        self.n_actions = self.get_n_actions(md)
    
    def index_to_action(self, index):
        indices = np.array(np.unravel_index(int(index), self.nvec), dtype=self.md.dtype)
        return indices

class Box2D(ActionDiscretizer):
    @staticmethod
    def auto_init(raw_action_space: spaces.Box, n_bins_avg: int = 2):
        assert is_Box(raw_action_space)

        n_dim = len(raw_action_space.low)
        target_total_bins = n_bins_avg * n_dim  # 目标总数

        ranges = raw_action_space.high - raw_action_space.low
        ranges = np.clip(ranges, 1e-8, None)  # 防止除零

        weights = ranges / ranges.sum()
        n_bins_per_dim = weights * target_total_bins

        n_bins_per_dim = np.round(n_bins_per_dim).astype(int)
        n_bins_per_dim = np.maximum(n_bins_per_dim, 2)

        print黄(f"[Box2D] Warning: auto_init, total_bins={target_total_bins}, space={raw_action_space}, n_bins_per_dim={n_bins_per_dim}")

        return Box2D(raw_action_space, n_bins_per_dim)
        
    def __init__(self, raw_action_space: spaces.Box, n_bins_per_dim: Union[np.ndarray, List]):
        assert is_Box(raw_action_space)
        if isinstance(n_bins_per_dim, list):
            n_bins_per_dim = np.array(n_bins_per_dim)
        elif isinstance(n_bins_per_dim, int):
            n_bins_per_dim = np.array([n_bins_per_dim] * len(raw_action_space.low))
        assert len(n_bins_per_dim) == len(raw_action_space.low)

        self.raw_action_space = raw_action_space
        self.n_bins = np.array(n_bins_per_dim) - 1  # Adjust bins to include both lower and upper boundaries

        self.low = self.raw_action_space.low
        self.high = self.raw_action_space.high

        self.intervals = (self.high - self.low) / self.n_bins

        self.n_actions = np.prod(self.n_bins + 1)  # Adjust to include original number of bins
        assert isinstance(self.n_actions, (np.int32, np.int64, int)), f"{self.n_actions}, {self.n_actions.dtype}"
        self.n_actions = int(self.n_actions)

    def index_to_action(self, index):
        indices = np.unravel_index(int(index), self.n_bins + 1)  # Adjust to include original number of bins
        values = self.low + np.array(indices) * self.intervals
        return values

    # def get_multidiscrete_space(self):
    #     return spaces.MultiDiscrete(self.n_bins + 1)  # BUG

class D2D(ActionDiscretizer):  # aka. DiscreteActSpaceWrapper
    def __init__(self, raw_action_space):
        assert is_Discrete(raw_action_space)
        assert raw_action_space.start == 0
        self.raw_action_space = raw_action_space
        self.n_actions = raw_action_space.n

    def index_to_action(self, index):
        assert 0 <= index < self.n_actions
        return index



##########################################################################################################
# from SIRI
class wasd_Discretizer(ActionDiscretizer):
    """
        None
        w, a, s, d
        wa, wd, sa, sd
    """
    def __init__(self):
        self.n_actions = 9
        self.coverter = np.array([
            [0, 0, 0, 0],  # None
            [1, 0, 0, 0],  # w
            [0, 1, 0, 0],  # a
            [0, 0, 1, 0],  # s
            [0, 0, 0, 1],  # d
            [1, 1, 0, 0],  # wa
            [1, 0, 0, 1],  # wd
            [0, 1, 1, 0],  # sa
            [0, 0, 1, 1]  # sd
        ])

    def index_to_action(self, index):
        if isinstance(index, (list, np.ndarray,)):
            raise NotImplementedError("Go Imlement batch method")
        else:
            return self.index_to_action_(index)

    def index_to_action_(self, index):
        assert index >=0 and index < self.n_actions
        assert not isinstance(index, (list, np.ndarray,))
        return self.coverter[index].copy()

    def action_to_index_(self, action):
        assert isinstance(action, np.ndarray) and len(action.shape) == 1 and len(action) == 4
        ret = None
        for i in range(len(self.coverter)):
            if np.array_equal(action, self.coverter[i]):
                ret = i
        if ret is None:
            lprint(self, f"ws or ad action {str(action)}, aborted")
            ret = 0
        return ret
    
    def action_to_index(self, action):
        assert isinstance(action, np.ndarray)
        if len(action.shape) == 2:
            ret = np.zeros(action.shape[0], dtype=np.int32)
            for i in range(len(ret)):
                ret[i] = self.action_to_index_(action[i])
            return ret
        else:
            return self.action_to_index_(action)

    def get_discrete_space(self):
        return spaces.Discrete(self.n_actions)


class Float2D_presets():  # no boundaries, presets needed
    def __init__(self, presets: Union[np.ndarray, list]):  # presets: [-0.5, 0., 0.5]
        if isinstance(presets, list):
            presets = np.array(presets)
        assert isinstance(presets, np.ndarray)
        assert np.all(np.diff(presets) <= 0), f"presets 必须是单向下降的, {repr(presets)}"
        assert len(presets.shape) == 1
        self.presets = presets
        self.n_actions = len(presets)
        # from uhtk.imitation.filter import mouse_filter
        # self.filter = mouse_filter(**FILTER)

    def index_to_action(self, index):
        if isinstance(index, (list, np.ndarray,)):
            raise NotImplementedError("Go Imlement batch method")
        else:
            return self.index_to_action_(index)

    def index_to_action_(self, index: int):
        if index < 0 or index >= self.n_actions:
            raise IndexError(f"索引 {index} 超出范围 [0, {self.n_actions - 1}]")
        return self.presets[index]

    def action_to_index_(self, continuous):
        assert isinstance(continuous, (int, float, np.float32, np.float64))
        # continuous = self.filter.step(continuous)
        diffs = np.abs(self.presets - continuous)
        return np.argmin(diffs).astype(np.int32)
    
    def action_to_index(self, continuous):
        if isinstance(continuous, (list, np.ndarray)):
            assert len(continuous.shape) == 1
            ret = np.zeros(continuous.shape, dtype=np.int32)
            for i in range(len(ret)):
                ret[i] = self.action_to_index_(continuous[i])
            return ret
        else:
            return self.action_to_index_(continuous)

