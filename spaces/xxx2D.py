import numpy as np
import gymnasium as gym
import gymnasium.spaces as spaces
from typing import Union, List, Tuple
from uhtk.siri.utils.lprint import lprint
from .xxx2D_api import ActionDiscretizer

def XXX2D(space: gym.Space) -> ActionDiscretizer:
    if isinstance(space, gym.spaces.Discrete):
        return D2D(space)
    elif isinstance(space, gym.spaces.Box):
        raise NotImplementedError("Box (non-discrete) spaces need a 'n_bins' param to cut, go use Box2D.")
    elif isinstance(space, gym.spaces.MultiDiscrete):
        # eg. MultiDiscrete [1, 2, 3, 4, 5] -> n_action: 1*2*3*4*5
        return MD2D(space)
    else:
        raise ValueError(f"Unsupported action space type: {type(space)}")

class MD2D(ActionDiscretizer): 
    def get_n_actions(space: gym.Space) -> int:
        assert isinstance(space, gym.spaces.MultiDiscrete)
        # eg. MultiDiscrete [1, 2, 3, 4, 5] -> n_action: 1*2*3*4*5
        return np.prod(space.nvec)

    def __init__(self, md: gym.spaces.MultiDiscrete):
        self.md = md
        self.nvec = md.nvec
        self.n = self.get_n_actions(md)
    
    def index_to_action(self, index):
        indices = np.array(np.unravel_index(int(index), self.nvec), dtype=self.md.dtype)
        return indices

class Box2D(ActionDiscretizer):
    def __init__(self, raw_action_space: spaces.Box, n_bins_per_dim: Union[np.ndarray, List]):
        assert isinstance(raw_action_space, spaces.Box)
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
        assert isinstance(self.n_actions, np.int32)
        self.n_actions = int(self.n_actions)

    def index_to_action(self, index):
        indices = np.unravel_index(int(index), self.n_bins + 1)  # Adjust to include original number of bins
        values = self.low + np.array(indices) * self.intervals
        return values

    # def get_multidiscrete_space(self):
    #     return spaces.MultiDiscrete(self.n_bins + 1)  # BUG

class D2D(ActionDiscretizer):  # aka. DiscreteActSpaceWrapper
    def __init__(self, raw_action_space):
        assert isinstance(raw_action_space, spaces.Discrete)
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

