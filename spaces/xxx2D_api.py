import gymnasium.spaces as spaces
from abc import ABC, abstractmethod


#     if isinstance(raw_action_space, spaces.Box):
#     self = BoxDiscretizer(raw_action_space, n_bins_per_dim)
#     return self
# elif isinstance(raw_action_space, spaces.Discrete):
#     self = DiscreteActSpaceWrapper(raw_action_space)
#     return self
# else:
#     raise NotImplementedError

class ActionDiscretizer(ABC):
    def __init__(self):
        self.n_actions = 0
    
    @property
    def n(self):
        return self.n_actions

    def index_to_action(self, index):
        raise NotImplementedError

    def action_to_index(self,action):
        # seldom used
        raise NotImplementedError

    def get_discrete_space(self):
        return spaces.Discrete(self.n_actions)