import os
import abc

import numpy as np


class Agent(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def act(self, t, obs_history, act_history):
        raise NotImplementedError

    def reset(self):
        pass

    def set_log_dir(self, log_dir):
        self.log_dir = log_dir
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)

    def set_goal(self, goal):
        self.goal = goal


class DummyAgent(Agent):
    def __init__(self, a_dim):
        self.a_dim = a_dim

    def act(self, t, obs_history, state_obs_history):
        return np.zeros(self.a_dim)


class PlanningAgent(Agent):
    def __init__(self, a_dim, optimizer, replan_interval, initial_action=None):
        self.plan = list()
        self.a_dim = a_dim
        self.steps_since_replan = 0
        self.optimizer = optimizer
        self.replan_interval = replan_interval
        self.log_dir = None
        self.goal = None
        self.actions = []
        if initial_action is None:
            self.initial_action = np.zeros(self.a_dim)
        else:
            self.initial_action = np.array(initial_action)

    def reset(self):
        self.plan = list()
        self.steps_since_replan = 1e10
        self.goal = None
        self.log_dir = None
        self.actions = []

    def act(self, t, obs_history, state_obs_history):
        if t == 0:
            self.plan = [self.initial_action]
        elif self.steps_since_replan >= self.replan_interval or len(self.plan) == 0:
            init_mean = None
            if len(self.plan) > 0:
                init_mean = self.plan
            self.plan = self.optimizer.plan(
                t,
                self.log_dir,
                obs_history,
                state_obs_history,
                self.actions,
                self.goal,
                init_mean=init_mean,
            )
            self.steps_since_replan = 0
        action = self.plan[0]
        self.actions.append(np.copy(action))
        self.plan = self.plan[1:]
        self.steps_since_replan += 1
        return action


class SimplePlanningAgent(Agent):
    """Lightweight planning agent for MPC control.
    
    This agent uses a trajectory optimizer (e.g., CEM) to plan action sequences
    that reach a specified goal. It maintains action history for temporal context
    and supports periodic replanning.
    """
    
    def __init__(self, a_dim, optimizer, replan_interval, initial_action=None, num_context=2, use_initial_action=True):
        """Initialize planning agent.
        
        Args:
            a_dim: Action dimension
            optimizer: Trajectory optimizer (e.g., CEMOptimizer)
            replan_interval: Number of steps between replanning
            initial_action: Initial action to execute (defaults to zero)
            num_context: Number of historical actions for temporal context
        """
        self.plan = list()
        self.a_dim = a_dim
        self.steps_since_replan = 0
        self.optimizer = optimizer
        self.replan_interval = replan_interval
        self.num_context = num_context
        self.log_dir = None
        self.goal = None
        self.use_initial_action = use_initial_action
        if initial_action is None:
            self.initial_action = np.zeros(self.a_dim)
        else:
            self.initial_action = np.array(initial_action)
        # Pre-fill action history with initial actions for temporal context
        self.actions = [self.initial_action.copy() for _ in range(self.num_context)]

    def reset(self):
        """Reset agent state for new episode."""
        self.plan = list()
        self.steps_since_replan = int(1e10)
        self.goal = None
        self.log_dir = None
        self.actions = [self.initial_action.copy() for _ in range(self.num_context)]

    def act(self, t, obs_history, state_obs_history):
        """Select action for current timestep.
        
        Args:
            t: Current timestep
            obs_history: Historical observations
            state_obs_history: Historical states
            
        Returns:
            Action to execute
        """
        # At t=0, optionally use initial action
        if t == 0 and self.use_initial_action:
            self.plan = [self.initial_action]
        elif t == 0:
            self.plan = self.optimizer.plan(
                t,
                self.log_dir,
                obs_history,
                state_obs_history,
                self.actions,
                self.goal,
                init_mean=None,
            )
            self.steps_since_replan = 0
        # Replan if interval reached or plan exhausted
        elif self.steps_since_replan >= self.replan_interval or len(self.plan) == 0:
            init_mean = self.plan if len(self.plan) > 0 else None
            self.plan = self.optimizer.plan(
                t,
                self.log_dir,
                obs_history,
                state_obs_history,
                self.actions,
                self.goal,
                init_mean=init_mean,
            )
            self.steps_since_replan = 0
        
        # Execute first action in plan
        action = self.plan[0]
        self.actions.append(np.copy(action))
        self.plan = self.plan[1:]
        self.steps_since_replan += 1
        return action


class RandomAgent(Agent):
    def __init__(
        self,
        optimizer,
        replan_interval,
        action_mean,
        action_std,
        a_dim,
        initial_action=None,
    ):
        self.plan = list()
        self.a_dim = a_dim
        self.optimizer = optimizer
        self.log_dir = None
        self.goal = None
        self.actions = []
        self.action_mean, self.action_std = np.array(action_mean), np.array(action_std)
        if initial_action is None:
            self.initial_action = np.zeros(self.a_dim)
        else:
            self.initial_action = np.array(initial_action)

    def reset(self):
        self.plan = list()
        self.steps_since_replan = 1e10
        self.goal = None
        self.log_dir = None
        self.actions = []

    def set_goal(self, goal):
        self.goal = goal

    def get_random_action(self):
        return np.random.normal(size=self.a_dim) * self.action_std + self.action_mean

    def act(self, t, obs_history, state_obs_history):
        if t == 0:
            self.plan = [self.initial_action]
        else:
            self.plan = [self.get_random_action()]
        action = self.plan[0]
        if action[-1] > 0:
            action[-1] = 1
        else:
            action[-1] = -1
        self.actions.append(np.copy(action))
        self.plan = self.plan[1:]
        return action


__all__ = [
    "Agent",
    "DummyAgent",
    "PlanningAgent",
    "SimplePlanningAgent",
    "RandomAgent",
]
