from dataclasses import dataclass
import numpy as np

from .environment import AttentionEnv


@dataclass
class RLTransition:
    state: int
    action: int
    reward: float
    next_state: int
    done: bool


class RLAttentionWrapper:
    """
    Wrapper for RL training. Agent controls the platform's aggressiveness.
    Actions: 0 = decrease, 1 = keep same, 2 = increase
    State is discretized into bins of attention and boredom.
    """

    def __init__(self, episode_length: int = 300):
        self.env = AttentionEnv(max_time_steps=episode_length)
        self.episode_length = episode_length

    def reset(self):
        state = self.env.reset()
        return self._discretize_state(state)

    def step(self, action: int):
        # adjust aggressiveness based on action
        if action == 0:
            self.env.algo_aggressiveness = max(0.0, self.env.algo_aggressiveness - 0.1)
        elif action == 2:
            self.env.algo_aggressiveness = min(1.0, self.env.algo_aggressiveness + 0.1)
        # action 1 = no change

        state, done = self.env.step()

        # reward function: want high attention, low boredom/fatigue
        reward = (
            state.attention_level
            - 0.5 * (state.boredom + state.fatigue)
        ) / 100.0

        next_state_disc = self._discretize_state(state)
        return next_state_disc, reward, done, state

    def _discretize_state(self, state):
        """
        bin the continuous state into discrete buckets
        attention: low/med/high, boredom: low/high = 6 total states
        """
        att = state.attention_level
        bor = state.boredom

        if att < 30:
            att_bin = 0
        elif att < 70:
            att_bin = 1
        else:
            att_bin = 2

        bor_bin = 0 if bor < 50 else 1

        return att_bin * 2 + bor_bin  # combine into single index

    @property
    def n_states(self):
        return 6

    @property
    def n_actions(self):
        return 3
