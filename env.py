from gym import Env, spaces
from gym.utils import seeding
import numpy as np
from typing import Any, Dict, Tuple

from config import MAX_REWARD, MIN_REWARD, NUMBER_OF_APLHABETS, NUMBER_OF_TRIES, WORD_LENGTH
from utils import load_wordle_data


class WordleEnv(Env):
    def __init__(self) -> None:
        super(WordleEnv, self).__init__()

        # rendering pipeline
        self.metadata = {
            "render.modes": ["human", "ansi"]
        }

        # "actions": 0-25 -> 26 alphabets
        self.action_shape = NUMBER_OF_APLHABETS
        self.action_space = spaces.Discrete(self.action_shape)

        # "letters": 5x6 grid of alphabets
        #       -1 -> No letter
        #       0-25 -> Letter index
        # "masks": 5x6 grid of position masks
        #       -1  -> Not present in the word
        #        0 -> Present in the word but in another position
        #        1 -> Present in the word and in the correct position
        self.observation_shape = (NUMBER_OF_TRIES, WORD_LENGTH)
        self.observation_space = spaces.Dict(spaces={
            "letters": spaces.Box(low=-1, high=26, shape=self.observation_shape, dtype=int),
            "masks": spaces.Box(low=-1, high=2, shape=self.observation_shape, dtype=int)
        })
        self._observation_space_base = np.ones(
            self.observation_shape, dtype=int)

        # Rewards are from -1 to 1
        self.reward_range = (MIN_REWARD, MAX_REWARD)

        # Parameters
        self._state = None
        self._current_position = None
        self._current_answer = None

        self._allowed_guesses, self._possible_answers = load_wordle_data()
        self._number_of_answers = len(self._possible_answers)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self) -> Any:
        self._state = {
            "letters": np.ones_like(self._observation_space_base) * -1,
            "masks": np.ones_like(self._observation_space_base) * -1
        }
        self._current_position = (0, 0)
        self._current_answer = self._possible_answers[
            self.np_random.randint(self._number_of_answers)
        ]
        self._info = {}

        return self._state

    def _reset_row(self, row: int) -> None:
        for key in ("letters", "masks"):
            self._state[key][row, :] = -1 * \
                np.ones_like(self._state["letters"][row, :])

        if row == 0:
            self._current_position = (0, 0)
        else:
            self._current_position = (row - 1, 0)

    def _evaluate_row_mask(self, row: int) -> float:
        reward = 0.0
        for i, char in enumerate(self._get_word(row)):
            if char == self._current_answer[i]:
                self._state["masks"][row][i] = 1
                reward += 0.2
            elif char in self._current_answer:
                self._state["masks"][row][i] = 0
                reward += 0.1
            else:
                self._state["masks"][row][i] = -1
                reward += 0.0

        return reward

    def _get_word(self, row: int) -> str:
        return ''.join([chr(65 + char) for char in self._state[row, :]])

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict[str, Any]]:
        assert self.action_space.contains(
            action), f"{action} ({type(action)}) is not a valid action."
        assert self._state is None, "Environment has not been reset() before calling step()."

        current_row, current_col = self._current_position
        self._state["letters"][current_row][current_col] = action

        if current_col == WORD_LENGTH - 1:
            word = self._get_word(current_row)

            if word == self._current_answer:
                # Give full reward of 1.0 for correct answer
                return self._state, 1.0, True, self._info

            if word not in self._allowed_guesses:
                # Clear the row and send a negative reward of -0.75
                self._reset_row(current_row)
                return self._state, -0.75, False, self._info

            if current_row == NUMBER_OF_TRIES - 1 and word != self._current_answer:
                # All tries exhausted, send maximum negative reward of -1.0
                return self._state, -1.0, True, self._info

            # Get reward for each action based on the mask
            self._current_position = (current_row + 1, 0)
            return self._state, self._evaluate_row_mask(current_row), False, self._info
        else:
            # A small negative reward encouraging shorter plays
            return self._state, -0.01, False, self._info
