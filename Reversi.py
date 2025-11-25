import enum
from typing import Tuple, List, Callable, Union

import numpy as np
import numpy.typing as npt

import helper
from games import Game
from minimax import play_full_game


_state_type = npt.NDArray[np.int8]
_action_type = Tuple[np.int8, np.int8]
_player_type = np.int8


class TileState(enum.Enum):
    EMPTY = np.int8(0)
    BLACK = np.int8(1)
    WHITE = np.int8(-1)


DIRECTIONS: List[npt.NDArray[np.int8]] = [
    np.array([-1, -1], dtype=np.int8),
    np.array([-1, 0], dtype=np.int8),
    np.array([-1, 1], dtype=np.int8),
    np.array([0, -1], dtype=np.int8),
    np.array([0, 1], dtype=np.int8),
    np.array([1, -1], dtype=np.int8),
    np.array([1, 0], dtype=np.int8),
    np.array([1, 1], dtype=np.int8)]


class Reversi(Game[_state_type, _action_type, _player_type]):

    def __init__(self, start_state: _state_type, heuristic: Callable[[_state_type, _player_type], float]):
        super().__init__(start_state)
        self._heuristic = heuristic

        if start_state.shape[0] != start_state.shape[1]:
            raise ValueError("Reversi board must be square.")

    @property
    def side(self):
        return self._start_state.shape[0]

    def is_terminal(self, state: _state_type) -> bool:
        return not self._can_act(state, TileState.BLACK.value) and not self._can_act(state, TileState.WHITE.value)

    def evaluate(self, state: _state_type, player: _player_type) -> float:
        return self._heuristic(state, player)

    def utility(self, state: _state_type, player: _player_type) -> float:
        return state.sum() * np.sign(player)

    def get_actions(self, state: _state_type, player: _player_type) -> List[_action_type]:
        candidates = helper.indices_with_neighbor(state, TileState.EMPTY.value, -player)
        actions: List[_action_type] = []
        for cand_idx in range(candidates.shape[0]):
            cand: npt.NDArray[np.intp] = candidates[cand_idx]
            for direction in DIRECTIONS:
                if len(self._calc_flipped_in_dir(state, cand, player, direction)) > 0:
                    actions.append(tuple(cand))
                    break
        return actions

    def act(self, state: _state_type, action: _action_type, player: _player_type) -> Tuple[_state_type, _player_type]:
        flipped = []
        for direction in DIRECTIONS:
            flipped.extend(self._calc_flipped_in_dir(state, action, player, direction))

        new_state = state.copy()
        new_state[action] = player
        flip_mask = np.zeros_like(new_state, dtype=bool)
        flat_idx_arr = np.ravel_multi_index(np.array(flipped).transpose(), flip_mask.shape)
        np.ravel(flip_mask)[flat_idx_arr] = True
        new_state[flip_mask] = player
        next_player = -player if self._can_act(new_state, -player) else player
        return new_state, next_player

    @classmethod
    def _can_act(cls, state: _state_type, player: _player_type) -> bool:
        candidates = helper.indices_with_neighbor(state, TileState.EMPTY.value, -player)
        for cand_idx in range(candidates.shape[0]):
            for direction in DIRECTIONS:
                if len(cls._calc_flipped_in_dir(state, candidates[cand_idx], player, direction)) > 0:
                    return True
        return False

    @staticmethod
    def _calc_flipped_in_dir(state: _state_type, action: Union[_action_type, npt.NDArray[np.intp]], player: _player_type, direction: npt.NDArray[np.int8]) -> List[Tuple[np.intp, np.intp]]:
        flipped: List[Tuple[np.intp, np.intp]] = []
        loc = tuple(np.array(action) + direction)
        while 0 <= loc[0] < state.shape[0] and 0 <= loc[1] < state.shape[1]:
            if state[loc] == TileState.EMPTY.value:
                return []
            elif state[loc] == player:
                return flipped
            else:
                flipped.append(loc)
                loc = tuple(np.array(loc) + direction)
        return []


def difference_heuristic(state: _state_type, player: _player_type) -> float:
    return state.sum() * int(player)


@helper.static_vars(masks={})
def side_heuristic(state: _state_type, player: _player_type) -> float:
    side = state.shape[0]
    if side not in side_heuristic.masks:
        mask = np.zeros((side, side), dtype=bool)
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
        side_heuristic.masks[side] = mask
    return state[side_heuristic.masks[side]].sum() * int(player)


def corner_heuristic(state: _state_type, player: _player_type) -> float:
    return (state[0, 0] + state[0, -1] + state[-1, 0] + state[-1, -1]) * int(player)


@helper.static_vars(max_values={})
def combined_heuristic(state: _state_type, player: _player_type) -> float:
    side = state.shape[0]
    if side not in combined_heuristic.max_values:
        combined_heuristic.max_values[side] = ((side * 9) - 4) * side

    return (difference_heuristic(state, player) +
            side_heuristic(state, player) * side +
            corner_heuristic(state, player) * (side ** 2)) / combined_heuristic.max_values[side]


def create_start_state(board_size: int = 8) -> _state_type:
    state = np.full((board_size, board_size), TileState.EMPTY.value, dtype=np.int8)
    mid = board_size // 2
    state[mid - 1, mid - 1] = TileState.WHITE.value
    state[mid, mid] = TileState.WHITE.value
    state[mid - 1, mid] = TileState.BLACK.value
    state[mid, mid - 1] = TileState.BLACK.value
    return state


game = Reversi(create_start_state(), combined_heuristic)
end = play_full_game(game, TileState.BLACK.value, depth=4)

print("Game over.")
print(end)
print("Sum:" , end.sum())
