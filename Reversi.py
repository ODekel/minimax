import enum
import sys
from typing import Tuple, List, Callable, Union

import numpy as np
import numpy.typing as npt

import helper
from games import Game, TState, TPlayer
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

    def __init__(self, start_state: _state_type, heuristic: Callable[[_state_type, _player_type, 'Reversi'], float]):
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
        return 0 # self._heuristic(state, player, self)

    def utility(self, state: _state_type, player: _player_type) -> float:
        return 0 # state.sum() * int(player)

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

    def act(self, state: _state_type, action: _action_type, player: _player_type) -> _state_type:
        flipped = []
        for direction in DIRECTIONS:
            flipped.extend(self._calc_flipped_in_dir(state, action, player, direction))

        new_state = state.copy()
        new_state[action] = player
        flip_mask = np.zeros_like(new_state, dtype=bool)
        flat_idx_arr = np.ravel_multi_index(np.array(flipped).transpose(), flip_mask.shape)
        np.ravel(flip_mask)[flat_idx_arr] = True
        new_state[flip_mask] = player
        return new_state

    def next_player(self, state: TState, player: TPlayer) -> TPlayer:
        return -player

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


def _heuristic(state: _state_type, player: _player_type, game: Reversi) -> float:
    corners = np.array(((0, 0), (0, -1), (-1, 0), (-1, -1)))
    corners_value = state[corners[:,0], corners[:,1]].sum() * (state.shape[0] ** 2)
    near_corners = np.array((((0, 1), (1, 0), (1, 1)),
                            ((0, -2), (1, -1), (1, -2)),
                            ((-2, 0), (-1, 1), (-2, 1)),
                            ((-2, -1), (-1, -2), (-2, -2))))
    near_corners_value = 0
    for row in range(corners.shape[0]):
        if state[tuple(corners[row])] == TileState.EMPTY.value:
            near_corners_value -= state[near_corners[row][:,0], near_corners[row][:,1]].sum()
    near_corners_value *= state.shape[0]
    mobility = len(game.get_actions(state, player)) - len(game.get_actions(state, -player))
    return (corners_value + near_corners_value + mobility) * int(player) / ((state.shape[0] ** 2) * 5)


def create_start_state(board_size: int = 8) -> _state_type:
    state = np.full((board_size, board_size), TileState.EMPTY.value, dtype=np.int8)
    mid = board_size // 2
    state[mid - 1, mid - 1] = TileState.WHITE.value
    state[mid, mid] = TileState.WHITE.value
    state[mid - 1, mid] = TileState.BLACK.value
    state[mid, mid - 1] = TileState.BLACK.value
    return state


def _display(prev_state: _state_type, state: _state_type, action: _action_type, player: _player_type,
             state_count: int) -> None:
    print()
    print(f"State {state_count - 1}")
    player_char = 'X' if player == TileState.BLACK.value else 'O'
    _display_state(prev_state)
    print(f"State {state_count}, Player {player_char} moved, Action: {action}")
    _display_state(state)
    x_count = np.sum(state == TileState.BLACK.value)
    o_count = np.sum(state == TileState.WHITE.value)
    total_count = x_count + o_count
    print(f"Result - Player X: {x_count} disks, "
          f"Player O: {o_count} disks, "
          f"Total: {total_count} disks")


@helper.static_vars(char_arrs={})
def _display_state(state: _state_type) -> None:
    side = state.shape[0]
    if side not in _display_state.char_arrs:
        _display_state.char_arrs[side] = np.zeros((side, side), dtype=np.uint8)
    chars = _display_state.char_arrs[side]
    chars[state == TileState.EMPTY.value] = ord('-')
    chars[state == TileState.BLACK.value] = ord('X')
    chars[state == TileState.WHITE.value] = ord('O')
    np.savetxt(sys.stdout, chars, fmt='%c', delimiter='')


reversi = Reversi(create_start_state(), _heuristic)
end = play_full_game(reversi, TileState.BLACK.value, depth=3, display=_display)

print("Game over.")
print(end)
print("Sum:" , end.sum())
