import enum
import sys
from typing import Tuple, List, Callable, Union

import numpy as np
import numpy.typing as npt

import helper
from games import Game, TState, TPlayer
from minimax import play_full_game, play_methodical

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
        return self._heuristic(state, player, self)

    def utility(self, state: _state_type, player: _player_type) -> float:
        return state.sum() * int(player)

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
    def _calc_flipped_in_dir(state: _state_type, action: Union[_action_type, npt.NDArray[np.intp]],
                             player: _player_type, direction: npt.NDArray[np.int8]) -> List[Tuple[np.intp, np.intp]]:
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


def _display_all_actions(game: Reversi, first_player: _player_type, disk_count: int) -> None:
    state, player = play_methodical(game, first_player, max_turns=disk_count - 4)
    for action in game.get_actions(state, player):
        _display(state, game.act(state, action, player), action, player, disk_count + 1)


def _display_methodical(game: Reversi, first_player: _player_type, actions_to_display: int) -> None:
    state, player = play_methodical(game, first_player, display=_display, max_turns=actions_to_display)
    end_state = play_methodical(Reversi(state, _heuristic), player)[0]
    print()
    _display_game_over(end_state)


@helper.static_vars(sides={})
def _heuristic(state: _state_type, player: _player_type, game: Reversi) -> float:
    side = state.shape[0]
    sides = _heuristic.sides.get(side)
    if sides is None:
        sides = np.ones((side, side), dtype=bool)
        sides[1:-1, 1:-1] = False
        _heuristic.sides[side] = sides
    corners = np.array(((0, 0), (0, -1), (-1, 0), (-1, -1)))
    corners_value = state[corners[:, 0], corners[:, 1]].sum() * (side ** 2)
    near_corners = np.array((((0, 1), (1, 0), (1, 1)),
                            ((0, -2), (1, -1), (1, -2)),
                            ((-2, 0), (-1, 1), (-2, 1)),
                            ((-2, -1), (-1, -2), (-2, -2))))
    near_corners_value = 0
    for row in range(corners.shape[0]):
        if state[tuple(corners[row])] == TileState.EMPTY.value:
            near_corners_value -= state[near_corners[row][:, 0], near_corners[row][:, 1]].sum()
    near_corners_value *= side
    sides_value = state[sides].sum() * (side / 2)
    mobility = len(game.get_actions(state, TileState.BLACK.value)) - len(game.get_actions(state, TileState.WHITE.value))
    return (corners_value + near_corners_value + sides_value + mobility) * int(player) / ((side ** 2) * 9)


def _create_start_state(board_size: int = 8) -> _state_type:
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
    _display_state_result(state)


def _display_state_result(state: _state_type) -> None:
    x_count = np.sum(state == TileState.BLACK.value)
    o_count = np.sum(state == TileState.WHITE.value)
    total_count = x_count + o_count
    print(f"Result - Player X: {x_count} disks, "
          f"Player O: {o_count} disks, "
          f"Total: {total_count} disks")


def _display_game_over(state: _state_type) -> None:
    print()
    print("GAME OVER")
    _display_state(state)
    _display_state_result(state)
    score = np.sum(state == TileState.BLACK.value) - np.sum(state == TileState.WHITE.value)
    winner = 'BLACK' if score > 0 else 'WHITE' if score < 0 else 'DRAW'
    print(f"WINNER: {winner}")


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


args = sys.argv[1:]
if not args:
    print('Please provider options')
    exit(0)

reversi = Reversi(_create_start_state(), _heuristic)
if args[0] == '--displayAllActions':
    if len(args) == 1:
        print('Please provider actions num')
        exit(0)
    _display_all_actions(reversi, TileState.BLACK.value, int(args[1]))
elif args[0] == '--methodical':
    if len(args) == 1:
        print('Please provider actions num')
        exit(0)
    _display_methodical(reversi, TileState.BLACK.value, int(args[1]))
elif args[0] == 'H':
    if len(args) == 1:
        depth = 1
    elif len(args) == 2:
        print('Please provide depth')
        exit(0)
    else:
        depth = int(args[2])
    end = play_full_game(reversi, TileState.BLACK.value, depth=depth)[0]
    _display_game_over(end)
