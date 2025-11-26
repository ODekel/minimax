import math
from typing import TypeVar, Callable, Optional

import numpy as np

from games import Game


TState = TypeVar("TState")
TAction = TypeVar("TAction")
TPlayer = TypeVar("TPlayer")
_display_type = Callable[[TState, TState, TAction, TPlayer, int], None]


_rng = np.random.default_rng()


def play_full_game(game: Game[TState, TAction, TPlayer], first_player: TPlayer, depth: int,
                   display: _display_type=None) -> TState:
    next_player = first_player
    next_state = game.start_state
    state_count = 0
    next_action = minimax(game, next_state, next_player, depth)
    while next_action is not None or not game.is_terminal(next_state):
        prev_state, prev_player = next_state, next_player
        if next_action is not None:
            next_state = game.act(next_state, next_action, next_player)
        next_player = game.next_player(next_state, next_player)
        state_count += 1
        if display is not None:
            display(prev_state, next_state, next_action, prev_player, state_count)
        next_action = minimax(game, next_state, next_player, depth)
    return next_state


def minimax(game: Game[TState, TAction, TPlayer], state: TState, player: TPlayer, depth: int) -> Optional[TAction]:
    return _find_max(game, state, player, player, depth)[-1]


def _find_max(game: Game[TState, TAction, TPlayer], state: TState, player: TPlayer, original_player: TPlayer, depth: int):
    if game.is_terminal(state):
        return game.utility(state, original_player), None
    if depth == 0:
        return game.evaluate(state, original_player), None

    actions = game.get_actions(state, player)
    if len(actions) == 0:
        return _find_next_depth(game, state, player, original_player, depth)[0], None

    max_eval = -math.inf
    max_action = None
    for action in actions:
        result = game.act(state, action, player)
        next_eval, _ = _find_next_depth(game, result, player, original_player, depth)

        if next_eval > max_eval:
            max_eval = next_eval
            max_action = action
        elif next_eval == max_eval:
            if _rng.random() < 0.5:
                max_action = action

    return max_eval, max_action


def _find_min(game: Game[TState, TAction, TPlayer], state: TState, player: TPlayer, original_player: TPlayer, depth: int):
    if game.is_terminal(state):
        return game.utility(state, original_player), None
    if depth == 0:
        return game.evaluate(state, original_player), None

    actions = game.get_actions(state, player)
    if len(actions) == 0:
        return _find_next_depth(game, state, player, original_player, depth)[0], None

    min_eval = math.inf
    min_action = None
    for action in actions:
        result = game.act(state, action, player)
        next_eval, _ = _find_next_depth(game, result, player, original_player, depth)

        if next_eval < min_eval:
            min_eval = next_eval
            min_action = action
        elif next_eval == min_eval:
            if _rng.random() < 0.5:
                min_action = action

    return min_eval, min_action


def _find_next_depth(game: Game[TState, TAction, TPlayer], state: TState, player: TPlayer, original_player: TPlayer,
                     depth: int):
    next_player = game.next_player(state, player)
    if next_player == original_player:
        return _find_max(game, state, next_player, original_player, depth - 1)
    else:
        return _find_min(game, state, next_player, original_player, depth - 1)
