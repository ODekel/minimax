import math
from typing import TypeVar, Callable, Optional, Tuple

import numpy as np

from games import Game


TState = TypeVar("TState")
TAction = TypeVar("TAction")
TPlayer = TypeVar("TPlayer")
_display_type = Callable[[TState, TState, TAction, TPlayer, int], None]

_rng = np.random.default_rng()


def play_full_game(game: Game[TState, TAction, TPlayer], first_player: TPlayer, depth: int,
                   display: _display_type = None, randomize_on_equal: bool = True, max_turns: int = math.inf
                   ) -> Tuple[TState, TPlayer]:
    return _play(game, first_player, lambda g, s, p: minimax(g, s, p, depth, randomize_on_equal), display, max_turns)


def minimax(game: Game[TState, TAction, TPlayer], state: TState, player: TPlayer, depth: int,
            randomize_on_equal: bool = True) -> Optional[TAction]:
    return _find_max(game, state, player, player, depth, randomize_on_equal)[-1]


def play_methodical(game: Game[TState, TAction, TPlayer], first_player: TPlayer, display: _display_type = None,
                    max_turns: int = math.inf) -> Tuple[TState, TPlayer]:
    return _play(game, first_player, _methodical_next_action, display, max_turns)


def _play(game: Game[TState, TAction, TPlayer], first_player: TPlayer,
          action_selector: Callable[[Game[TState, TAction, TPlayer], TState, TPlayer], Optional[TAction]],
          display: _display_type, max_turns: int) -> Tuple[TState, TPlayer]:
    next_player = first_player
    next_state = game.start_state
    state_count = 0
    next_action = action_selector(game, next_state, first_player)
    while (next_action is not None or not game.is_terminal(next_state)) and state_count < max_turns:
        prev_state, prev_player = next_state, next_player
        if next_action is not None:
            next_state = game.act(next_state, next_action, next_player)
        next_player = game.next_player(next_state, next_player)
        state_count += 1
        if display is not None:
            display(prev_state, next_state, next_action, prev_player, state_count)
        next_action = action_selector(game, next_state, next_player)
    return next_state, next_player


def _methodical_next_action(game: Game[TState, TAction, TPlayer], state: TState, player: TPlayer) -> Optional[TAction]:
    actions = game.get_actions(state, player)
    if not actions:
        return None
    return actions[0]


def _find(game: Game[TState, TAction, TPlayer], state: TState, player: TPlayer, original_player: TPlayer, depth: int,
          cmp: Callable[[float, float], bool], initial_value: float, randomize_on_equal: bool) -> Tuple[float, TAction]:
    if game.is_terminal(state):
        return game.utility(state, original_player), None
    if depth == 0:
        return game.evaluate(state, original_player), None

    actions = game.get_actions(state, player)
    if len(actions) == 0:
        return _find_next_depth(game, state, player, original_player, depth, randomize_on_equal)[0], None

    chosen_eval = initial_value
    chosen_action = None
    for action in actions:
        result = game.act(state, action, player)
        next_eval, _ = _find_next_depth(game, result, player, original_player, depth, randomize_on_equal)

        if cmp(next_eval, chosen_eval):
            chosen_eval = next_eval
            chosen_action = action
        elif next_eval == chosen_eval and randomize_on_equal:
            if _rng.random() < 0.5:
                chosen_action = action

    return chosen_eval, chosen_action


def _find_max(game, state, player, original_player, depth, randomize_on_equal):
    return _find(game, state, player, original_player, depth, lambda new, prev: new > prev, -math.inf,
                 randomize_on_equal)


def _find_min(game, state, player, original_player, depth, randomize_on_equal):
    return _find(game, state, player, original_player, depth, lambda new, prev: new < prev, math.inf,
                 randomize_on_equal)


def _find_next_depth(game: Game[TState, TAction, TPlayer], state: TState, player: TPlayer, original_player: TPlayer,
                     depth: int, randomize_on_equal: bool):
    next_player = game.next_player(state, player)
    if next_player == original_player:
        return _find_max(game, state, next_player, original_player, depth - 1, randomize_on_equal)
    else:
        return _find_min(game, state, next_player, original_player, depth - 1, randomize_on_equal)
