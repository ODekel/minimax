import math
from typing import TypeVar

from games import Game


TState = TypeVar("TState")
TAction = TypeVar("TAction")
TPlayer = TypeVar("TPlayer")


def play_full_game(game: Game[TState, TAction, TPlayer], first_player: TPlayer, depth: int):
    next_player = first_player
    next_state = game.start_state
    next_action = minimax(game, next_state, next_player, depth)
    while next_action is not None:
        print(next_state)
        next_state, next_player = game.act(next_state, next_action, next_player)
        next_action = minimax(game, next_state, next_player, depth)
    return next_state


def minimax(game: Game[TState, TAction, TPlayer], state: TState, player: TPlayer, depth: int):
    e, a = _find_max(game, state, player, player, depth)
    print(e)
    return a


def _find_max(game: Game[TState, TAction, TPlayer], state: TState, player: TPlayer, original_player: TPlayer, depth: int):
    if game.is_terminal(state):
        return game.utility(state, original_player), None
    if depth == 0:
        return game.evaluate(state, original_player), None

    max_eval = -math.inf
    max_action = None
    for action in game.get_actions(state, player):
        result, next_player = game.act(state, action, player)
        if next_player == original_player:
            next_eval, next_action = _find_max(game, result, next_player, original_player, depth - 1)
        else:
            next_eval, next_action = _find_min(game, result, next_player, original_player, depth - 1)

        if next_eval > max_eval:
            max_eval = next_eval
            max_action = action

    return max_eval, max_action


def _find_min(game: Game[TState, TAction, TPlayer], state: TState, player: TPlayer, original_player: TPlayer, depth: int):
    if game.is_terminal(state):
        return game.utility(state, original_player), None
    if depth == 0:
        return game.evaluate(state, original_player), None

    min_eval = math.inf
    min_action = None
    for action in game.get_actions(state, player):
        result, next_player = game.act(state, action, player)
        if next_player == original_player:
            next_eval, next_action = _find_max(game, result, next_player, original_player, depth - 1)
        else:
            next_eval, next_action = _find_min(game, result, next_player, original_player, depth - 1)

        if next_eval < min_eval:
            min_eval = next_eval
            min_action = action

    return min_eval, min_action
