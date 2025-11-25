from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Tuple

TState = TypeVar("TState")
TAction = TypeVar("TAction")
TPlayer = TypeVar("TPlayer")


class Game(Generic[TState, TAction, TPlayer], ABC):

    def __init__(self, start_state: TState):
        self._start_state = start_state

    @property
    def start_state(self) -> TState:
        return self._start_state

    @abstractmethod
    def is_terminal(self, state: TState) -> bool:
        pass

    @abstractmethod
    def evaluate(self, state: TState, player: TPlayer) -> float:
        pass

    @abstractmethod
    def utility(self, state: TState, player: TPlayer) -> float:
        pass

    @abstractmethod
    def get_actions(self, state: TState, player: TPlayer) -> List[TAction]:
        pass

    @abstractmethod
    def act(self, state: TState, action: TAction, player: TPlayer) -> Tuple[TState, TPlayer]:
        pass
