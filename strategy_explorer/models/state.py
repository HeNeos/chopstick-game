from dataclasses import dataclass
from enum import Enum
from typing import NewType

HashState = NewType("HashState", int)
Hand = NewType("Hand", int)
PlayerHands = tuple[Hand, Hand]


class Players(Enum):
    Human = 1
    Computer = -1


@dataclass
class State:
    computer_hands: tuple[Hand, Hand]
    human_hands: tuple[Hand, Hand]
    turn: Players
    distance: int
