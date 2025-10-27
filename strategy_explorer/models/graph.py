from dataclasses import dataclass
from enum import Enum
from typing import NewType
from models.state import State, HashState

GraphEdge = NewType("GraphEdge", list[State])


@dataclass
class GraphNode:
    hash: HashState
    state: State


class GraphState(Enum):
    Winning = 1
    Losing = -1
    Drawing = 0


@dataclass
class Graph:
    nodes: list[GraphNode]
    edges: dict[HashState, GraphEdge]


@dataclass
class MarkovNode:
    node: GraphNode
    probability: float
    state: GraphState


@dataclass
class MarkovGraph:
    nodes: dict[HashState, MarkovNode]
    edges: dict[HashState, list[MarkovNode]]
