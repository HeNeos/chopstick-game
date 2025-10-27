import json
from models.state import State, Hand, PlayerHands, Players, HashState
from queue import Queue
from models.graph import (
    Graph,
    GraphState,
    GraphNode,
    GraphEdge,
    MarkovGraph,
    MarkovNode,
)

INITIAL_STATE: State = State(
    computer_hands=(Hand(1), Hand(1)),
    human_hands=(Hand(1), Hand(1)),
    turn=Players.Human,
    distance=0,
)


def is_win_state(state: State) -> bool:
    computer_defeat: bool = state.computer_hands[0] == state.computer_hands[1] == 0
    player_alive: bool = state.human_hands[0] != 0 or state.human_hands[1] != 0
    return computer_defeat and player_alive


def is_lose_state(state: State) -> bool:
    return is_win_state(
        State(
            computer_hands=state.human_hands,
            human_hands=state.computer_hands,
            turn=Players(state.turn.value),
            distance=state.distance,
        )
    )


def normalize_hand(hand: PlayerHands) -> PlayerHands:
    if hand[0] < hand[1]:
        return PlayerHands((Hand(hand[1]), Hand(hand[0])))
    return hand


def movements(
    attacker: PlayerHands, attacked: PlayerHands
) -> list[tuple[PlayerHands, PlayerHands]]:
    new_states_list: list[tuple[PlayerHands, PlayerHands]] = []
    for attacker_hand in attacker:
        if attacker_hand == 0:
            continue
        first_attacked_hand = attacked[0]
        second_attacked_hand = attacked[1]

        new_states_list.append(
            (
                attacker,
                PlayerHands(
                    (
                        Hand((first_attacked_hand + attacker_hand) % 5),
                        second_attacked_hand,
                    )
                ),
            )
        )

        if second_attacked_hand != 0:
            new_states_list.append(
                (
                    attacker,
                    PlayerHands(
                        (
                            first_attacked_hand,
                            Hand((second_attacked_hand + attacker_hand) % 5),
                        )
                    ),
                ),
            )

    if attacker[-1] == 0 and attacker[0] % 2 == 0:
        new_states_list.append(
            (PlayerHands((Hand(attacker[0] // 2), Hand(attacker[0] // 2))), attacked)
        )

    return [
        (normalize_hand(new_state[0]), normalize_hand(new_state[1]))
        for new_state in new_states_list
    ]


def encode_state(state: State) -> HashState:
    value: int = 0
    for player in [state.computer_hands, state.human_hands]:
        for hand in player:
            value *= 10
            value += hand

    value *= 10
    match state.turn:
        case Players.Human:
            value += 1
        case Players.Computer:
            value += 2

    return HashState(value)


def build_graph() -> Graph:
    queue: Queue[State] = Queue(maxsize=0)
    queue.put(INITIAL_STATE)
    visited: set[HashState] = {encode_state(INITIAL_STATE)}
    graph: Graph = Graph(nodes=[], edges={})
    while not queue.empty():
        u: State = queue.get()
        u_encoded: HashState = encode_state(u)
        graph.nodes.append(GraphNode(hash=u_encoded, state=u))
        if is_win_state(u) or is_lose_state(u):
            if u_encoded not in graph.edges:
                graph.edges[u_encoded] = GraphEdge([])
            # if is_lose_state(u):
            # print(u.distance)
            # print(u.distance)
            continue

        attacker: PlayerHands
        attacked: PlayerHands
        match u.turn:
            case Players.Human:
                attacker = u.human_hands
                attacked = u.computer_hands
            case Players.Computer:
                attacker = u.computer_hands
                attacked = u.human_hands

        next_v: list[tuple[PlayerHands, PlayerHands]] = movements(
            attacker=attacker, attacked=attacked
        )

        unique_movements: list[State] = []
        unique_movements_hash: set[HashState] = set()
        # print(u)
        for v in next_v:
            computer_hands: PlayerHands
            human_hands: PlayerHands
            match u.turn:
                case Players.Human:
                    computer_hands = v[1]
                    human_hands = v[0]
                case Players.Computer:
                    computer_hands = v[0]
                    human_hands = v[1]
            next_state: State = State(
                computer_hands=computer_hands,
                human_hands=human_hands,
                turn=Players(-u.turn.value),
                distance=u.distance + 1,
            )
            encoded_next_state: HashState = encode_state(next_state)
            if encoded_next_state in unique_movements_hash:
                continue
            unique_movements_hash.add(encoded_next_state)
            unique_movements.append(next_state)
            # print(next_state)
            if encoded_next_state in visited:
                continue
            queue.put(next_state)
            visited.add(encoded_next_state)

        # print("===============================")
        graph.edges[u_encoded] = GraphEdge(unique_movements)

    return graph


def reverse_graph(graph: Graph) -> Graph:
    reversed_edges: dict[HashState, GraphEdge] = {}
    for graph_node in graph.nodes:
        next_states = graph.edges[graph_node.hash]
        for next_state in next_states:
            encoded_state = encode_state(next_state)
            if encoded_state not in reversed_edges:
                reversed_edges[encoded_state] = GraphEdge([])
            reversed_edges[encoded_state].append(graph_node.state)

    return Graph(nodes=graph.nodes, edges=reversed_edges)


def get_all_lose_nodes(graph: Graph) -> list[GraphNode]:
    return [
        GraphNode(hash=graph_node.hash, state=graph_node.state)
        for graph_node in graph.nodes
        if is_lose_state(graph_node.state)
    ]


def get_all_win_nodes(graph: Graph) -> list[GraphNode]:
    return [
        GraphNode(hash=graph_node.hash, state=graph_node.state)
        for graph_node in graph.nodes
        if is_win_state(graph_node.state)
    ]


def bfs_multi_source(
    graph: Graph,
    reversed_graph: Graph,
    lose_sources: list[GraphNode],
    win_sources: list[GraphNode],
    iterations: int = 100,
) -> dict[HashState, MarkovNode]:
    winning_nodes = [
        MarkovNode(node=source, probability=1.0, state=GraphState.Winning)
        for source in lose_sources
    ]
    losing_nodes = [
        MarkovNode(node=source, probability=0.0, state=GraphState.Losing)
        for source in win_sources
    ]
    markov_probabilities: dict[HashState, MarkovNode] = {
        node.node.hash: node for node in winning_nodes + losing_nodes
    }
    for _iteration in range(iterations):
        queue: Queue[MarkovNode] = Queue(maxsize=0)
        visited: set[HashState] = set(
            node.node.hash for node in winning_nodes + losing_nodes
        )
        _ = [queue.put(node) for node in winning_nodes + losing_nodes]
        while not queue.empty():
            u: MarkovNode = queue.get()
            for next_state in graph.edges[u.node.hash]:
                next_node: GraphNode = GraphNode(
                    hash=encode_state(next_state), state=next_state
                )
                if next_node.hash in visited:
                    continue
                v_edges: GraphEdge = reversed_graph.edges[next_node.hash]
                visited_neighbors: set[HashState] = {
                    encode_state(w_state) for w_state in v_edges
                } & set(markov_probabilities.keys())
                # probabilities_sum: float = sum(
                #     markov_probabilities[neighbor].probability
                #     for neighbor in visited_neighbors
                # )
                # next_probability = (probabilities_sum / len(v_edges))

                probabilities: list[GraphState] = [
                    markov_probabilities[neighbor].state
                    for neighbor in visited_neighbors
                ]
                next_probability: float
                match u.node.state.turn:
                    case Players.Human:
                        next_probability = (
                            1.0 if GraphState.Winning in probabilities else 0.0
                        )
                    case Players.Computer:
                        next_probability = (
                            0.0 if GraphState.Losing in probabilities else 1.0
                        )

                next_node_state: GraphState
                if next_probability > 0.99:
                    next_node_state = GraphState.Winning
                elif next_probability < 0.01:
                    next_node_state = GraphState.Losing
                else:
                    next_node_state = GraphState.Drawing
                v: MarkovNode = MarkovNode(
                    node=next_node, probability=next_probability, state=next_node_state
                )
                queue.put(v)
                visited.add(next_node.hash)
                markov_probabilities[next_node.hash] = v

    return markov_probabilities


graph: Graph = build_graph()
lose_states: list[GraphNode] = get_all_lose_nodes(graph)
win_states: list[GraphNode] = get_all_win_nodes(graph)
reversed_graph: Graph = reverse_graph(graph)

markov_probabilities: dict[HashState, MarkovNode] = bfs_multi_source(
    graph=reversed_graph,
    reversed_graph=graph,
    lose_sources=lose_states,
    win_sources=win_states,
)

markov_graph: MarkovGraph = MarkovGraph(
    nodes=markov_probabilities,
    edges={
        state_hash: [
            markov_probabilities[encoded_state]
            for state in graph.edges[state_hash]
            if (encoded_state := encode_state(state))
            and encoded_state in markov_probabilities
        ]
        for state_hash, _ in markov_probabilities.items()
    },
)

json_graph = {
    state_hash: {
        "node": [
            markov_node.node.state.computer_hands,
            markov_node.node.state.human_hands,
        ],
        "edges": [
            {
                "probability": markov_neighbor.probability,
                "state": markov_neighbor.state.value,
                "node_hash": markov_neighbor.node.hash,
                "node": [
                    markov_neighbor.node.state.computer_hands,
                    markov_neighbor.node.state.human_hands,
                ],
            }
            for markov_neighbor in markov_graph.edges[state_hash]
        ],
    }
    for state_hash, markov_node in markov_graph.nodes.items()
}

with open("markov.json", "w") as f:
    json.dump(json_graph, f, indent=2)
