"""
8-Puzzle Solver using A* Search
================================

A self-contained Python implementation suitable for Google Colab or GitHub.

Default example:
    Start: (1, 2, 3,
            4, 0, 5,
            6, 7, 8)

    Goal:  (1, 2, 3,
            4, 5, 6,
            7, 8, 0)

The blank tile is represented by 0. A* uses:
    f(n) = g(n) + h(n)

where:
    g(n) = number of moves from the start state
    h(n) = Manhattan-distance heuristic

Run:
    python 8_puzzle_astar.py

You can also import this file into a notebook and call:
    result = a_star([1, 2, 3, 4, 0, 5, 6, 7, 8], size=3)
"""

from heapq import heappop, heappush
from itertools import count
from math import isqrt
from typing import Dict, List, Optional, Sequence, Tuple

State = Tuple[int, ...]


def create_goal(size: int = 3) -> State:
    """Create the standard goal state for an N x N puzzle."""
    if size < 2:
        raise ValueError("Puzzle size must be at least 2.")
    return tuple(range(1, size * size)) + (0,)


def validate_state(state: Sequence[int], size: int) -> State:
    """Validate a state and return it as an immutable tuple."""
    if size < 2:
        raise ValueError("Puzzle size must be at least 2.")

    expected = size * size
    if len(state) != expected:
        raise ValueError(
            f"A {size}x{size} puzzle needs {expected} values; got {len(state)}."
        )

    normalized = tuple(state)
    if set(normalized) != set(range(expected)):
        raise ValueError(
            f"State must contain every integer from 0 to {expected - 1} exactly once."
        )

    return normalized


def heuristic(state: State, size: int = 3, goal: Optional[State] = None) -> int:
    """
    Calculate the Manhattan-distance heuristic.

    The blank (0) is ignored. For each numbered tile, Manhattan distance is
    the number of row and column steps between its current and goal positions.
    """
    if goal is None:
        goal = create_goal(size)

    goal_positions = {tile: index for index, tile in enumerate(goal)}
    total_distance = 0

    for index, tile in enumerate(state):
        if tile == 0:
            continue

        goal_index = goal_positions[tile]
        current_row, current_col = divmod(index, size)
        goal_row, goal_col = divmod(goal_index, size)

        total_distance += abs(current_row - goal_row) + abs(current_col - goal_col)

    return total_distance


def get_neighbors(state: State, size: int = 3) -> List[Tuple[State, str]]:
    """
    Generate legal neighboring states.

    Returns a list of (new_state, move) pairs. The move describes the direction
    in which the blank tile moves: Up, Down, Left, or Right.
