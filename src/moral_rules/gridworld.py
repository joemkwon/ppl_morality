"""The grid-world: cell types, moves, BFS path-finding, and location sampling.

A grid-world is a list-of-lists of single-character cells:

* ``"S"`` -- sidewalk (rule-following agents may only walk here)
* ``"G"`` -- grass    (walking here is the rule violation)
* ``"F"`` -- the goal / destination cell

Movement is 8-connected (4 cardinal + 4 diagonal). Two BFS path-finders are
provided: one restricted to sidewalk (the strict rule-following plan) and one
that ignores the grass/sidewalk distinction (the shortest possible plan).

Note on a faithfully-preserved cleanup: the original research BFS seeded its
visited set with ``set((sx, sy))`` -- a set of the two integer coordinates
rather than ``{(sx, sy)}``. Because BFS still dequeues the goal on a shortest
path first, returned path *lengths* are unaffected, so this release uses the
correct ``{(sx, sy)}`` seed. This is verified to leave every trajectory's
shortcutiness unchanged (see ``tests/test_gridworld.py``).
"""

from __future__ import annotations

import random
from collections import deque

Coord = tuple[int, int]
Step = dict  # {"coordinate": (x, y), "type": "S"|"G"|"F", "action": str | None}

# (name, dx, dy). Diagonal moves are the four hyphenated names.
_DIRECTIONS: list[tuple[str, int, int]] = [
    ("west", -1, 0),
    ("east", 1, 0),
    ("north", 0, -1),
    ("south", 0, 1),
    ("north-west", -1, -1),
    ("north-east", 1, -1),
    ("south-west", -1, 1),
    ("south-east", 1, 1),
]
CARDINAL_ACTIONS = {"north", "east", "west", "south"}


def get_neighbors(
    gridworld: list[list[str]], x: int, y: int, include_grass: bool = False
) -> list[tuple[int, int, str]]:
    """In-bounds neighbours of ``(x, y)``.

    With ``include_grass=False`` only sidewalk (``"S"``) cells are returned --
    this is what makes a path "rule-following".
    """
    height = len(gridworld)
    width = len(gridworld[0])
    neighbors: list[tuple[int, int, str]] = []
    for direction, dx, dy in _DIRECTIONS:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            if include_grass or gridworld[ny][nx] == "S":
                neighbors.append((nx, ny, direction))
    return neighbors


def _bfs(gridworld: list[list[str]], start: Coord, goal: Coord, include_grass: bool) -> list[Step]:
    start_x, start_y = start
    goal_x, goal_y = goal
    queue: deque[tuple[int, int, str | None, list[Step]]] = deque([(start_x, start_y, None, [])])
    visited: set[Coord] = {(start_x, start_y)}

    while queue:
        x, y, action, path = queue.popleft()
        if (x, y) == (goal_x, goal_y):
            path.append({"coordinate": (x, y), "type": gridworld[y][x], "action": action})
            return path
        for nx, ny, direction in get_neighbors(gridworld, x, y, include_grass=include_grass):
            if (nx, ny) not in visited:
                visited.add((nx, ny))
                new_path = path + [
                    {"coordinate": (x, y), "type": gridworld[y][x], "action": direction}
                ]
                queue.append((nx, ny, direction, new_path))
    return []  # no path found


def full_rule_following_trajectory(
    gridworld: list[list[str]], start: Coord, goal: Coord
) -> list[Step]:
    """Shortest path that stays entirely on the sidewalk (the strict rule plan)."""
    return _bfs(gridworld, start, goal, include_grass=False)


def shortest_path_trajectory(gridworld: list[list[str]], start: Coord, goal: Coord) -> list[Step]:
    """Shortest path ignoring the rule (may cut across grass)."""
    return _bfs(gridworld, start, goal, include_grass=True)


def perimeter_sidewalk_squares(gridworld: list[list[str]]) -> list[Coord]:
    """All sidewalk cells on the outer ring -- the valid start/goal locations."""
    height = len(gridworld)
    width = len(gridworld[0])
    squares: list[Coord] = []
    for x in range(width):
        if gridworld[0][x] == "S":
            squares.append((x, 0))
        if gridworld[height - 1][x] == "S":
            squares.append((x, height - 1))
    for y in range(1, height - 1):
        if gridworld[y][0] == "S":
            squares.append((0, y))
        if gridworld[y][width - 1] == "S":
            squares.append((width - 1, y))
    return squares


def sample_location(gridworld: list[list[str]]) -> Coord | None:
    """Uniformly sample a perimeter sidewalk cell (seed via ``config.seed_everything``)."""
    squares = perimeter_sidewalk_squares(gridworld)
    return random.choice(squares) if squares else None
