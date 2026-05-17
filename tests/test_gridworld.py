"""Grid-world / BFS invariants, incl. the faithfully-preserved cleanup check."""

from collections import deque

from moral_rules.gridworld import (
    full_rule_following_trajectory,
    get_neighbors,
    shortest_path_trajectory,
)


def _legacy_bfs(gridworld, start, goal, include_grass):
    """Replica of the ORIGINAL research BFS, with its ``set((sx, sy))`` quirk.

    The original seeded ``visited`` with a set of the two integer coordinates
    rather than ``{(sx, sy)}``. We reproduce that exactly here to prove the
    cleaned implementation returns the same path *lengths* on every map.
    """
    sx, sy = start
    gx, gy = goal
    queue = deque([(sx, sy, None, [])])
    visited = set((sx, sy))  # noqa: C405 -- intentional bug-for-bug replica
    while queue:
        x, y, action, path = queue.popleft()
        if (x, y) == (gx, gy):
            return path + [{"c": (x, y)}]
        for nx, ny, d in get_neighbors(gridworld, x, y, include_grass=include_grass):
            if (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny, d, path + [{"c": (x, y)}]))
    return []


def test_rule_following_stays_on_sidewalk(trajectories):
    for gw, traj in trajectories:
        start = traj[0]["coordinate"]
        end = traj[-1]["coordinate"]
        path = full_rule_following_trajectory(gw, start, end)
        assert path, "rule-following path must exist"
        assert path[-1]["coordinate"] == tuple(end)
        for step in path[:-1]:  # last step is the (possibly 'F') goal cell
            x, y = step["coordinate"]
            assert gw[y][x] == "S", "rule-following path must stay on sidewalk"


def test_cleaned_bfs_matches_legacy_lengths(trajectories):
    """The visited-set cleanup must not change any path length (=> any sigma)."""
    for gw, traj in trajectories:
        start, end = traj[0]["coordinate"], traj[-1]["coordinate"]
        for include_grass in (False, True):
            fn = shortest_path_trajectory if include_grass else full_rule_following_trajectory
            assert len(fn(gw, start, end)) == len(_legacy_bfs(gw, start, end, include_grass))


def test_shortest_is_no_longer_than_rule_following(trajectories):
    for gw, traj in trajectories:
        start, end = traj[0]["coordinate"], traj[-1]["coordinate"]
        assert len(shortest_path_trajectory(gw, start, end)) <= len(
            full_rule_following_trajectory(gw, start, end)
        )
