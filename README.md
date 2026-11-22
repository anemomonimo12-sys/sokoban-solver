# hallway-rooms push-A* — a Sokoban solver

An entry for make.fun's **sokoban-solver-01** competition: `main.ts` (Node 22,
no dependencies) reads each instance at `/input/<id>/level.txt` and writes the
player's moves to `/output/<id>/solution.txt`, solving every level in the
fewest pushes it can.

## How it works

1. **Split the level into rooms.** Drawn levels are a one-square hallway with
   self-contained rooms off it: a box can never travel between rooms, because
   crossing would strand it on a hallway square, and every hallway square is
   a *dead square* (a box standing there can be pushed to no goal). Cutting
   the dead squares of the player's row away leaves one floor component per
   room, so the level's push optimum is the sum of its rooms' optima.
2. **Solve each room push-optimally with A\* over pushes.** A state is the
   room's boxes plus the player's reachable region (flood fill, named by its
   lowest square); one expansion is one push, and the walk between pushes is
   filled in afterwards by shortest path. The heuristic is a box-to-goal
   assignment bound built from lone-box push distances (reverse pull search
   per goal, cheapest distinct-goal assignment by subset DP). Dead-square
   and frozen 2×2-block pruning keep the search small. The bound never
   overstates, so the first solved state the search settles is optimal.
3. **Stitch the rooms together** in board order — other rooms' boxes stand
   as walls (on their goals once done), the player walks on from wherever
   the previous room left it.

The search is budgeted by expansions counted, never by the clock, and every
tie breaks by square and direction order, so repeated runs write
byte-identical answers. A level that does not split into rooms falls back to
the same A\* over the whole board.

## Layout

- `main.ts` — the whole solver.
- `LICENSE` — MIT.
