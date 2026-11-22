// hallway-rooms push-A*: an original Sokoban solver written for make.fun's
// sokoban-solver-01 competition (MIT, see LICENSE).
//
// The idea, in three parts:
//
// 1. Split the level into rooms. The levels this competition draws are a
//    one-square hallway with self-contained rooms off it: a box can never
//    travel between rooms (to leave a room it would have to cross the
//    hallway, and every hallway square is a dead square — a box standing
//    there can be pushed to no goal). So the squares of the player's own
//    row that are dead are cut away, the floor that remains falls into one
//    component per room, and each component is solved on its own. The
//    level's push optimum is the sum of its rooms' optima.
//
// 2. Solve each room push-optimally with A* over pushes. A state is the
//    room's boxes plus the player's reachable region (named by its lowest
//    square, found by flood fill): a push is possible exactly when the
//    square behind the box lies in that region, so one expansion step is
//    one push and the walk between pushes is filled in afterwards. The
//    estimate is a box-to-goal assignment bound: for every goal, a reverse
//    search gives the fewest pushes that move a lone box from any square
//    onto it, and the bound is the cheapest assignment of distinct goals
//    to the boxes under those distances (a subset DP; with more goals
//    than a DP table holds, the sum of each box's own minimum, which is
//    also a lower bound). Two deadlocks are pruned: a box pushed onto a
//    dead square, and a frozen 2 x 2 block of walls and boxes holding a
//    box off its goal. The estimate never overstates, so the first solved
//    state the search settles (pops) is push-optimal for the room.
//
// 3. Stitch the rooms together: solve them in board order, the other
//    rooms' boxes standing as walls (on their goals once their room is
//    done), the player walking on from wherever the previous room left
//    it, and replay each room's pushes with the shortest walk to each
//    push's square.
//
// Everything is budgeted by expansions counted, never by the clock, and
// every tie breaks by square and direction order, so two runs of this
// program on the same set write byte-identical answers. If a level does
// not split into rooms, the same A* runs on the whole level at once.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Directions in a fixed order: 0 up, 1 down, 2 left, 3 right (k ^ 1 is the opposite).
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const WALK = ['u', 'd', 'l', 'r'];
const PUSH = ['U', 'D', 'L', 'R'];
const INF = 1e9;

// One work budget for the whole set, counted in A* expansions: generous
// enough that no drawn room comes close, finite so a pathological level
// still answers (with an empty file) inside the wall budget.
const budget = { left: 30_000_000 };

/** A level in XSB text: width, height, wall mask, goals, boxes (sorted), player. */
function parseLevel(text) {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const size = w * h;
  const wall = new Uint8Array(size).fill(1);
  const goals = [];
  const boxes = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  // The outermost frame is wall whatever the text shows.
  for (let x = 0; x < w; x++) { wall[x] = 1; wall[(h - 1) * w + x] = 1; }
  for (let y = 0; y < h; y++) { wall[y * w] = 1; wall[y * w + w - 1] = 1; }
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, size, wall, goals, boxes, player };
}

/** Geometric neighbours: nbr[i * 4 + k] is the square one step in direction k from i, or -1 off the board. */
function neighbourTable(w, h) {
  const nbr = new Int32Array(w * h * 4).fill(-1);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      for (let k = 0; k < 4; k++) {
        const nx = x + DX[k];
        const ny = y + DY[k];
        if (nx >= 0 && nx < w && ny >= 0 && ny < h) nbr[i * 4 + k] = ny * w + nx;
      }
    }
  }
  return nbr;
}

/**
 * The squares from which a lone box can be pushed onto some goal, found by
 * pulling boxes backwards from the goals: a box now on x could have been
 * pushed there from the square behind it only if both that square and the
 * square the player stood on are open floor.
 */
function liveSquares(size, wall, nbr, goals) {
  const live = new Uint8Array(size);
  const queue = [];
  for (const g of goals) { live[g] = 1; queue.push(g); }
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    for (let k = 0; k < 4; k++) {
      const y = nbr[x * 4 + (k ^ 1)];
      if (y < 0 || wall[y] || live[y]) continue;
      const stand = nbr[y * 4 + (k ^ 1)];
      if (stand < 0 || wall[stand]) continue;
      live[y] = 1;
      queue.push(y);
    }
  }
  return live;
}

/**
 * For each goal, the fewest pushes that move a lone box from any square
 * onto it (other boxes ignored): the same backward pull search as
 * liveSquares, one goal at a time, keeping the distance. -1: unreachable.
 */
function pushDistances(size, wall, nbr, goals) {
  return goals.map((g) => {
    const dist = new Int32Array(size).fill(-1);
    dist[g] = 0;
    const queue = [g];
    for (let qi = 0; qi < queue.length; qi++) {
      const x = queue[qi];
      for (let k = 0; k < 4; k++) {
        const y = nbr[x * 4 + (k ^ 1)];
        if (y < 0 || wall[y] || dist[y] >= 0) continue;
        const stand = nbr[y * 4 + (k ^ 1)];
        if (stand < 0 || wall[stand]) continue;
        dist[y] = dist[x] + 1;
        queue.push(y);
      }
    }
    return dist;
  });
}

/**
 * A lower bound on the pushes a position still needs: the cheapest
 * assignment of a distinct goal to every box under the lone-box push
 * distances, by DP over goal subsets (few goals), or the sum of each
 * box's own nearest-goal distance (many goals; still a bound, since the
 * pushes spent on one box cannot serve another). INF when some box can
 * reach no goal at all.
 */
function lowerBound(boxes, tables) {
  const n = boxes.length;
  const m = tables.length;
  if (m < n) return INF;
  if (m > 10) {
    let sum = 0;
    for (const b of boxes) {
      let best = INF;
      for (let j = 0; j < m; j++) {
        const d = tables[j][b];
        if (d >= 0 && d < best) best = d;
      }
      if (best >= INF) return INF;
      sum += best;
    }
    return sum;
  }
  const full = 1 << m;
  const dp = new Float64Array(full).fill(INF);
  dp[0] = 0;
  let answer = INF;
  for (let mask = 0; mask < full; mask++) {
    const base = dp[mask];
    if (base >= INF) continue;
    let i = 0;
    for (let t = mask; t; t &= t - 1) i++;
    if (i === n) { if (base < answer) answer = base; continue; }
    if (i > n) continue;
    const b = boxes[i];
    for (let j = 0; j < m; j++) {
      if (mask & (1 << j)) continue;
      const d = tables[j][b];
      if (d < 0) continue;
      const nm = mask | (1 << j);
      if (base + d < dp[nm]) dp[nm] = base + d;
    }
  }
  return answer;
}

/** Mark the player's reachable region (boxes as obstacles) with `stamp`; return its lowest square. */
function floodRegion(wall, nbr, occupied, from, seen, stamp, stack) {
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    for (let k = 0; k < 4; k++) {
      const q = nbr[p * 4 + k];
      if (q >= 0 && !wall[q] && !occupied[q] && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/** Whether the box just pushed to `t` froze in a 2 x 2 block of walls and boxes with a box off its goal. */
function frozenBlock(w, wall, occupied, goalSet, t) {
  const x = t % w;
  const y = (t / w) | 0;
  for (const [ox, oy] of [[0, 0], [-1, 0], [0, -1], [-1, -1]]) {
    const ax = x + ox;
    const ay = y + oy;
    if (ax < 0 || ay < 0 || ax + 1 >= w) continue;
    const a = ay * w + ax;
    const cells = [a, a + 1, a + w, a + w + 1];
    let blocked = true;
    let offGoal = false;
    for (const c of cells) {
      if (!wall[c] && !occupied[c]) { blocked = false; break; }
      if (occupied[c] && !goalSet[c]) offGoal = true;
    }
    if (blocked && offGoal) return true;
  }
  return false;
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower LURD letters (first direction wins ties). */
function walkPath(wall, nbr, occupied, from, to) {
  if (from === to) return '';
  const size = wall.length;
  const prev = new Int32Array(size).fill(-1);
  const how = new Int32Array(size).fill(-1);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (let k = 0; k < 4; k++) {
      const q = nbr[p * 4 + k];
      if (q < 0 || wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = k;
      queue.push(q);
    }
  }
  if (prev[to] === -1) return null;
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = WALK[how[p]] + s;
  return s;
}

/** Replay a solved node chain into a full move string: walk to each push's square, then push. */
function reconstruct(board, nBoxes, nParent, nDir, nFrom, startBoxes, startPlayer, goalNode) {
  const { wall, nbr } = board;
  const chain = [];
  for (let s = goalNode; s > 0; s = nParent[s]) chain.push(s);
  chain.reverse();
  const occupied = new Uint8Array(board.size);
  let player = startPlayer;
  let boxes = startBoxes.slice();
  let moves = '';
  for (const s of chain) {
    const k = nDir[s];
    const from = nFrom[s];
    const stand = nbr[from * 4 + (k ^ 1)];
    for (const b of boxes) occupied[b] = 1;
    const walk = walkPath(wall, nbr, occupied, player, stand);
    for (const b of boxes) occupied[b] = 0;
    if (walk === null) return null;
    moves += walk + PUSH[k];
    player = from;
    boxes = nBoxes[s].slice();
  }
  return { pushes: chain.length, moves };
}

/**
 * The push-optimal solution of one (sub-)board: A* over pushes from the
 * start boxes and player square. Returns { pushes, moves } with the walks
 * filled in, or null when the budget runs out or no solution exists.
 */
function solveOptimal(board, startBoxes, startPlayer) {
  const { size, wall, goals, nbr } = board;
  const goalSet = new Uint8Array(size);
  for (const g of goals) goalSet[g] = 1;
  const live = liveSquares(size, wall, nbr, goals);
  const tables = pushDistances(size, wall, nbr, goals);
  if (startBoxes.some((b) => !live[b])) return null;
  const startBound = lowerBound(startBoxes, tables);
  if (startBound >= INF) return null;
  const solved = (boxes) => boxes.every((b) => goalSet[b]);

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const seenNext = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  let stampNext = 0;

  // Node store: boxes, the player's actual square, pushes so far, parent,
  // the push that made the node (direction and the box's old square), key.
  const nBoxes = [startBoxes.slice()];
  const nPlayer = [startPlayer];
  const nG = [0];
  const nParent = [-1];
  const nDir = [-1];
  const nFrom = [-1];
  const nKey = [''];
  const best = new Map(); // state key -> node index holding its best g
  const keyOf = (boxes, regionLow) => String.fromCharCode(regionLow, ...boxes);

  for (const b of startBoxes) occupied[b] = 1;
  const startLow = floodRegion(wall, nbr, occupied, startPlayer, seen, ++stamp, stack);
  for (const b of startBoxes) occupied[b] = 0;
  nKey[0] = keyOf(startBoxes, startLow);
  best.set(nKey[0], 0);
  if (solved(startBoxes)) {
    return reconstruct(board, nBoxes, nParent, nDir, nFrom, startBoxes, startPlayer, 0);
  }

  // Binary heap on (f = g + bound, then deeper g first, then insertion order).
  const heap = [[startBound, 0, 0, 0]];
  let seq = 1;
  const before = (a, b) => a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e) => {
    heap.push(e);
    let i = heap.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!before(heap[i], heap[p])) break;
      [heap[i], heap[p]] = [heap[p], heap[i]];
      i = p;
    }
  };
  const heapPop = () => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && before(heap[c + 1], heap[c])) c++;
        if (!before(heap[c], heap[i])) break;
        [heap[i], heap[c]] = [heap[c], heap[i]];
        i = c;
      }
    }
    return top;
  };

  while (heap.length > 0) {
    if (budget.left <= 0) return null;
    const [, g, , idx] = heapPop();
    if (best.get(nKey[idx]) !== idx) continue; // superseded by a cheaper route to this state
    if (solved(nBoxes[idx])) {
      return reconstruct(board, nBoxes, nParent, nDir, nFrom, startBoxes, startPlayer, idx);
    }
    budget.left--;
    const boxes = nBoxes[idx];
    for (const b of boxes) occupied[b] = 1;
    const regionStamp = ++stamp;
    floodRegion(wall, nbr, occupied, nPlayer[idx], seen, regionStamp, stack);
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi];
      for (let k = 0; k < 4; k++) {
        const stand = nbr[b * 4 + (k ^ 1)];
        if (stand < 0 || seen[stand] !== regionStamp) continue;
        const t = nbr[b * 4 + k];
        if (t < 0 || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(board.w, wall, occupied, goalSet, t);
        let next = null;
        let low = -1;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((a, c) => a - c);
          low = floodRegion(wall, nbr, occupied, b, seenNext, ++stampNext, stack);
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen) continue;
        const key = keyOf(next, low);
        const known = best.get(key);
        if (known !== undefined && nG[known] <= g + 1) continue;
        const bound = lowerBound(next, tables);
        if (bound >= INF) continue;
        const ni = nBoxes.length;
        nBoxes.push(next);
        nPlayer.push(b);
        nG.push(g + 1);
        nParent.push(idx);
        nDir.push(k);
        nFrom.push(b);
        nKey.push(key);
        best.set(key, ni);
        heapPush([g + 1 + bound, g + 1, seq++, ni]);
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  return null;
}

/**
 * Split a level into its rooms: cut the dead squares of the player's row
 * (the hallway), flood-fill what remains, and keep the components that
 * hold boxes. Null when the level is not a clean set of rooms (a box in
 * the hallway, a component whose boxes and goals disagree, a goal lost
 * to the cut).
 */
function splitRooms(level, nbr) {
  const { w, size, wall, goals, boxes, player } = level;
  const live = liveSquares(size, wall, nbr, goals);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(size);
  for (let x = 0; x < w; x++) {
    const i = row * w + x;
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const comp = new Int32Array(size).fill(-1);
  const rooms = [];
  for (let s = 0; s < size; s++) {
    if (wall[s] || cut[s] || comp[s] >= 0) continue;
    const id = rooms.length;
    const squares = [s];
    comp[s] = id;
    for (let i = 0; i < squares.length; i++) {
      const p = squares[i];
      for (let k = 0; k < 4; k++) {
        const q = nbr[p * 4 + k];
        if (q >= 0 && !wall[q] && !cut[q] && comp[q] < 0) {
          comp[q] = id;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const b of boxes) {
    if (comp[b] < 0) return null;
    rooms[comp[b]].boxes.push(b);
  }
  for (const g of goals) {
    if (comp[g] < 0) return null;
    rooms[comp[g]].goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  const rs = rooms.filter((r) => r.boxes.length > 0);
  for (const r of rs) {
    r.boxes.sort((a, b) => a - b);
    r.goals.sort((a, b) => a - b);
  }
  return rs;
}

/** The square the player stands on after a move string, from `from` on a board `w` wide. */
function endSquare(w, from, moves) {
  let p = from;
  for (const ch of moves.toLowerCase()) {
    if (ch === 'u') p -= w;
    else if (ch === 'd') p += w;
    else if (ch === 'l') p -= 1;
    else if (ch === 'r') p += 1;
  }
  return p;
}

/** Solve one level's text into its move string ('' when no solution is found). */
function solveLevel(text) {
  const level = parseLevel(text);
  const nbr = neighbourTable(level.w, level.h);
  const rooms = splitRooms(level, nbr);
  if (rooms !== null && rooms.length > 0) {
    let player = level.player;
    let moves = '';
    const doneFlags = rooms.map(() => false);
    let allDone = true;
    for (let ri = 0; ri < rooms.length; ri++) {
      const room = rooms[ri];
      const wall = level.wall.slice();
      rooms.forEach((other, oi) => {
        if (oi === ri) return;
        for (const s of doneFlags[oi] ? other.goals : other.boxes) wall[s] = 1;
      });
      const board = { w: level.w, h: level.h, size: level.size, wall, goals: room.goals, nbr };
      const res = solveOptimal(board, room.boxes.slice(), player);
      if (res === null) { allDone = false; break; }
      moves += res.moves;
      player = endSquare(level.w, player, res.moves);
      doneFlags[ri] = true;
    }
    if (allDone) return moves;
  }
  // Fallback: the whole level in one search.
  const board = { w: level.w, h: level.h, size: level.size, wall: level.wall, goals: level.goals, nbr };
  const res = solveOptimal(board, level.boxes.slice(), level.player);
  return res === null ? '' : res.moves;
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  const moves = solveLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
