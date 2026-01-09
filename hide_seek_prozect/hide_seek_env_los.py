# hide_seek_env_los.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Set

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

# ===== 관측/타일 인코딩 =====
EMPTY = 0
WALL = 1
DOOR = 2
BOX = 3
HAMMER = 4
SEEKER = 5
RUNNER = 6

AGENTS = ["seeker", "runner"]


@dataclass
class DoorState:
    hp: int


class HideSeekLOS(ParallelEnv):
    """
    Parallel PettingZoo Env
    - seeker: 술래
    - runner: 도망자
    - 시야(LOS) + 위험반경/탐색 보상 shaping
    - 커리큘럼 stage에 따라 문/상자/망치 활성화
    """

    metadata = {"name": "hide_seek_los_v0"}

    def __init__(
        self,
        map_str: str,
        stage: int = 0,
        door_hp: int = 6,
        sight: int = 8,
        max_steps: int = 500,
        step_penalty: float = -0.001,
        catch_dist: int = 1,
        los_blocks_by_boxes: bool = True,
        resolve_order: str = "random",
        danger_r: int = 6,
        danger_coef: float = 0.01,
        see_reward: float = 0.02,
        see_penalty: float = 0.02,
        explore_reward: float = 0.003,
        catch_reward: float = 5.0,
        caught_penalty: float = 5.0,
        escape_reward: float = 1.0,
        escape_penalty: float = 1.0,
        seeker_freeze_steps: int = 0,
    ):
        super().__init__()
        # ... (중략: 맵 파싱 및 변수 초기화 로직은 동일) ...
        self.map_str = map_str
        self.map_lines = [list(line.rstrip("\n")) for line in map_str.strip("\n").splitlines()]
        self.H = len(self.map_lines)
        self.W = max(len(r) for r in self.map_lines) if self.H else 0
        for r in self.map_lines:
            if len(r) < self.W:
                r.extend([" "] * (self.W - len(r)))

        self.stage = int(stage)
        self.door_hp = int(door_hp)
        self.sight = int(sight)
        self.max_steps = int(max_steps)
        self.step_penalty = float(step_penalty)
        self.catch_dist = int(catch_dist)
        self.los_blocks_by_boxes = bool(los_blocks_by_boxes)
        self.resolve_order = resolve_order
        self.danger_r = int(danger_r)
        self.danger_coef = float(danger_coef)
        self.see_reward = float(see_reward)
        self.see_penalty = float(see_penalty)
        self.explore_reward = float(explore_reward)
        self.catch_reward = float(catch_reward)
        self.caught_penalty = float(caught_penalty)
        self.escape_reward = float(escape_reward)
        self.escape_penalty = float(escape_penalty)
        self.seeker_freeze_steps = int(seeker_freeze_steps)
        self.freeze_left = 0

        # ----- spaces -----
        self.action_spaces = {a: spaces.Discrete(7) for a in AGENTS}

        # [핵심 수정] CNN 입력을 위한 Observation Space 변경
        # Shape: (채널 수, 높이, 너비) -> (5, 17, 17)
        # 채널 순서: 0:벽/외부, 1:문, 2:상자, 3:망치, 4:상대방
        obs_hw = 2 * self.sight + 1
        self.observation_spaces = {
            a: spaces.Dict({
                "local": spaces.Box(low=0, high=1, shape=(obs_hw, obs_hw, 5), dtype=np.float32),
                "has_hammer": spaces.Discrete(2),
                "opponent_visible": spaces.Discrete(2), # 보조 정보 추가
            })
            for a in AGENTS
        }

        self.agents: List[str] = AGENTS[:]
        self.step_count = 0
        self.base: np.ndarray = np.zeros((self.H, self.W), dtype=np.int32)
        self.doors: Dict[Tuple[int, int], DoorState] = {}
        self.boxes: List[Tuple[int, int]] = []
        self.hammer_pos: Optional[Tuple[int, int]] = None
        self.pos: Dict[str, Tuple[int, int]] = {a: (0,0) for a in AGENTS}
        self.has_hammer: Dict[str, int] = {a: 0 for a in AGENTS}
        self.seeker_visited: Set[Tuple[int, int]] = set()

    # -------------------------------
    # Map parse / init
    # -------------------------------
    def _parse_map(self):
        self.base = np.zeros((self.H, self.W), dtype=np.int32)
        self.doors = {}
        self.boxes = []
        self.hammer_pos = None
        self.pos = {a: None for a in AGENTS} # type: ignore
        self.has_hammer = {a: 0 for a in AGENTS}

        for y in range(self.H):
            for x in range(self.W):
                c = self.map_lines[y][x]
                if c == "#": self.base[y, x] = WALL
                elif c in ("d", "D"):
                    if self.stage >= 1:
                        self.base[y, x] = DOOR
                        self.doors[(y, x)] = DoorState(hp=self.door_hp)
                elif c == "b":
                    if self.stage >= 2: self.boxes.append((y, x))
                elif c == "H":
                    if self.stage >= 3: self.hammer_pos = (y, x)
                elif c == "S": self.pos["seeker"] = (y, x)
                elif c == "R": self.pos["runner"] = (y, x)

        if self.pos["seeker"] is None: raise ValueError("S 없음")
        if self.pos["runner"] is None: self.pos["runner"] = self._auto_place_runner()
        self.pos["runner"] = self._sanitize_spawn(self.pos["runner"])

    def _auto_place_runner(self):
        sy, sx = self.pos["seeker"]
        empties = []
        for y in range(self.H):
            for x in range(self.W):
                if self.base[y, x] == EMPTY and (y,x) not in self.boxes and self.hammer_pos != (y,x) and (y,x) != (sy,sx):
                    empties.append((y, x))
        return empties[-1] if empties else (0,0)

    def _sanitize_spawn(self, pos):
        y, x = pos
        y = max(0, min(self.H - 1, y)); x = max(0, min(self.W - 1, x))
        if self._is_spawnable_empty(y, x): return (y, x)
        nearest = self._find_nearest_spawnable((y, x))
        return nearest if nearest else self._auto_place_runner()

    def _is_spawnable_empty(self, y, x):
        return self._in_bounds(y, x) and self.base[y, x] == EMPTY and (y, x) not in self.boxes and self.hammer_pos != (y, x)

    def _find_nearest_spawnable(self, start):
        from collections import deque
        q = deque([start]); seen = {start}
        while q:
            y, x = q.popleft()
            if self._is_spawnable_empty(y, x): return (y, x)
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if self._in_bounds(ny, nx) and (ny, nx) not in seen:
                    seen.add((ny, nx)); q.append((ny, nx))
        return None

    def _in_bounds(self, y, x): return 0 <= y < self.H and 0 <= x < self.W
    def _blocked(self, y, x):
        return not self._in_bounds(y, x) or self.base[y, x] in (WALL, DOOR) or (y, x) in self.boxes

    def _los_clear(self, y0, x0, y1, x1):
        dy, dx = abs(y1 - y0), abs(x1 - x0)
        sy, sx = (1 if y0 < y1 else -1), (1 if x0 < x1 else -1)
        err = dx - dy
        y, x = y0, x0
        while True:
            if (y, x) != (y0, x0) and (y, x) != (y1, x1):
                if self.base[y, x] in (WALL, DOOR): return False
                if self.los_blocks_by_boxes and (y, x) in self.boxes: return False
            if (y, x) == (y1, x1): break
            e2 = 2 * err
            if e2 > -dy: err -= dy; x += sx
            if e2 < dx: err += dx; y += sy
        return True

    def _visible(self, a, b):
        ay, ax = self.pos[a]; by, bx = self.pos[b]
        if abs(ay - by) + abs(ax - bx) > self.sight: return False
        return self._los_clear(ay, ax, by, bx)

    def reset(self, seed=None, options=None):
        self.agents = AGENTS[:]
        self.step_count = 0
        self._parse_map()
        self.seeker_visited = {self.pos["seeker"]}
        self.freeze_left = self.seeker_freeze_steps
        return self._get_obs(), {a: {} for a in self.agents}

    def _move(self, agent, dy, dx):
        y, x = self.pos[agent]
        ny, nx = y + dy, x + dx
        if not self._blocked(ny, nx) and (ny, nx) not in self.pos.values():
            self.pos[agent] = (ny, nx)

    def _attack(self, agent):
        if self.stage < 3 or self.has_hammer[agent] == 0: return
        y, x = self.pos[agent]
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if self._in_bounds(ny, nx):
                if self.base[ny, nx] == DOOR:
                    ds = self.doors.get((ny, nx))
                    if ds:
                        ds.hp -= 1
                        if ds.hp <= 0: self.base[ny, nx] = EMPTY; del self.doors[(ny, nx)]
                        return
                if (ny, nx) in self.boxes:
                    self.boxes.remove((ny, nx)); return

    def _pickup(self, agent):
        if self.stage >= 3 and self.pos[agent] == self.hammer_pos and not self.has_hammer[agent]:
            self.has_hammer[agent] = 1; self.hammer_pos = None

    def _drop(self, agent):
        if self.stage >= 3 and self.has_hammer[agent] and self.hammer_pos is None:
            self.hammer_pos = self.pos[agent]; self.has_hammer[agent] = 0

    def step(self, actions):
        if not self.agents: return {}, {}, {}, {}, {}
        self.step_count += 1
        seeker_prev = self.pos["seeker"]
        order = self.agents[:]
        if self.resolve_order == "random": np.random.shuffle(order)
        if self.freeze_left > 0: self.freeze_left -= 1

        for a in order:
            act = int(actions.get(a, 0))
            if a == "seeker" and self.freeze_left > 0 and act in (0, 1, 2, 3): continue
            if act == 0: self._move(a, -1, 0)
            elif act == 1: self._move(a, 1, 0)
            elif act == 2: self._move(a, 0, -1)
            elif act == 3: self._move(a, 0, 1)
            elif act == 4: self._pickup(a)
            elif act == 5: self._drop(a)
            elif act == 6: self._attack(a)

        sy, sx = self.pos["seeker"]; ry, rx = self.pos["runner"]
        dist = abs(sy - ry) + abs(sx - rx)
        rewards = {a: float(self.step_penalty) for a in self.agents}
        terminated = {a: False for a in self.agents}
        truncated = {a: False for a in self.agents}

        if dist <= self.danger_r:
            p = self.danger_coef * (self.danger_r - dist + 1)
            rewards["seeker"] += p; rewards["runner"] -= p

        seeker_sees = self._visible("seeker", "runner")
        if seeker_sees:
            rewards["seeker"] += self.see_reward; rewards["runner"] -= self.see_penalty

        if self.pos["seeker"] != seeker_prev and self.pos["seeker"] not in self.seeker_visited:
            self.seeker_visited.add(self.pos["seeker"]); rewards["seeker"] += self.explore_reward

        if seeker_sees and dist <= self.catch_dist:
            rewards["seeker"] += self.catch_reward; rewards["runner"] -= self.caught_penalty
            terminated = {a: True for a in self.agents}
        elif self.step_count >= self.max_steps:
            rewards["runner"] += self.escape_reward; rewards["seeker"] -= self.escape_penalty
            truncated = {a: True for a in self.agents}

        return self._get_obs(), rewards, terminated, truncated, {a: {"dist": dist} for a in self.agents}

    # [핵심 수정] CNN 기반 다채널 Observation 생성 함수
    def _get_obs(self) -> Dict[str, Dict]:
        obs: Dict[str, Dict] = {}
        r = self.sight
        hw = 2 * r + 1

        seeker_sees = self._visible("seeker", "runner")
        runner_sees = self._visible("runner", "seeker")

        for a in self.agents:
            y, x = self.pos[a]

            #  (IF HWC: (Height, Width, Channels)
            local = np.zeros((hw, hw, 5), dtype=np.float32)

            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    yy, xx = y + dy, x + dx
                    row, col = dy + r, dx + r

                    # 채널 0: 벽 및 맵 외부
                    if (not self._in_bounds(yy, xx)) or (self.base[yy, xx] == WALL):
                        local[row, col, 0] = 1.0

                        # 맵 밖이면 더 볼 필요 없음(다른 채널도 0 유지)
                        if not self._in_bounds(yy, xx):
                            continue

                    # 채널 1: 문
                    if self._in_bounds(yy, xx) and (self.base[yy, xx] == DOOR):
                        local[row, col, 1] = 1.0

                    # 채널 2: 상자
                    if self._in_bounds(yy, xx) and ((yy, xx) in self.boxes):
                        local[row, col, 2] = 1.0

                    # 채널 3: 망치
                    if self._in_bounds(yy, xx) and (self.hammer_pos == (yy, xx)):
                        local[row, col, 3] = 1.0

            # 채널 4: 상대방 위치 (시야에 들어온 경우만)
            sees_opponent = seeker_sees if a == "seeker" else runner_sees
            if sees_opponent:
                opp = "runner" if a == "seeker" else "seeker"
                oy, ox = self.pos[opp]
                ody, odx = oy - y, ox - x
                if abs(ody) <= r and abs(odx) <= r:
                    local[ody + r, odx + r, 4] = 1.0

            obs[a] = {
                "local": local,
                "has_hammer": int(self.has_hammer[a]),
                "opponent_visible": int(sees_opponent),
            }

        return obs

    

    def render_ascii(self) -> str:
        """현재 상태를 ASCII 문자열로 반환(프린트 X)."""
        grid = np.array(self.base, dtype=np.int32)
        out = [[" " for _ in range(self.W)] for _ in range(self.H)]

        for y in range(self.H):
            for x in range(self.W):
                if grid[y, x] == WALL:
                    out[y][x] = "#"
                elif grid[y, x] == DOOR:
                    out[y][x] = "d"
                else:
                    # 기본은 빈칸 '.', 원하면 ' '로 바꿔도 됨
                    out[y][x] = " "

        for (y, x) in self.boxes:
            out[y][x] = "b"

        if self.hammer_pos is not None:
            hy, hx = self.hammer_pos
            out[hy][hx] = "H"

        sy, sx = self.pos["seeker"]
        ry, rx = self.pos["runner"]
        out[sy][sx] = "S"
        out[ry][rx] = "R"

        return "\n".join("".join(row) for row in out)
        
    def render(self):
        print(self.render_ascii())


    def close(self):
        pass
