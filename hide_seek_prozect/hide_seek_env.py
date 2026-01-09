"""
hide_seek_env.py

2D 탑다운(그리드/타일) 기반 멀티에이전트 숨바꼭질 환경 (PettingZoo ParallelEnv)

요구사항 반영:
- 맵은 문자열로 입력 (네가 준 ascii 맵 형태)
- 벽 '#'
- 빈칸 '.' 또는 ' ' (공백)
- 문 'd' 또는 'D' (망치로 부수면 제거)
- 상자 'b' (여러 개 가능, 밀기 가능)
- 망치 'H' (줍기/놓기 가능, 들고 있을 때만 문 파괴 가능)
- 술래 시작 'S'
- 도망자 시작 'R' (없으면 자동 배치)

행동(action):
0: 위, 1: 아래, 2: 왼, 3: 오
4: 줍기(pickup)
5: 놓기(drop)
6: 공격(attack) -> 인접 4방향에 있는 문을 타격

관측(observation):
- "local": 에이전트 기준으로 (2*sight+1) x (2*sight+1) 로컬 맵
- "has_hammer": 0/1

주의:
- 이 버전은 "시야로 발견"이 아니라, 기본은 dist<=1 (근접)로 승패 판정.
  숨기/문막기 전략을 강하게 만들고 싶으면, 다음 단계에서 LOS(가시선) 기반 발견으로 바꾸는 걸 추천.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv


# ===== 타일/오브젝트 인코딩 (관측에서 쓰는 정수값) =====
EMPTY = 0
WALL = 1
DOOR = 2
BOX = 3
HAMMER = 4

AGENTS = ["seeker", "runner"]


@dataclass
class DoorState:
    """문 타일의 내구도(HP)를 저장한다."""
    hp: int


class HideSeekEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv 구현체.
    - step(actions): 모든 에이전트의 행동을 "한 턴에 동시에" 처리한다.
    - reset(): 초기화 후 관측을 돌려준다.
    """

    metadata = {"name": "hide_seek_v0"}

    def __init__(
        self,
        map_str: str,
        door_hp: int = 5,
        sight: int = 5,
        max_steps: int = 600,
        step_penalty: float = 0.0,
        resolve_order: str = "seeker_first",
    ):
        """
        Args:
            map_str: ascii 맵 문자열
            door_hp: 문 초기 체력
            sight: 로컬 관측 반경 (sight=5면 11x11)
            max_steps: 스텝 제한 (넘으면 runner 생존승 처리)
            step_penalty: 매 스텝마다 양쪽에 동일한 패널티(시간 압박)
            resolve_order: 동시 처리 시 충돌을 단순화하기 위한 실행 순서
                          - "seeker_first" or "runner_first" or "random"
        """
        super().__init__()

        self.map_str = map_str
        self.map_lines = [list(line.rstrip("\n")) for line in map_str.strip("\n").splitlines()]
        self.H = len(self.map_lines)
        self.W = max(len(r) for r in self.map_lines) if self.H > 0 else 0

        # 줄 길이가 다르면 패딩(공백)으로 맞춤
        for r in self.map_lines:
            if len(r) < self.W:
                r.extend([" "] * (self.W - len(r)))

        self.door_hp = int(door_hp)
        self.sight = int(sight)
        self.max_steps = int(max_steps)
        self.step_penalty = float(step_penalty)
        self.resolve_order = resolve_order

        # 액션 공간: 0~6
        self.action_spaces = {a: spaces.Discrete(7) for a in AGENTS}

        # 관측 공간
        obs_h = 2 * self.sight + 1
        obs_w = 2 * self.sight + 1
        self.observation_spaces = {
            a: spaces.Dict(
                {
                    "local": spaces.Box(low=0, high=4, shape=(obs_h, obs_w), dtype=np.int32),
                    "has_hammer": spaces.Discrete(2),
                }
            )
            for a in AGENTS
        }

        # 내부 상태 초기화
        self.reset()

    # ------------------------------
    # 맵 파싱 / 초기화
    # ------------------------------
    def _parse_map(self) -> None:
        """
        map_lines를 읽어서 게임 상태(벽/문/상자/망치/시작위치)를 구성한다.

        base[y,x]는 "고정 타일"의 기본 상태만 저장:
        - EMPTY / WALL / DOOR

        상자/망치/에이전트는 동적이므로 별도 변수에 저장한다.
        """
        self.base = np.zeros((self.H, self.W), dtype=np.int32)
        self.doors: Dict[Tuple[int, int], DoorState] = {}

        self.boxes: List[Tuple[int, int]] = []
        self.hammer_pos: Optional[Tuple[int, int]] = None

        self.pos: Dict[str, Tuple[int, int]] = {a: None for a in AGENTS}  # type: ignore
        self.has_hammer: Dict[str, int] = {a: 0 for a in AGENTS}

        for y in range(self.H):
            for x in range(self.W):
                c = self.map_lines[y][x]

                if c == "#":
                    self.base[y, x] = WALL

                elif c in ("d", "D"):
                    self.base[y, x] = DOOR
                    self.doors[(y, x)] = DoorState(hp=self.door_hp)

                elif c == "b":
                    # 여러 개 가능
                    self.boxes.append((y, x))

                elif c == "H":
                    self.hammer_pos = (y, x)

                elif c == "S":
                    self.pos["seeker"] = (y, x)

                elif c == "R":
                    self.pos["runner"] = (y, x)

                elif c in (".", " "):
                    # 빈칸
                    pass

                else:
                    # 알 수 없는 문자는 빈칸 처리(원하면 에러로 바꿔도 됨)
                    pass

        # 필수 요소 검증
        if self.pos["seeker"] is None:
            raise ValueError("맵에 seeker 시작점 'S'가 없습니다.")

        if self.hammer_pos is None:
            raise ValueError("맵에 망치 시작점 'H'가 없습니다.")

        # runner 시작점이 없으면 자동 배치 (권장: 맵에 R 직접 추가)
        if self.pos["runner"] is None:
            self.pos["runner"] = self._auto_place_runner()

    def _auto_place_runner(self) -> Tuple[int, int]:
        """
        runner 시작점 'R'이 맵에 없는 경우 호출.
        빈칸(EMPTY) 중 하나를 골라 runner를 배치한다.

        - 단순히 "가장 아래/오른쪽 쪽 빈칸"을 선택.
        - 더 똑똑하게 하려면 seeker에서 가장 먼 칸을 고르는 방식도 가능.
        """
        empties = []
        for y in range(self.H):
            for x in range(self.W):
                if self.base[y, x] != EMPTY:
                    continue
                if (y, x) in self.boxes:
                    continue
                if self.hammer_pos == (y, x):
                    continue
                empties.append((y, x))

        if not empties:
            raise ValueError("runner를 자동 배치할 빈칸이 없습니다. 맵을 확인하세요.")

        return empties[-1]  # 맨 마지막 빈칸(대체로 맵 하단/우측)

    def reset(self, seed=None, options=None):
        """
        에피소드 초기화.
        PettingZoo ParallelEnv 규약: (observations, infos) 반환
        """
        self.agents = AGENTS[:]
        self.step_count = 0

        # 맵 재파싱(오브젝트 초기위치/문HP 초기화)
        self._parse_map()

        obs = self._get_obs()
        infos = {a: {} for a in self.agents}
        return obs, infos

    # ------------------------------
    # 유틸: 충돌/탐색
    # ------------------------------
    def _in_bounds(self, y: int, x: int) -> bool:
        """좌표가 맵 안에 있는지."""
        return 0 <= y < self.H and 0 <= x < self.W

    def _is_agent_at(self, y: int, x: int) -> bool:
        """해당 좌표에 seeker 또는 runner가 있는지."""
        return (y, x) == self.pos["seeker"] or (y, x) == self.pos["runner"]

    def _is_box_at(self, y: int, x: int) -> bool:
        """해당 좌표에 상자가 있는지."""
        return (y, x) in self.boxes

    def _blocked(self, y: int, x: int) -> bool:
        """
        이동 불가능 여부를 반환.
        - 벽/문은 막힘
        - 상자도 막힘(하지만 밀기는 별도 로직에서 처리)
        - 맵 밖도 막힘
        """
        if not self._in_bounds(y, x):
            return True
        if self.base[y, x] == WALL:
            return True
        if self.base[y, x] == DOOR:
            return True
        if self._is_box_at(y, x):
            return True
        return False

    # ------------------------------
    # 행동 처리: 이동/밀기/줍기/공격
    # ------------------------------
    def _try_move(self, agent: str, dy: int, dx: int) -> None:
        """
        agent를 (dy, dx)만큼 이동 시도.
        - 목표 칸이 상자면 밀기 시도
        - 목표 칸이 비어있으면 이동
        - 문/벽이면 이동 불가
        - 다른 에이전트가 있으면 이동 불가 (단순 규칙)
        """
        ay, ax = self.pos[agent]
        ny, nx = ay + dy, ax + dx

        if not self._in_bounds(ny, nx):
            return

        # 목표 칸에 다른 에이전트가 있으면 이동 불가
        if self._is_agent_at(ny, nx):
            return

        # 상자를 밀어야 하는 경우
        if (ny, nx) in self.boxes:
            by, bx = ny + dy, nx + dx  # 상자가 밀려날 칸

            # 상자 목표 칸이 막혀있으면 밀기 불가
            if self._blocked(by, bx) or self._is_agent_at(by, bx):
                return

            # 상자 이동(리스트 내 좌표 갱신)
            idx = self.boxes.index((ny, nx))
            self.boxes[idx] = (by, bx)

            # 상자를 밀면서 agent도 한 칸 전진
            self.pos[agent] = (ny, nx)
            return

        # 일반 이동
        if not self._blocked(ny, nx):
            self.pos[agent] = (ny, nx)

    def _pickup(self, agent: str) -> None:
        """
        망치 줍기:
        - agent가 망치 위치와 같은 칸이면 줍는다.
        - 이미 들고 있으면 무시
        """
        if self.has_hammer[agent] == 1:
            return
        if self.hammer_pos is None:
            return
        if self.pos[agent] == self.hammer_pos:
            self.has_hammer[agent] = 1
            self.hammer_pos = None  # 맵에서 사라짐(인벤토리로)

    def _drop(self, agent: str) -> None:
        """
        망치 놓기:
        - 들고 있을 때만 가능
        - 현재 칸이 빈칸이어야 놓을 수 있음
        """
        if self.has_hammer[agent] == 0:
            return
        y, x = self.pos[agent]

        # 현재 칸이 "기본 타일로 EMPTY"이고 상자도 없고 망치도 없을 때만 놓기
        if self.base[y, x] != EMPTY:
            return
        if self._is_box_at(y, x):
            return
        if self.hammer_pos is not None:
            # 망치는 맵에 1개만 유지한다는 단순 규칙
            return

        self.has_hammer[agent] = 0
        self.hammer_pos = (y, x)

    def _attack(self, agent: str) -> None:
        """
        공격(문 부수기):
        - 망치를 들고 있을 때만 공격 가능
        - 인접 4방향에 문이 있으면 그 문을 1회 타격
        - 문 HP가 0 이하가 되면 문이 제거되어 통과 가능해짐(base에서 DOOR -> EMPTY)
        """
        if self.has_hammer[agent] == 0:
            return

        y, x = self.pos[agent]
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dy, dx in neighbors:
            ty, tx = y + dy, x + dx
            if not self._in_bounds(ty, tx):
                continue
            if self.base[ty, tx] != DOOR:
                continue

            # 문 상태 가져와 HP 감소
            st = self.doors.get((ty, tx))
            if st is None:
                # base에 DOOR인데 dict에 없는 경우(비정상) -> 제거 처리
                self.base[ty, tx] = EMPTY
                return

            st.hp -= 1

            # 파괴되면 문 제거
            if st.hp <= 0:
                self.base[ty, tx] = EMPTY
                self.doors.pop((ty, tx), None)

            # 한 번 공격하면 "가장 먼저 발견한 문 1개만" 때린다.
            break

    # ------------------------------
    # step / 관측 / 종료조건
    # ------------------------------
    def step(self, actions: Dict[str, int]):
        """
        한 틱(step) 진행.
        ParallelEnv는 모든 에이전트의 action을 한 번에 받는다.

        처리 순서:
        - resolve_order에 따라 seeker/runner 행동 순서를 결정
        - 각 에이전트의 행동을 적용
        - 승패 판정 및 보상 계산
        """
        self.step_count += 1

        # 실행 순서 결정(동시 처리 충돌을 단순화하기 위한 장치)
        if self.resolve_order == "runner_first":
            order = ["runner", "seeker"]
        elif self.resolve_order == "random":
            order = AGENTS[:]
            np.random.shuffle(order)
        else:
            order = ["seeker", "runner"]

        for a in order:
            if a not in self.agents:
                continue
            act = int(actions.get(a, 0))

            if act == 0:
                self._try_move(a, -1, 0)
            elif act == 1:
                self._try_move(a, 1, 0)
            elif act == 2:
                self._try_move(a, 0, -1)
            elif act == 3:
                self._try_move(a, 0, 1)
            elif act == 4:
                self._pickup(a)
            elif act == 5:
                self._drop(a)
            elif act == 6:
                self._attack(a)

        # 기본 보상(시간 패널티)
        rewards = {a: 0.0 for a in self.agents}
        if self.step_penalty != 0.0:
            for a in rewards:
                rewards[a] += self.step_penalty

        terminated = {a: False for a in self.agents}
        truncated = {a: False for a in self.agents}

        # 종료조건 1: seeker가 runner를 잡음(기본: 맨해튼 거리 1 이하)
        sy, sx = self.pos["seeker"]
        ry, rx = self.pos["runner"]
        dist = abs(sy - ry) + abs(sx - rx)

        if dist <= 1:
            rewards["seeker"] += 1.0
            rewards["runner"] += -1.0
            for a in self.agents:
                terminated[a] = True

        # 종료조건 2: 시간 초과 -> runner 생존승
        if self.step_count >= self.max_steps and not any(terminated.values()):
            rewards["runner"] += 1.0
            rewards["seeker"] += -1.0
            for a in self.agents:
                truncated[a] = True

        obs = self._get_obs()
        infos = {a: {} for a in self.agents}
        return obs, rewards, terminated, truncated, infos

    def _get_obs(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        각 에이전트에게 로컬 관측을 제공.
        - local: 주변 타일(벽/문/상자/망치)
        - has_hammer: 인벤토리 상태

        NOTE:
        - 현재는 상대 에이전트를 관측에 포함하지 않는다(더 "모르게" 시작).
          원하면 local에 상대를 별도 값(예: 5)로 넣어서 부분관측 숨바꼭질로 확장 가능.
        """
        out = {}
        r = self.sight

        for a in self.agents:
            y, x = self.pos[a]
            local = np.zeros((2 * r + 1, 2 * r + 1), dtype=np.int32)

            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    yy, xx = y + dy, x + dx

                    v = EMPTY
                    if self._in_bounds(yy, xx):
                        v = int(self.base[yy, xx])
                        if (yy, xx) in self.boxes:
                            v = BOX
                        if self.hammer_pos == (yy, xx):
                            v = HAMMER
                    local[dy + r, dx + r] = v

            out[a] = {
                "local": local,
                "has_hammer": int(self.has_hammer[a]),
            }

        return out

    # ------------------------------
    # 디버그/시각화(텍스트)
    # ------------------------------
    def render_ascii(self) -> str:
        """
        현재 상태를 콘솔에서 보기 쉽도록 ASCII로 렌더링한다.

        우선순위(겹치면 위가 이김):
        seeker 'S', runner 'R', hammer 'H', box 'b', door 'd', wall '#', empty ' '
        """
        canvas = np.full((self.H, self.W), " ", dtype="<U1")

        # base 먼저
        for y in range(self.H):
            for x in range(self.W):
                if self.base[y, x] == WALL:
                    canvas[y, x] = "#"
                elif self.base[y, x] == DOOR:
                    canvas[y, x] = "d"
                else:
                    canvas[y, x] = " "

        # boxes
        for (y, x) in self.boxes:
            canvas[y, x] = "b"

        # hammer
        if self.hammer_pos is not None:
            hy, hx = self.hammer_pos
            canvas[hy, hx] = "H"

        # agents
        sy, sx = self.pos["seeker"]
        ry, rx = self.pos["runner"]
        canvas[sy, sx] = "S"
        canvas[ry, rx] = "R"

        return "\n".join("".join(row.tolist()) for row in canvas)
