

import sys
import os
import math
import random
from collections import deque
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGraphicsScene, QGraphicsView, QGraphicsItem,
    QFrame, QFileDialog, QTextEdit, QGridLayout, QGraphicsLineItem,
    QGraphicsPixmapItem, QGraphicsPathItem
)
from PyQt6.QtGui import (
    QImage, QPixmap, QColor, QPen, QBrush, QPainter, QFont, QPainterPath
)
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF

VERSION = "3.8.5"
print(f"[TD3 MAZE] Running version {VERSION} from: {__file__}")

# -----------------
# Config
# -----------------
DEFAULT_MAP = "city_sprite.png"
DEFAULT_CAR_SPRITE = "car_topdown.png"   # front faces RIGHT at 0°

# -----------------
# Colors
# -----------------
C_BG_DARK = QColor("#2E3440")
C_PANEL = QColor("#3B4252")
C_INFO_BG = QColor("#4C566A")
C_ACCENT = QColor("#88C0D0")
C_TEXT = QColor("#ECEFF4")
C_SUCCESS = QColor("#A3BE8C")
C_FAILURE = QColor("#BF616A")
C_SENSOR_GREEN = QColor(90, 200, 120, 230)
C_SENSOR_RED = QColor(220, 80, 80, 230)
C_TRAIL = QColor(255, 70, 70, 210)

# -----------------
# Car + sensors
# -----------------
CAR_DRAW_W = 34
CAR_DRAW_H = 18

SENSOR_DIST = 60.0
SENSOR_ANGLES = [-70, -45, -25, 0, 25, 45, 70]
RAY_STEP = 2.0

BASE_SPEED = 1.8
THROTTLE_GAIN = 2.0
MAX_SPEED = 5.5
MAX_REVERSE = 2.7

MAX_TURN_DEG = 12.0

# -----------------
# Target visuals (bigger)
# -----------------
TARGET_RADIUS_ACTIVE = 16
TARGET_RADIUS_INACTIVE = 12
GOAL_RADIUS = 24

# -----------------
# Road detection
# -----------------
ROAD_BRIGHT_THRESH = 0.60
ROAD_DILATE_RADIUS = 4

# Action smoothing
ACTION_EMA_ALPHA = 0.30
MAX_STEER_DELTA = 0.22

# -----------------
# Straight-line stabilizer (junction-aware)
# -----------------
STRAIGHT_FRONT_MIN = 0.65
STRAIGHT_BALANCE_MAX = 0.10
STRAIGHT_STEER_DAMP = 0.25
STEER_DEADBAND = 0.06
STRAIGHT_TARGET_ANGLE_MAX = 0.10

# -----------------
# Stuck / escape
# -----------------
NO_PROGRESS_STEPS = 140
NO_PROGRESS_EPS = 6.0
RESPAWN_ANGLE_JITTER = 25.0

ESCAPE_STEPS = 28
ESCAPE_STEER_GAIN = 1.15
ESCAPE_FORWARD_THROTTLE = 0.85
ESCAPE_REVERSE_THROTTLE = -0.95
ESCAPE_FRONT_TOO_CLOSE = 0.18

# -----------------
# Exploration bonus (anti-loop) — TRIPLED
# -----------------
VISIT_CELL = 22
NOVELTY_BONUS = 1.05     # 3x of 0.35
NOVELTY_DECAY = 0.5
NOVELTY_MAX = 1.05       # 3x cap

# -----------------
# TD3
# -----------------
SEED = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATE_DIM = 9
ACTION_DIM = 2
REPLAY_SIZE = 120000
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.005
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
POLICY_NOISE = 0.20
NOISE_CLIP = 0.50
POLICY_DELAY = 2
EXPL_NOISE_START = 0.30
EXPL_NOISE_END = 0.05
EXPL_NOISE_DECAY = 0.9993
START_TRAIN_AFTER = 1200

TARGET_COLORS = [
    QColor(0, 255, 255), QColor(255, 100, 255), QColor(0, 255, 100), QColor(255, 150, 0),
    QColor(100, 150, 255), QColor(255, 50, 150), QColor(150, 255, 50), QColor(255, 255, 0)
]


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def angle_diff(a: float, b: float) -> float:
    d = (a - b) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


# =====================
# TD3 Networks
# =====================
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh()
        )

    def forward(self, s):
        return self.net(s)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=1)
        return self.q1(sa), self.q2(sa)

    def q1_only(self, s, a):
        sa = torch.cat([s, a], dim=1)
        return self.q1(sa)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buf)

    def add(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.tensor(np.array(s), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(a), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(r), dtype=torch.float32, device=DEVICE).unsqueeze(1),
            torch.tensor(np.array(ns), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(d), dtype=torch.float32, device=DEVICE).unsqueeze(1),
        )


class TD3Agent:
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM):
        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.actor_t = Actor(state_dim, action_dim).to(DEVICE)
        self.actor_t.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_t = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_t.load_state_dict(self.critic.state_dict())

        self.opt_actor = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.replay = ReplayBuffer(REPLAY_SIZE)
        self.total_it = 0
        self.expl_noise = EXPL_NOISE_START

    @torch.no_grad()
    def act(self, state_np, explore=True):
        s = torch.tensor(state_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        a = self.actor(s).squeeze(0).cpu().numpy()
        if explore:
            a = a + np.random.normal(0, self.expl_noise, size=a.shape)
        return np.clip(a, -1.0, 1.0)

    def train_step(self):
        if len(self.replay) < max(BATCH_SIZE, START_TRAIN_AFTER):
            return None
        self.total_it += 1
        s, a, r, ns, d = self.replay.sample(BATCH_SIZE)

        with torch.no_grad():
            noise = (torch.randn_like(a) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
            next_a = (self.actor_t(ns) + noise).clamp(-1.0, 1.0)
            tq1, tq2 = self.critic_t(ns, next_a)
            tq = torch.min(tq1, tq2)
            y = r + GAMMA * (1.0 - d) * tq

        cq1, cq2 = self.critic(s, a)
        critic_loss = nn.MSELoss()(cq1, y) + nn.MSELoss()(cq2, y)

        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        actor_loss = None
        if self.total_it % POLICY_DELAY == 0:
            actor_loss = -self.critic.q1_only(s, self.actor(s)).mean()
            self.opt_actor.zero_grad()
            actor_loss.backward()
            self.opt_actor.step()

            with torch.no_grad():
                for p, pt in zip(self.actor.parameters(), self.actor_t.parameters()):
                    pt.data.mul_(1.0 - TAU).add_(TAU * p.data)
                for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()):
                    pt.data.mul_(1.0 - TAU).add_(TAU * p.data)

        self.expl_noise = max(EXPL_NOISE_END, self.expl_noise * EXPL_NOISE_DECAY)
        return float(critic_loss.item()), (float(actor_loss.item()) if actor_loss is not None else None)


# =====================
# Environment / Brain
# =====================
class CarBrain:
    def __init__(self, map_image: QImage):
        self.map = map_image
        self.w, self.h = map_image.width(), map_image.height()
        self.agent = TD3Agent()

        self.start_pos = QPointF(100, 100)
        self.car_pos = QPointF(100, 100)
        self.car_angle = 0.0

        # targets
        self.targets: List[QPointF] = []
        self.target_reached: List[bool] = []
        self.current_target_idx = 0
        self.targets_reached = 0
        self.target_pos = QPointF(200, 200)

        self.episode_reward = 0.0

        self.last_safe_pos = QPointF(self.car_pos)
        self.last_safe_angle = 0.0

        self.prev_dist = None
        self.best_dist = None
        self.no_progress_ctr = 0

        self._ema_action = np.zeros((ACTION_DIM,), dtype=np.float32)
        self._prev_action = np.zeros((ACTION_DIM,), dtype=np.float32)

        self.escape_left = 0

        self.count_offroad = 0
        self.count_oob = 0
        self.count_stuck = 0

        self.sensor_rays: List[Tuple[QPointF, QPointF, float]] = []

        self.road_mask = self._build_road_mask()

        # visit grid for novelty
        self.grid_w = max(1, int(math.ceil(self.w / VISIT_CELL)))
        self.grid_h = max(1, int(math.ceil(self.h / VISIT_CELL)))
        self.visit_counts = np.zeros((self.grid_h, self.grid_w), dtype=np.int32)

    def total_respawns(self) -> int:
        return int(self.count_offroad + self.count_oob + self.count_stuck)

    def _build_road_mask(self) -> np.ndarray:
        img = self.map.convertToFormat(QImage.Format.Format_RGB32)
        w, h = img.width(), img.height()
        ptr = img.bits(); ptr.setsize(h * img.bytesPerLine())
        arr = np.frombuffer(ptr, np.uint8).reshape((h, img.bytesPerLine()))
        b = arr[:, 0:w*4:4].astype(np.float32)
        g = arr[:, 1:w*4:4].astype(np.float32)
        r = arr[:, 2:w*4:4].astype(np.float32)
        bright = (r + g + b) / (3.0 * 255.0)
        mask = bright >= ROAD_BRIGHT_THRESH

        rad = int(ROAD_DILATE_RADIUS)
        if rad > 0:
            dil = mask.copy()
            for dy in range(-rad, rad + 1):
                for dx in range(-rad, rad + 1):
                    if dx == 0 and dy == 0:
                        continue
                    ys1 = max(0, dy)
                    ys0 = max(0, -dy)
                    xs1 = max(0, dx)
                    xs0 = max(0, -dx)
                    dil[ys1:h-ys0, xs1:w-xs0] |= mask[ys0:h-ys1, xs0:w-xs1]
            mask = dil
        return mask

    def is_on_road(self, x: float, y: float) -> bool:
        ix, iy = int(x), int(y)
        if 0 <= ix < self.w and 0 <= iy < self.h:
            return bool(self.road_mask[iy, ix])
        return False

    # ---- Novelty bonus ----
    def _visit_cell(self, p: QPointF) -> Tuple[int, int]:
        gx = int(clamp(p.x() // VISIT_CELL, 0, self.grid_w - 1))
        gy = int(clamp(p.y() // VISIT_CELL, 0, self.grid_h - 1))
        return gy, gx

    def novelty_reward(self, p: QPointF) -> float:
        gy, gx = self._visit_cell(p)
        self.visit_counts[gy, gx] += 1
        c = float(self.visit_counts[gy, gx])
        bonus = NOVELTY_BONUS / (c ** NOVELTY_DECAY)
        return float(min(NOVELTY_MAX, bonus))

    # ---- Targets (any-order) ----
    def add_target(self, point: QPointF):
        self.targets.append(QPointF(point.x(), point.y()))
        self.target_reached.append(False)
        if len(self.targets) == 1:
            self.current_target_idx = 0
            self.target_pos = self.targets[0]

    def set_start_pos(self, point: QPointF):
        self.start_pos = QPointF(point.x(), point.y())
        self.car_pos = QPointF(point.x(), point.y())
        self.car_angle = 0.0
        # Reset exploration map when user defines a new start
        self.visit_counts[:] = 0
        if self.is_on_road(self.car_pos.x(), self.car_pos.y()):
            self.last_safe_pos = QPointF(self.car_pos)
            self.last_safe_angle = float(self.car_angle)

    def _nearest_unreached_target(self) -> Optional[int]:
        if not self.targets:
            return None
        best_i = None
        best_d = 1e18
        for i, t in enumerate(self.targets):
            if self.target_reached[i]:
                continue
            d = math.hypot(t.x() - self.car_pos.x(), t.y() - self.car_pos.y())
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    def _update_active_target(self):
        ni = self._nearest_unreached_target()
        if ni is None:
            return
        self.current_target_idx = int(ni)
        self.target_pos = self.targets[self.current_target_idx]

    def _check_any_target_reached(self) -> Tuple[bool, Optional[int]]:
        """Return (hit, idx) for any unreached target within GOAL_RADIUS."""
        for i, t in enumerate(self.targets):
            if self.target_reached[i]:
                continue
            if math.hypot(t.x() - self.car_pos.x(), t.y() - self.car_pos.y()) < GOAL_RADIUS:
                return True, i
        return False, None

    def _all_targets_done(self) -> bool:
        return bool(self.targets) and all(self.target_reached)

    def reset_episode_keep_targets(self, reset_counters: bool = True):
        self.episode_reward = 0.0
        self.car_pos = QPointF(self.start_pos)
        self.car_angle = float(random.uniform(0, 360))
        self.last_safe_pos = QPointF(self.car_pos)
        self.last_safe_angle = float(self.car_angle)
        self.prev_dist = None
        self.best_dist = None
        self.no_progress_ctr = 0
        self._ema_action[:] = 0.0
        self._prev_action[:] = 0.0
        self.escape_left = 0
        if reset_counters:
            self.count_offroad = 0
            self.count_oob = 0
            self.count_stuck = 0
        # IMPORTANT: keep visit_counts across respawns/episodes to break loops
        # Reset target completion for a fresh mission run
        self.target_reached = [False for _ in self.targets]
        self.targets_reached = 0
        self._update_active_target()

    def respawn_at_last_safe(self, align_to_opening: bool = True):
        self.car_pos = QPointF(self.last_safe_pos)
        if align_to_opening:
            st, _ = self.get_state()
            best_i = int(np.argmax(st[:len(SENSOR_ANGLES)]))
            desired = (self.car_angle + SENSOR_ANGLES[best_i]) % 360.0
            self.car_angle = float((desired + random.uniform(-RESPAWN_ANGLE_JITTER, RESPAWN_ANGLE_JITTER)) % 360.0)
        else:
            self.car_angle = float((self.last_safe_angle + random.uniform(-RESPAWN_ANGLE_JITTER, RESPAWN_ANGLE_JITTER)) % 360.0)

    # ---- Sensors ----
    def _ray_distance_to_edge(self, abs_angle_deg: float) -> Tuple[float, QPointF, QPointF]:
        rad = math.radians(abs_angle_deg)
        cx, cy = float(self.car_pos.x()), float(self.car_pos.y())
        dx, dy = math.cos(rad), math.sin(rad)
        full_end = QPointF(cx + dx * SENSOR_DIST, cy + dy * SENSOR_DIST)
        on0 = self.is_on_road(cx, cy)
        last_on = 0.0
        seen_road = on0
        d = 0.0
        while d <= SENSOR_DIST:
            px = cx + dx * d
            py = cy + dy * d
            on = self.is_on_road(px, py)
            if on:
                seen_road = True
                last_on = d
            else:
                if seen_road:
                    break
            d += RAY_STEP
        green_end = QPointF(cx + dx * last_on, cy + dy * last_on)
        return last_on, green_end, full_end

    def get_state(self) -> Tuple[np.ndarray, float]:
        sensor_vals = []
        self.sensor_rays = []
        for a in SENSOR_ANGLES:
            abs_ang = self.car_angle + a
            edge_dist, green_end, full_end = self._ray_distance_to_edge(abs_ang)
            edge_norm = float(edge_dist / SENSOR_DIST)
            sensor_vals.append(edge_norm)
            self.sensor_rays.append((green_end, full_end, edge_norm))

        dx = self.target_pos.x() - self.car_pos.x()
        dy = self.target_pos.y() - self.car_pos.y()
        dist = math.hypot(dx, dy)

        angle_to_target = math.degrees(math.atan2(dy, dx))
        diff = (angle_to_target - self.car_angle) % 360
        if diff > 180:
            diff -= 360

        norm_dist = min(dist / 900.0, 1.0)
        norm_angle = diff / 180.0

        state = sensor_vals + [norm_angle, norm_dist]
        return np.array(state, dtype=np.float32), dist

    def _smooth_action(self, action: np.ndarray) -> np.ndarray:
        steer = float(action[0])
        prev = float(self._prev_action[0])
        steer = clamp(steer, prev - MAX_STEER_DELTA, prev + MAX_STEER_DELTA)
        action[0] = steer
        self._ema_action = (1.0 - ACTION_EMA_ALPHA) * self._ema_action + ACTION_EMA_ALPHA * action
        self._ema_action = np.clip(self._ema_action, -1.0, 1.0)
        self._prev_action = self._ema_action.copy()
        return self._ema_action

    def _apply_straight_stabilizer(self, action_vec: np.ndarray, state_now: np.ndarray) -> np.ndarray:
        if self.escape_left > 0:
            return action_vec
        norm_angle = float(state_now[-2])
        if abs(norm_angle) > STRAIGHT_TARGET_ANGLE_MAX:
            return action_vec
        left = float(np.mean(state_now[0:3]))
        front = float(state_now[3])
        right = float(np.mean(state_now[4:7]))
        balance = abs(left - right)
        if front >= STRAIGHT_FRONT_MIN and balance <= STRAIGHT_BALANCE_MAX:
            action_vec = action_vec.copy()
            action_vec[0] = float(action_vec[0]) * STRAIGHT_STEER_DAMP
            if abs(float(action_vec[0])) < STEER_DEADBAND:
                action_vec[0] = 0.0
        return action_vec

    def _escape_action(self, state_now: np.ndarray) -> np.ndarray:
        best_i = int(np.argmax(state_now[:len(SENSOR_ANGLES)]))
        desired_ang = (self.car_angle + SENSOR_ANGLES[best_i]) % 360.0
        err = angle_diff(desired_ang, self.car_angle)
        steer = clamp((err / 45.0) * ESCAPE_STEER_GAIN, -1.0, 1.0)
        front_edge = float(state_now[len(SENSOR_ANGLES)//2])
        throttle = ESCAPE_REVERSE_THROTTLE if front_edge < ESCAPE_FRONT_TOO_CLOSE else ESCAPE_FORWARD_THROTTLE
        return np.array([steer, throttle], dtype=np.float32)

    def step(self, action_vec: np.ndarray) -> Tuple[np.ndarray, float, bool, str, Optional[int]]:
        """Returns (next_state, reward, done, reason, reached_target_idx)."""
        state_now, dist_before = self.get_state()
        if self.best_dist is None:
            self.best_dist = dist_before

        if self.escape_left > 0:
            self.escape_left -= 1
            action_vec = self._escape_action(state_now)
            reason_override = 'escaping'
        else:
            reason_override = None

        action_vec = np.array(action_vec, dtype=np.float32)
        action_vec = self._smooth_action(action_vec)
        action_vec = self._apply_straight_stabilizer(action_vec, state_now)

        steer = float(action_vec[0])
        throttle = float(action_vec[1])

        turn_delta = steer * MAX_TURN_DEG
        speed = BASE_SPEED + throttle * THROTTLE_GAIN
        speed = clamp(speed, -MAX_REVERSE, MAX_SPEED)

        new_angle = float((self.car_angle + turn_delta) % 360.0)
        rad = math.radians(new_angle)
        nx = float(self.car_pos.x() + math.cos(rad) * speed)
        ny = float(self.car_pos.y() + math.sin(rad) * speed)

        reward = -0.02
        done = False
        reason = 'moving' if reason_override is None else reason_override
        reached_idx = None

        if nx < 1 or ny < 1 or nx > self.w - 2 or ny > self.h - 2:
            reward = -100.0
            reason = 'respawn_out_of_bounds'
            self.count_oob += 1
            self.episode_reward += reward
            self.respawn_at_last_safe(align_to_opening=True)
            self.escape_left = ESCAPE_STEPS
            ns, _ = self.get_state()
            return ns, reward, False, reason, None

        if not self.is_on_road(nx, ny):
            reward = -100.0
            reason = 'respawn_off_road'
            self.count_offroad += 1
            self.episode_reward += reward
            self.respawn_at_last_safe(align_to_opening=True)
            self.escape_left = ESCAPE_STEPS
            ns, _ = self.get_state()
            return ns, reward, False, reason, None

        # Commit
        self.car_pos = QPointF(nx, ny)
        self.car_angle = new_angle
        self.last_safe_pos = QPointF(nx, ny)
        self.last_safe_angle = float(new_angle)

        # novelty reward (tripled)
        reward += self.novelty_reward(self.car_pos)

        # any target reached?
        hit, idx = self._check_any_target_reached()
        if hit:
            reached_idx = int(idx)
            self.target_reached[reached_idx] = True
            self.targets_reached = int(sum(self.target_reached))
            reward += 150.0
            reason = 'target_reached'

            if self._all_targets_done():
                reward += 220.0
                done = True
                reason = 'all_targets_reached'
            else:
                # choose next active target
                self._update_active_target()
                # reset shaping baseline for next target
                _, nd = self.get_state()
                self.prev_dist = nd
                self.best_dist = nd
                self.no_progress_ctr = 0

            self.episode_reward += reward
            ns, _ = self.get_state()
            return ns, reward, done, reason, reached_idx

        next_state, dist_after = self.get_state()

        # shaping to current active target
        if self.prev_dist is not None:
            delta = self.prev_dist - dist_after
            reward += clamp(delta * 0.7, -1.5, 1.5)
        self.prev_dist = dist_after

        # no-progress detection
        if dist_after < (self.best_dist - NO_PROGRESS_EPS):
            self.best_dist = dist_after
            self.no_progress_ctr = 0
        else:
            self.no_progress_ctr += 1

        if self.no_progress_ctr >= NO_PROGRESS_STEPS:
            reward -= 15.0
            reason = 'respawn_stuck'
            self.count_stuck += 1
            self.no_progress_ctr = 0
            self.episode_reward += reward
            self.respawn_at_last_safe(align_to_opening=True)
            self.escape_left = ESCAPE_STEPS
            ns, _ = self.get_state()
            return ns, reward, False, reason, None

        self.episode_reward += reward
        return next_state, reward, False, reason, None


# =====================
# Graphics Items
# =====================
class SensorDot(QGraphicsItem):
    def __init__(self, color: QColor):
        super().__init__()
        self.setZValue(90)
        self.color = color

    def set_color(self, color: QColor):
        self.color = color
        self.update()

    def boundingRect(self):
        return QRectF(-3, -3, 6, 6)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self.color))
        painter.drawEllipse(QPointF(0, 0), 2.5, 2.5)


class CarFallbackItem(QGraphicsItem):
    def __init__(self):
        super().__init__()
        self.setZValue(100)
        self.brush = QBrush(C_ACCENT)
        self.pen = QPen(Qt.GlobalColor.white, 1)

    def boundingRect(self):
        return QRectF(-CAR_DRAW_W/2, -CAR_DRAW_H/2, CAR_DRAW_W, CAR_DRAW_H)

    def paint(self, painter, option, widget=None):
        painter.setBrush(self.brush)
        painter.setPen(self.pen)
        painter.drawRoundedRect(self.boundingRect(), 3, 3)


class CarSpriteItem(QGraphicsPixmapItem):
    def __init__(self, pixmap: QPixmap, draw_w: int = CAR_DRAW_W, draw_h: int = CAR_DRAW_H):
        super().__init__()
        self.setZValue(100)
        pm = pixmap.scaled(draw_w, draw_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(pm)
        self.setTransformOriginPoint(pm.width()/2, pm.height()/2)
        self.setOffset(-pm.width()/2, -pm.height()/2)


class TargetItem(QGraphicsItem):
    def __init__(self, color=None, is_active=True, number=1):
        super().__init__()
        self.setZValue(60)
        self.color = color if color else QColor(0, 255, 255)
        self.is_active = is_active
        self.number = number
        self.reached = False

    def set_active(self, active: bool):
        self.is_active = active
        self.update()

    def set_reached(self, reached: bool):
        self.reached = reached
        self.update()

    def boundingRect(self):
        r = TARGET_RADIUS_ACTIVE + 8
        return QRectF(-r, -r, 2*r, 2*r)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # reached targets: grey-ish
        if self.reached:
            col = QColor(self.color)
            col.setAlpha(90)
            painter.setBrush(QBrush(col))
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            painter.drawEllipse(QPointF(0, 0), TARGET_RADIUS_INACTIVE, TARGET_RADIUS_INACTIVE)
        else:
            if self.is_active:
                painter.setBrush(QBrush(self.color))
                painter.setPen(QPen(Qt.GlobalColor.white, 2))
                painter.drawEllipse(QPointF(0, 0), TARGET_RADIUS_ACTIVE, TARGET_RADIUS_ACTIVE)
            else:
                dim = QColor(self.color); dim.setAlpha(140)
                painter.setBrush(QBrush(dim))
                painter.setPen(QPen(Qt.GlobalColor.white, 1))
                painter.drawEllipse(QPointF(0, 0), TARGET_RADIUS_INACTIVE, TARGET_RADIUS_INACTIVE)

        painter.setPen(QPen(Qt.GlobalColor.white))
        painter.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        painter.drawText(QRectF(-TARGET_RADIUS_ACTIVE, -TARGET_RADIUS_ACTIVE, 2*TARGET_RADIUS_ACTIVE, 2*TARGET_RADIUS_ACTIVE),
                         Qt.AlignmentFlag.AlignCenter, str(self.number))


class RewardChart(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(150)
        self.setStyleSheet(f"background-color: {C_PANEL.name()}; border-radius: 5px;")
        self.scores: List[float] = []
        self.max_points = 50

    def update_chart(self, new_score: float):
        self.scores.append(float(new_score))
        if len(self.scores) > self.max_points:
            self.scores.pop(0)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, C_PANEL)
        if len(self.scores) < 2:
            return
        mn, mx = min(self.scores), max(self.scores)
        if mx == mn:
            mx += 1
        step_x = w / (self.max_points - 1)
        pts = []
        for i, sc in enumerate(self.scores):
            x = i * step_x
            ratio = (sc - mn) / (mx - mn)
            y = h - (ratio * (h * 0.8) + (h * 0.1))
            pts.append(QPointF(x, y))
        path = QPainterPath(); path.moveTo(pts[0])
        for p in pts[1:]:
            path.lineTo(p)
        painter.setPen(QPen(C_ACCENT, 2))
        painter.drawPath(path)


# =====================
# App
# =====================
class NeuralNavApp(QMainWindow):
    def __init__(self):
        super().__init__()
        set_seed(SEED)

        self.setWindowTitle(f"NeuralNav TD3 v{VERSION} (Explore x3 + Any-order Targets)")
        self.resize(1500, 950)

        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {C_BG_DARK.name()}; }}
            QLabel {{ color: {C_TEXT.name()}; font-family: Arial, Helvetica, sans-serif; font-size: 13px; }}
            QPushButton {{ background-color: {C_PANEL.name()}; color: white; border: 1px solid {C_INFO_BG.name()}; padding: 8px; border-radius: 4px; }}
            QPushButton:hover {{ background-color: {C_INFO_BG.name()}; }}
            QPushButton:checked {{ background-color: {C_ACCENT.name()}; color: black; }}
            QTextEdit {{ background-color: {C_PANEL.name()}; color: #D8DEE9; border: none; font-family: Menlo, Consolas, monospace; font-size: 11px; }}
            QFrame {{ border: none; }}
        """)

        central = QWidget(); self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        panel = QFrame(); panel.setFixedWidth(460)
        panel.setStyleSheet(f"background-color: {C_BG_DARK.name()};")
        vbox = QVBoxLayout(panel); vbox.setSpacing(10)

        vbox.addWidget(QLabel("CONTROLS"))
        self.lbl_status = QLabel("1) Click Map -> START\n2) Click Map -> TARGET(S)\nRight-click when done")
        self.lbl_status.setStyleSheet(f"background-color: {C_INFO_BG.name()}; padding: 10px; border-radius: 5px;")
        vbox.addWidget(self.lbl_status)

        row1 = QHBoxLayout()
        self.btn_run = QPushButton("▶ START (Space)")
        self.btn_run.setCheckable(True)
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.toggle_training)
        row1.addWidget(self.btn_run)

        self.btn_load_map = QPushButton("📂 LOAD MAP")
        self.btn_load_map.clicked.connect(self.load_map_dialog)
        row1.addWidget(self.btn_load_map)
        vbox.addLayout(row1)

        row2 = QHBoxLayout()
        self.btn_load_car = QPushButton("🚗 LOAD CAR")
        self.btn_load_car.clicked.connect(self.load_car_dialog)
        row2.addWidget(self.btn_load_car)

        self.btn_reset = QPushButton("↺ RESET ALL (R)")
        self.btn_reset.clicked.connect(self.full_reset)
        row2.addWidget(self.btn_reset)
        vbox.addLayout(row2)

        vbox.addWidget(QLabel("REWARD HISTORY"))
        self.chart = RewardChart(); vbox.addWidget(self.chart)

        stats_frame = QFrame(); stats_frame.setStyleSheet(f"background-color: {C_PANEL.name()}; border-radius: 5px;")
        sf = QGridLayout(stats_frame); sf.setContentsMargins(10, 10, 10, 10)
        sf.addWidget(QLabel("Expl Noise:"), 0, 0)
        self.val_noise = QLabel(f"{EXPL_NOISE_START:.3f}"); self.val_noise.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf.addWidget(self.val_noise, 0, 1)
        sf.addWidget(QLabel("Last Reward:"), 1, 0)
        self.val_rew = QLabel("0"); self.val_rew.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf.addWidget(self.val_rew, 1, 1)
        sf.addWidget(QLabel("Replay:"), 2, 0)
        self.val_buf = QLabel("0"); self.val_buf.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf.addWidget(self.val_buf, 2, 1)
        sf.addWidget(QLabel("Off-road:"), 3, 0)
        self.val_off = QLabel("0"); self.val_off.setStyleSheet(f"color: {C_FAILURE.name()}; font-weight: bold;")
        sf.addWidget(self.val_off, 3, 1)
        sf.addWidget(QLabel("Out-of-bounds:"), 4, 0)
        self.val_oob = QLabel("0"); self.val_oob.setStyleSheet(f"color: {C_FAILURE.name()}; font-weight: bold;")
        sf.addWidget(self.val_oob, 4, 1)
        sf.addWidget(QLabel("Stuck:"), 5, 0)
        self.val_stuck = QLabel("0"); self.val_stuck.setStyleSheet(f"color: {C_FAILURE.name()}; font-weight: bold;")
        sf.addWidget(self.val_stuck, 5, 1)
        sf.addWidget(QLabel("Total respawns:"), 6, 0)
        self.val_total = QLabel("0"); self.val_total.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf.addWidget(self.val_total, 6, 1)
        vbox.addWidget(stats_frame)

        vbox.addWidget(QLabel("LOGS"))
        self.log_console = QTextEdit(); self.log_console.setReadOnly(True)
        vbox.addWidget(self.log_console)

        main_layout.addWidget(panel)

        # scene
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setStyleSheet(f"border: 2px solid {C_PANEL.name()}; background-color: {C_BG_DARK.name()}")
        self.view.mousePressEvent = self.on_scene_click
        main_layout.addWidget(self.view)

        # visuals
        self.sensor_green_lines: List[QGraphicsLineItem] = []
        self.sensor_red_lines: List[QGraphicsLineItem] = []
        self.sensor_dots: List[SensorDot] = []

        # trail safe recreation
        self.trail_item: Optional[QGraphicsPathItem] = None
        self.trail_pen = QPen(C_TRAIL, 2)
        self.trail_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self.trail_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        self.trail_path = QPainterPath()
        self._trail_has_point = False

        # car
        self.car_item: Optional[QGraphicsItem] = None
        self.car_sprite_path = DEFAULT_CAR_SPRITE

        # map
        self.map_img = None
        self.setup_map(DEFAULT_MAP)

        # brain
        self.brain = CarBrain(self.map_img)

        # overlays
        self._create_or_replace_car_item(self.car_sprite_path)

        self.setup_state = 0
        self.sim_timer = QTimer(); self.sim_timer.timeout.connect(self.game_loop)

        self.target_items: List[TargetItem] = []

        self.log(f"<b>TD3 v{VERSION}</b> ready. Exploration reward is 3x and targets are any-order.")

    def log(self, msg: str):
        self.log_console.append(msg)
        sb = self.log_console.verticalScrollBar(); sb.setValue(sb.maximum())

    # ---- Trail safe recreation ----
    def ensure_trail_item(self):
        if self.trail_item is None:
            self.trail_item = QGraphicsPathItem()
            self.trail_item.setZValue(30)
            self.trail_item.setPen(self.trail_pen)
            self.trail_item.setPath(self.trail_path)
            self.scene.addItem(self.trail_item)
        else:
            try:
                _ = self.trail_item.scene()
            except RuntimeError:
                self.trail_item = None
                return self.ensure_trail_item()
            if self.trail_item.scene() != self.scene:
                self.scene.addItem(self.trail_item)

    def trail_clear(self):
        self.trail_path = QPainterPath()
        self._trail_has_point = False
        self.ensure_trail_item()
        self.trail_item.setPath(self.trail_path)

    def trail_add_point(self, p: QPointF, break_line: bool = False):
        self.ensure_trail_item()
        if break_line or (not self._trail_has_point):
            self.trail_path.moveTo(p)
            self._trail_has_point = True
        else:
            self.trail_path.lineTo(p)
        self.trail_item.setPath(self.trail_path)

    # ---- Sensor visuals ----
    def _init_sensor_visuals(self):
        for it in list(self.sensor_green_lines) + list(self.sensor_red_lines) + list(self.sensor_dots):
            try:
                if it.scene() == self.scene:
                    self.scene.removeItem(it)
            except Exception:
                pass
        self.sensor_green_lines = []
        self.sensor_red_lines = []
        self.sensor_dots = []

        pen_g = QPen(C_SENSOR_GREEN, 2)
        pen_r = QPen(C_SENSOR_RED, 2)
        for _ in SENSOR_ANGLES:
            lg = QGraphicsLineItem(); lg.setZValue(80); lg.setPen(pen_g)
            lr = QGraphicsLineItem(); lr.setZValue(80); lr.setPen(pen_r)
            self.scene.addItem(lg); self.scene.addItem(lr)
            self.sensor_green_lines.append(lg)
            self.sensor_red_lines.append(lr)
            dot = SensorDot(C_SENSOR_GREEN)
            dot.setZValue(85)
            self.scene.addItem(dot)
            self.sensor_dots.append(dot)

    # ---- Map ----
    def create_dummy_map(self, path: str):
        img = QImage(1000, 800, QImage.Format.Format_RGB32)
        img.fill(C_BG_DARK)
        p = QPainter(img)
        p.setBrush(Qt.GlobalColor.white)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(100, 100, 800, 600)
        p.setBrush(C_BG_DARK)
        p.drawEllipse(250, 250, 500, 300)
        p.end()
        img.save(path)

    def setup_map(self, path: str):
        if not os.path.exists(path):
            self.create_dummy_map(path)
        self.map_img = QImage(path).convertToFormat(QImage.Format.Format_RGB32)
        self.scene.clear()
        self.scene.addPixmap(QPixmap.fromImage(self.map_img))
        self._init_sensor_visuals()
        self.ensure_trail_item()

    # ---- Car ----
    def _create_or_replace_car_item(self, img_path: str):
        if self.car_item is not None:
            try:
                if self.car_item.scene() == self.scene:
                    self.scene.removeItem(self.car_item)
            except Exception:
                pass
            self.car_item = None

        pm = QPixmap(img_path) if img_path and os.path.exists(img_path) else QPixmap()
        if not pm.isNull():
            self.car_item = CarSpriteItem(pm, CAR_DRAW_W, CAR_DRAW_H)
        else:
            self.car_item = CarFallbackItem()
        self.scene.addItem(self.car_item)

    def load_car_dialog(self):
        f, _ = QFileDialog.getOpenFileName(self, "Load Car Image", "", "Images (*.png *.jpg *.jpeg)")
        if f:
            self.car_sprite_path = f
            self._create_or_replace_car_item(f)
            self.log(f"Loaded car image: <b>{os.path.basename(f)}</b>")

    def load_map_dialog(self):
        f, _ = QFileDialog.getOpenFileName(self, "Load Map", "", "Images (*.png *.jpg *.jpeg)")
        if f:
            self.full_reset(clear_logs=False)
            self.setup_map(f)
            self.brain = CarBrain(self.map_img)
            self._create_or_replace_car_item(self.car_sprite_path)
            self.trail_clear()
            self.log(f"Loaded map: <b>{os.path.basename(f)}</b>")

    # ---- Interactions ----
    def on_scene_click(self, event):
        pt = self.view.mapToScene(event.pos())
        if self.setup_state == 0:
            self.brain.set_start_pos(pt)
            self.car_item.setPos(pt)
            self.trail_add_point(pt, break_line=True)
            self.setup_state = 1
            self.lbl_status.setText("Click targets (Left). Right-click when done")
        elif self.setup_state == 1:
            if event.button() == Qt.MouseButton.LeftButton:
                self.brain.add_target(pt)
                idx = len(self.brain.targets) - 1
                color = TARGET_COLORS[idx % len(TARGET_COLORS)]
                num = len(self.brain.targets)
                ti = TargetItem(color, idx == 0, num)
                ti.setPos(pt)
                self.scene.addItem(ti)
                self.target_items.append(ti)
                self.lbl_status.setText(f"Targets: {num}\nRight-click to finish")
            elif event.button() == Qt.MouseButton.RightButton:
                if len(self.brain.targets) > 0:
                    # set active target as nearest unreached
                    self.brain._update_active_target()
                    self.setup_state = 2
                    self.btn_run.setEnabled(True)
                    self.lbl_status.setText(f"READY: {len(self.brain.targets)} target(s). Press Space")
                    self.brain.reset_episode_keep_targets(reset_counters=True)
                    self.update_visuals(break_trail=False)

    def full_reset(self, clear_logs: bool = True):
        self.sim_timer.stop()
        self.btn_run.setChecked(False)
        self.btn_run.setEnabled(False)
        self.btn_run.setText("▶ START (Space)")
        self.setup_state = 0

        for t in self.target_items:
            try:
                if t.scene() == self.scene:
                    self.scene.removeItem(t)
            except Exception:
                pass
        self.target_items = []

        self.brain.targets = []
        self.brain.target_reached = []
        self.brain.current_target_idx = 0
        self.brain.targets_reached = 0
        self.brain.target_pos = QPointF(200, 200)

        self.lbl_status.setText("1) Click Map -> START\n2) Click Map -> TARGET(S)")
        self.chart.scores = []; self.chart.update()
        self.trail_clear()
        if clear_logs:
            self.log_console.clear()
        self.log(f"<b>TD3 v{VERSION}</b> reset.")

    def toggle_training(self):
        if self.btn_run.isChecked():
            self.sim_timer.start(16)
            self.btn_run.setText("⏸ PAUSE")
        else:
            self.sim_timer.stop()
            self.btn_run.setText("▶ RESUME")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space and self.setup_state == 2:
            self.btn_run.click()
        elif event.key() == Qt.Key.Key_R:
            self.full_reset()

    def game_loop(self):
        if self.setup_state != 2:
            return

        state, _ = self.brain.get_state()
        a = self.brain.agent.act(state, explore=True)
        next_state, reward, done, reason, reached_idx = self.brain.step(a)

        # store transition (TD3 expects 5 tuple, but we store 4: done float)
        self.brain.agent.replay.add(state, a, reward, next_state, float(done))
        if len(self.brain.agent.replay) >= START_TRAIN_AFTER:
            self.brain.agent.train_step()

        self.val_noise.setText(f"{self.brain.agent.expl_noise:.3f}")
        self.val_rew.setText(f"{reward:.2f}")
        self.val_buf.setText(str(len(self.brain.agent.replay)))
        self.val_off.setText(str(self.brain.count_offroad))
        self.val_oob.setText(str(self.brain.count_oob))
        self.val_stuck.setText(str(self.brain.count_stuck))
        self.val_total.setText(str(self.brain.total_respawns()))

        break_trail = reason in ('respawn_off_road', 'respawn_out_of_bounds', 'respawn_stuck')
        self.update_visuals(break_trail=break_trail)

        if reached_idx is not None:
            total = len(self.brain.targets)
            self.log(f"<font color='{C_ACCENT.name()}'><b>🎯 Reached target #{reached_idx+1}</b> ({self.brain.targets_reached}/{total})</font>")

        if reason == 'all_targets_reached':
            total = len(self.brain.targets)
            self.log(f"<font color='{C_SUCCESS.name()}'><b>✅ ALL TARGETS REACHED</b> ({total}/{total})</font>")

        if done:
            self.chart.update_chart(self.brain.episode_reward)
            if reason == 'all_targets_reached':
                self.sim_timer.stop()
                self.btn_run.setChecked(False)
                self.btn_run.setEnabled(False)
                self.btn_run.setText("✅ COMPLETED")

    def update_visuals(self, break_trail: bool = False):
        self.car_item.setPos(self.brain.car_pos)
        self.car_item.setRotation(self.brain.car_angle)

        self.trail_add_point(self.brain.car_pos, break_line=break_trail)

        # update targets visual state
        for i, t in enumerate(self.target_items):
            t.set_active(i == self.brain.current_target_idx and (not self.brain.target_reached[i]))
            t.set_reached(self.brain.target_reached[i] if i < len(self.brain.target_reached) else False)

        # sensors
        cx, cy = self.brain.car_pos.x(), self.brain.car_pos.y()
        for i, (green_end, full_end, edge_norm) in enumerate(self.brain.sensor_rays):
            if i >= len(self.sensor_green_lines):
                continue
            self.sensor_green_lines[i].setLine(cx, cy, green_end.x(), green_end.y())
            self.sensor_red_lines[i].setLine(green_end.x(), green_end.y(), full_end.x(), full_end.y())
            self.sensor_dots[i].setPos(green_end)
            self.sensor_dots[i].set_color(C_SENSOR_GREEN if edge_norm > 0.25 else C_SENSOR_RED)

        self.scene.update()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = NeuralNavApp()
    win.show()
    sys.exit(app.exec())
