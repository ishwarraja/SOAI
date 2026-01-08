
""" 
CITY MAP NAVIGATION (A*) - v10.1 (Hybrid Road Detection: Works on BOTH Map Types)
================================================================================

You asked for a COMPLETE code version (based on v10) that works on BOTH:
1) Road-only / binary maps (white roads on dark background) like City_Image_1_converted.png
2) Colored city maps with white roads over colored regions like City_Map.png

Key changes vs v10.0:
✅ Hybrid road detection (robust across binary + colored maps)
   - A pixel is considered ROAD if:
       (V >= ROAD_V_THR and S <= ROAD_S_THR) OR (min(R,G,B) >= ROAD_MIN_THR)
     where V,S are from HSV computed on-the-fly.

✅ Planner grid build uses multi-sample per cell (not single center pixel)
   - Greatly improves A* connectivity on curved / thin roads.

✅ Option 2 image normalization
   - Any loaded image is resized to fit MAX_MAP_W x MAX_MAP_H (aspect preserved).

✅ Majority-vote road footprint for car motion
   - Prevents false OFF_ROAD due to anti-aliased edges.

✅ Stuck handling
   - Anchor-based stuck + circle/loop trap detection.
   - One local escape attempt, else restart mission from start (Target #1).

✅ Stuck counter displayed in UI.
✅ Red sensors drawn every frame.
✅ Log spam suppression with (xN) aggregation.

Run:
  python citymap_assignment_fixed_v10_1_hybrid_both_maps.py
"""

import sys
import os
import math
import time
import heapq
from collections import deque

import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGraphicsScene, QGraphicsView, QGraphicsItem,
    QFrame, QFileDialog, QTextEdit, QGraphicsPathItem, QGraphicsLineItem
)
from PyQt6.QtGui import (
    QImage, QPixmap, QColor, QPen, QBrush, QPainter, QFont, QPainterPath, QTextCursor
)
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF

# =====================
# OPTION 2: MAP NORMALIZATION
# =====================
MAX_MAP_W = 1200
MAX_MAP_H = 800
USE_SMOOTH_SCALING = True

# =====================
# THEME / COLORS
# =====================
C_BG_DARK   = QColor("#2E3440")
C_PANEL     = QColor("#3B4252")
C_INFO_BG   = QColor("#4C566A")
C_ACCENT    = QColor("#88C0D0")
C_TEXT      = QColor("#ECEFF4")
C_SUCCESS   = QColor("#A3BE8C")
C_FAILURE   = QColor("#BF616A")

C_PLANNED_PATH = QColor(136, 192, 208, 190)
C_TRAIL        = QColor(255, 0, 0, 210)

HEADING_LEN = 26

TARGET_COLORS = [
    QColor(0, 255, 255), QColor(255, 100, 255), QColor(0, 255, 100), QColor(255, 150, 0),
    QColor(100, 150, 255), QColor(255, 50, 150), QColor(150, 255, 50), QColor(255, 255, 0)
]

DEFAULT_MAP = 'City_Image_1_converted.png'  # You can change or use LOAD MAP

# =====================
# CAR / MOTION
# =====================
CAR_WIDTH  = 14
CAR_HEIGHT = 8

SPEED = 1.15
MAX_TURN_PER_TICK = 10.0

GOAL_RADIUS = 35

# =====================
# HYBRID ROAD DETECTION (works on both maps)
# =====================
# HSV-based thresholds
ROAD_V_THR = 0.78   # brightness
ROAD_S_THR = 0.28   # saturation (white has low saturation)

# RGB-min threshold backup (helps binary maps and very bright whites)
ROAD_MIN_THR = 0.85

# Road footprint sampling (9 points, majority vote)
ROAD_FOOTPRINT_R = 3
ROAD_MIN_HITS = 5

# =====================
# PLANNER
# =====================
CELL = 6
NEIGHBOR_8 = True
REPLAN_EVERY_N = 12

CRASH_PENALTY_RADIUS_CELLS = 6
CRASH_PENALTY_ADD = 2.5
CRASH_PENALTY_CAP = 25.0

# Pure pursuit lookahead
LOOKAHEAD_DIST = 28

# Planner walkability sampling inside each cell (multi-sample)
CELL_SAMPLES = 5          # 5-point (center + 4 corners-ish)
CELL_MIN_HITS = 3         # required on-road samples to mark cell walkable

# =====================
# STUCK + ESCAPE
# =====================
STUCK_SECONDS = 20.0
STUCK_ANCHOR_RADIUS = 30.0

STUCK_WINDOW = 10.0
STUCK_AREA_RADIUS = 55.0
STUCK_TURN_DEG = 900.0
STUCK_PATHLEN = 260.0

ESCAPE_SECONDS = 4.0
ESCAPE_SPEED = 0.95
ESCAPE_TURN = 30.0
ESCAPE_SUCCESS_DIST = 22.0

RESET_PENALTIES_ON_RESTART = False

# =====================
# TRAIL
# =====================
TRAIL_MAX_POINTS = 6000
TRAIL_DOWNSAMPLE = 1

# =====================
# SENSORS (VISUAL)
# =====================
SENSOR_DIST = 65
SENSOR_ANGLES = [-60, -40, -20, 0, 20, 40, 60]
SENSOR_PEN = QColor(255, 0, 0, 220)


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def angle_diff(a: float, b: float) -> float:
    """Smallest signed angle difference a-b in degrees."""
    d = (a - b) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


class RoadClassifier:
    """Hybrid road detector for both binary and colored maps."""

    def __init__(self, v_thr=ROAD_V_THR, s_thr=ROAD_S_THR, min_thr=ROAD_MIN_THR):
        self.v_thr = float(v_thr)
        self.s_thr = float(s_thr)
        self.min_thr = float(min_thr)

    @staticmethod
    def rgb_to_vs(r: int, g: int, b: int):
        """Fast HSV (only V and S) from 0..255 RGB."""
        rf = r / 255.0
        gf = g / 255.0
        bf = b / 255.0
        mx = rf if rf > gf else gf
        mx = mx if mx > bf else bf
        mn = rf if rf < gf else gf
        mn = mn if mn < bf else bf
        v = mx
        if mx <= 1e-9:
            s = 0.0
        else:
            s = (mx - mn) / mx
        return v, s

    def is_road_rgb(self, r: int, g: int, b: int) -> bool:
        # RGB-min fallback (helps pure binary maps)
        mn = min(r, g, b) / 255.0
        if mn >= self.min_thr:
            return True
        # HSV test
        v, s = self.rgb_to_vs(r, g, b)
        return (v >= self.v_thr) and (s <= self.s_thr)

    def is_road_qcolor(self, c: QColor) -> bool:
        return self.is_road_rgb(c.red(), c.green(), c.blue())


class CarItem(QGraphicsItem):
    def __init__(self):
        super().__init__()
        self.setZValue(100)
        self.brush = QBrush(C_ACCENT)
        self.pen = QPen(Qt.GlobalColor.white, 1)

    def boundingRect(self):
        return QRectF(-CAR_WIDTH/2, -CAR_HEIGHT/2, CAR_WIDTH, CAR_HEIGHT)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(self.brush)
        painter.setPen(self.pen)
        painter.drawRoundedRect(self.boundingRect(), 2, 2)
        painter.setBrush(Qt.GlobalColor.white)
        painter.drawRect(int(CAR_WIDTH/2)-2, -3, 2, 6)


class TargetItem(QGraphicsItem):
    def __init__(self, color=None, is_active=True, number=1):
        super().__init__()
        self.setZValue(50)
        self.color = color if color else QColor(0, 255, 255)
        self.is_active = is_active
        self.number = number

    def set_active(self, active: bool):
        self.is_active = active
        self.update()

    def boundingRect(self):
        return QRectF(-20, -20, 40, 40)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.is_active:
            painter.setBrush(QBrush(self.color))
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            painter.drawEllipse(QPointF(0, 0), 8, 8)
        else:
            dim = QColor(self.color)
            dim.setAlpha(120)
            painter.setBrush(QBrush(dim))
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            painter.drawEllipse(QPointF(0, 0), 6, 6)

        painter.setPen(QPen(Qt.GlobalColor.white))
        painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        painter.drawText(QRectF(-10, -10, 20, 20), Qt.AlignmentFlag.AlignCenter, str(self.number))


class RoadPlanner:
    def __init__(self, qimg: QImage, classifier: RoadClassifier):
        self.img = qimg
        self.clf = classifier
        self.w, self.h = qimg.width(), qimg.height()
        self.cell = CELL
        self.gw = max(1, self.w // self.cell)
        self.gh = max(1, self.h // self.cell)

        self.walk = np.zeros((self.gh, self.gw), dtype=np.uint8)
        self.penalty = np.zeros((self.gh, self.gw), dtype=np.float32)

        # sample offsets within each grid cell
        # use fraction of cell size
        s = max(1, self.cell // 3)
        samples = [(0, 0), (s, 0), (-s, 0), (0, s), (0, -s)]
        if CELL_SAMPLES >= 9:
            samples += [(s, s), (-s, s), (s, -s), (-s, -s)]

        for gy in range(self.gh):
            cy = int((gy + 0.5) * self.cell)
            cy = min(self.h - 1, max(0, cy))
            for gx in range(self.gw):
                cx = int((gx + 0.5) * self.cell)
                cx = min(self.w - 1, max(0, cx))
                hits = 0
                for ox, oy in samples:
                    x = int(max(0, min(self.w - 1, cx + ox)))
                    y = int(max(0, min(self.h - 1, cy + oy)))
                    c = QColor(self.img.pixel(x, y))
                    if self.clf.is_road_qcolor(c):
                        hits += 1
                self.walk[gy, gx] = 1 if hits >= CELL_MIN_HITS else 0

        self.nei = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if NEIGHBOR_8:
            self.nei += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    def to_cell(self, p: QPointF):
        gx = int(p.x() // self.cell)
        gy = int(p.y() // self.cell)
        return max(0, min(self.gw - 1, gx)), max(0, min(self.gh - 1, gy))

    def to_point(self, gx: int, gy: int):
        return QPointF((gx + 0.5) * self.cell, (gy + 0.5) * self.cell)

    def is_walk(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.gw and 0 <= gy < self.gh and bool(self.walk[gy, gx])

    def nearest_road_cell(self, gx: int, gy: int, radius_cells: int = 50):
        if self.is_walk(gx, gy):
            return gx, gy
        for r in range(1, radius_cells + 1):
            for dy in range(-r, r + 1):
                y = gy + dy
                x1 = gx - r
                x2 = gx + r
                if self.is_walk(x1, y):
                    return x1, y
                if self.is_walk(x2, y):
                    return x2, y
            for dx in range(-r + 1, r):
                x = gx + dx
                y1 = gy - r
                y2 = gy + r
                if self.is_walk(x, y1):
                    return x, y1
                if self.is_walk(x, y2):
                    return x, y2
        return None

    def add_crash_penalty(self, gx: int, gy: int, radius: int = CRASH_PENALTY_RADIUS_CELLS, add: float = CRASH_PENALTY_ADD):
        if gx is None or gy is None:
            return
        for dy in range(-radius, radius + 1):
            yy = gy + dy
            if yy < 0 or yy >= self.gh:
                continue
            for dx in range(-radius, radius + 1):
                xx = gx + dx
                if xx < 0 or xx >= self.gw:
                    continue
                if not self.is_walk(xx, yy):
                    continue
                d = math.hypot(dx, dy)
                if d > radius:
                    continue
                inc = add / (1.0 + d)
                self.penalty[yy, xx] = float(min(CRASH_PENALTY_CAP, self.penalty[yy, xx] + inc))

    def astar(self, start, goal):
        sx, sy = start
        gx, gy = goal
        if not self.is_walk(sx, sy) or not self.is_walk(gx, gy):
            return None

        def h(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        open_heap = [(0.0, (sx, sy))]
        came = {}
        gscore = {(sx, sy): 0.0}
        closed = set()

        while open_heap:
            _, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            if cur == (gx, gy):
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                path.reverse()
                return path

            closed.add(cur)
            cx, cy = cur
            for dx, dy in self.nei:
                nx, ny = cx + dx, cy + dy
                if not self.is_walk(nx, ny):
                    continue
                step = 1.4142 if (dx != 0 and dy != 0) else 1.0
                pen = float(self.penalty[ny, nx])
                ng = gscore[cur] + step + pen
                if ng < gscore.get((nx, ny), 1e18):
                    gscore[(nx, ny)] = ng
                    came[(nx, ny)] = cur
                    heapq.heappush(open_heap, (ng + h((nx, ny), (gx, gy)), (nx, ny)))
        return None


class NavBrain:
    def __init__(self, map_img: QImage):
        self.map = map_img
        self.w, self.h = map_img.width(), map_img.height()
        self.clf = RoadClassifier()
        self.planner = RoadPlanner(map_img, self.clf)

        self.start_pos = QPointF(100, 100)
        self.car_pos = QPointF(100, 100)
        self.car_angle = 0.0

        self.targets = []
        self.current_target_idx = 0
        self.targets_reached = 0
        self.target_pos = QPointF(200, 200)

        self.route_cells = None
        self.route_pts = deque()
        self.replan_counter = 0
        self.last_crash_pos = None

        # stuck anchor + history
        self._anchor_pos = QPointF(self.car_pos)
        self._anchor_time = time.monotonic()
        self._hist = deque()  # (t,x,y,angle)

        # escape
        self._escape_active = False
        self._escape_attempted = False
        self._escape_deadline = 0.0
        self._escape_origin = QPointF(self.car_pos)
        self._escape_turn_dir = 1

    # -----------
    # Position snapping
    # -----------
    def _snap_to_road(self, p: QPointF) -> QPointF:
        gx, gy = self.planner.to_cell(p)
        nn = self.planner.nearest_road_cell(gx, gy)
        if nn is None:
            return p
        return self.planner.to_point(nn[0], nn[1])

    # -----------
    # Lookahead waypoint
    # -----------
    def _choose_lookahead_point(self):
        if not self.route_pts:
            return None
        for p in self.route_pts:
            if math.hypot(p.x() - self.car_pos.x(), p.y() - self.car_pos.y()) >= LOOKAHEAD_DIST:
                return p
        return self.route_pts[-1]

    # -----------
    # Road check (9 samples majority vote)
    # -----------
    def _road_ok(self, x: float, y: float) -> bool:
        r = ROAD_FOOTPRINT_R
        offsets = [(0, 0), (r, 0), (-r, 0), (0, r), (0, -r), (r, r), (-r, r), (r, -r), (-r, -r)]
        hits = 0
        for ox, oy in offsets:
            px = int(max(0, min(self.w - 1, x + ox)))
            py = int(max(0, min(self.h - 1, y + oy)))
            c = QColor(self.map.pixel(px, py))
            if self.clf.is_road_qcolor(c):
                hits += 1
        return hits >= ROAD_MIN_HITS

    # -----------
    # Reset helpers
    # -----------
    def _reset_anchor(self):
        self._anchor_pos = QPointF(self.car_pos)
        self._anchor_time = time.monotonic()

    def _reset_escape(self):
        self._escape_active = False
        self._escape_attempted = False
        self._escape_deadline = 0.0
        self._escape_origin = QPointF(self.car_pos)
        self._escape_turn_dir = 1

    # -----------
    # Public reset methods
    # -----------
    def set_start_pos(self, pt: QPointF):
        self.car_pos = self._snap_to_road(QPointF(pt.x(), pt.y()))
        self.start_pos = QPointF(self.car_pos)
        self._reset_anchor(); self._reset_escape(); self._hist.clear()

    def add_target(self, pt: QPointF):
        self.targets.append(QPointF(pt.x(), pt.y()))
        if len(self.targets) == 1:
            self.current_target_idx = 0
            self.target_pos = self.targets[0]

    def reset_full(self):
        self.car_pos = QPointF(self.start_pos)
        self.car_angle = float(np.random.randint(0, 360))
        self.current_target_idx = 0
        self.targets_reached = 0
        if self.targets:
            self.target_pos = self.targets[0]
        self.route_cells = None
        self.route_pts.clear()
        self.replan_counter = 0
        self.last_crash_pos = None
        self._reset_anchor(); self._reset_escape(); self._hist.clear()

    def restart_mission(self):
        self.car_pos = QPointF(self.start_pos)
        self.car_angle = float(np.random.randint(0, 360))
        self.current_target_idx = 0
        self.targets_reached = 0
        if self.targets:
            self.target_pos = self.targets[0]
        self.route_cells = None
        self.route_pts.clear()
        self.replan_counter = 0
        self.last_crash_pos = None
        self._reset_anchor(); self._reset_escape(); self._hist.clear()
        if RESET_PENALTIES_ON_RESTART:
            try:
                self.planner.penalty[:] = 0.0
            except Exception:
                pass

    # -----------
    # Planner / crash penalty
    # -----------
    def record_crash(self, pos: QPointF):
        if pos is None:
            return
        gx, gy = self.planner.to_cell(pos)
        nn = self.planner.nearest_road_cell(gx, gy)
        if nn is None:
            return
        self.planner.add_crash_penalty(nn[0], nn[1])

    def respawn_keep_progress(self):
        base = self.last_crash_pos if self.last_crash_pos is not None else self.start_pos
        self.car_pos = self._snap_to_road(QPointF(base.x(), base.y()))
        self.car_angle = float(np.random.randint(0, 360))
        if self.targets:
            self.target_pos = self.targets[self.current_target_idx]
        self.route_cells = None
        self.route_pts.clear()
        self.replan_counter = 0
        self._reset_anchor(); self._reset_escape(); self._hist.clear()

    def plan_route(self) -> bool:
        self.car_pos = self._snap_to_road(self.car_pos)
        self.target_pos = self._snap_to_road(self.target_pos)

        start = self.planner.to_cell(self.car_pos)
        goal = self.planner.to_cell(self.target_pos)
        s2 = self.planner.nearest_road_cell(*start)
        g2 = self.planner.nearest_road_cell(*goal)
        if s2 is None or g2 is None:
            return False

        path = self.planner.astar(s2, g2)
        if path is None or len(path) < 2:
            return False

        self.route_cells = path
        self.route_pts = deque(self.planner.to_point(x, y) for x, y in path)

        if self.route_pts:
            p0 = self.route_pts[0]
            if math.hypot(p0.x() - self.car_pos.x(), p0.y() - self.car_pos.y()) < self.planner.cell * 1.5:
                self.route_pts.popleft()

        self.replan_counter = 0
        return True

    # -----------
    # Stuck detection
    # -----------
    def _update_history(self):
        now = time.monotonic()
        self._hist.append((now, float(self.car_pos.x()), float(self.car_pos.y()), float(self.car_angle)))
        while self._hist and (now - self._hist[0][0]) > STUCK_WINDOW:
            self._hist.popleft()

    def _circle_trap_detected(self) -> bool:
        if len(self._hist) < 8:
            return False
        t0 = self._hist[0][0]
        t1 = self._hist[-1][0]
        if (t1 - t0) < 6.0:
            return False

        xs = np.array([h[1] for h in self._hist], dtype=np.float32)
        ys = np.array([h[2] for h in self._hist], dtype=np.float32)
        angs = np.array([h[3] for h in self._hist], dtype=np.float32)

        cx = float(xs.mean()); cy = float(ys.mean())
        rad = float(np.max(np.hypot(xs - cx, ys - cy)))
        if rad > STUCK_AREA_RADIUS:
            return False

        path_len = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))

        total_turn = 0.0
        for i in range(1, len(angs)):
            total_turn += abs(angle_diff(float(angs[i]), float(angs[i-1])))

        return (total_turn >= STUCK_TURN_DEG) or (path_len >= STUCK_PATHLEN)

    def _anchor_stuck_detected(self) -> bool:
        if math.hypot(self.car_pos.x() - self._anchor_pos.x(), self.car_pos.y() - self._anchor_pos.y()) > STUCK_ANCHOR_RADIUS:
            self._reset_anchor()
            self._escape_attempted = False
            return False
        return (time.monotonic() - self._anchor_time) >= STUCK_SECONDS

    # -----------
    # Escape
    # -----------
    def begin_escape(self):
        self._escape_active = True
        self._escape_attempted = True
        self._escape_deadline = time.monotonic() + ESCAPE_SECONDS
        self._escape_origin = QPointF(self.car_pos)
        self._escape_turn_dir = 1 if np.random.rand() < 0.5 else -1
        self._reset_anchor()

    def _escape_step(self):
        now = time.monotonic()

        if math.hypot(self.car_pos.x() - self._escape_origin.x(), self.car_pos.y() - self._escape_origin.y()) >= ESCAPE_SUCCESS_DIST:
            self._escape_active = False
            self.route_pts.clear(); self.route_cells = None
            self.replan_counter = 0
            return True, 'escape_success'

        if now >= self._escape_deadline:
            self._escape_active = False
            return True, 'escape_failed'

        candidates = [
            self.car_angle + self._escape_turn_dir * ESCAPE_TURN,
            self.car_angle - self._escape_turn_dir * ESCAPE_TURN,
            self.car_angle + 180.0,
        ]

        moved = False
        for ang in candidates:
            rad = math.radians(ang % 360.0)
            nx = self.car_pos.x() + math.cos(rad) * ESCAPE_SPEED
            ny = self.car_pos.y() + math.sin(rad) * ESCAPE_SPEED
            if nx < 1 or ny < 1 or nx > self.w - 2 or ny > self.h - 2:
                continue
            if self._road_ok(nx, ny):
                self.car_angle = float(ang % 360.0)
                self.car_pos = QPointF(nx, ny)
                moved = True
                break
        if not moved:
            self.car_angle = float((self.car_angle + self._escape_turn_dir * ESCAPE_TURN) % 360.0)

        return False, 'escaping'

    # -----------
    # Simulation step
    # -----------
    def step(self):
        self._update_history()

        if self._escape_active:
            _, status = self._escape_step()
            if status == 'escape_success':
                self.last_crash_pos = None
                return False, 'escape_success'
            if status == 'escape_failed':
                self.last_crash_pos = QPointF(self.car_pos.x(), self.car_pos.y())
                return True, 'stuck'
            return False, 'escaping'

        stuck_now = self._anchor_stuck_detected() or self._circle_trap_detected()
        if stuck_now:
            if not self._escape_attempted:
                self.begin_escape()
                return False, 'escaping'
            self.last_crash_pos = QPointF(self.car_pos.x(), self.car_pos.y())
            return True, 'stuck'

        if (not self.route_pts) or (self.replan_counter >= REPLAN_EVERY_N):
            if not self.plan_route():
                self.last_crash_pos = QPointF(self.car_pos.x(), self.car_pos.y())
                return True, 'no_path'
        self.replan_counter += 1

        # goal reached?
        if math.hypot(self.target_pos.x() - self.car_pos.x(), self.target_pos.y() - self.car_pos.y()) <= GOAL_RADIUS:
            self.targets_reached += 1
            self.last_crash_pos = None
            if self.current_target_idx < len(self.targets) - 1:
                self.current_target_idx += 1
                self.target_pos = self.targets[self.current_target_idx]
                self.route_pts.clear(); self.route_cells = None
                self._reset_anchor(); self._escape_attempted = False
                self._hist.clear()
                return False, 'target_reached'
            return True, 'completed'

        # pop reached waypoints
        while self.route_pts and math.hypot(self.route_pts[0].x() - self.car_pos.x(), self.route_pts[0].y() - self.car_pos.y()) < self.planner.cell * 1.2:
            self.route_pts.popleft()

        look = self._choose_lookahead_point()
        if look is None:
            return False, 'moving'

        desired = math.degrees(math.atan2(look.y() - self.car_pos.y(), look.x() - self.car_pos.x()))
        turn = clamp(angle_diff(desired, self.car_angle) * 0.25, -MAX_TURN_PER_TICK, MAX_TURN_PER_TICK)
        self.car_angle = float((self.car_angle + turn) % 360.0)

        rad = math.radians(self.car_angle)
        nx = self.car_pos.x() + math.cos(rad) * SPEED
        ny = self.car_pos.y() + math.sin(rad) * SPEED

        if nx < 1 or ny < 1 or nx > self.w - 2 or ny > self.h - 2:
            self.last_crash_pos = QPointF(nx, ny)
            return True, 'out_of_bounds'

        if not self._road_ok(nx, ny):
            self.last_crash_pos = QPointF(nx, ny)
            return True, 'off_road'

        self.car_pos = QPointF(nx, ny)
        return False, 'moving'


class NeuralNavApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('NeuralNav: A* v10.1 (Hybrid, both maps)')
        self.resize(1300, 850)

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

        # Left panel
        panel = QFrame(); panel.setFixedWidth(470)
        panel.setStyleSheet(f"background-color: {C_BG_DARK.name()};")
        vbox = QVBoxLayout(panel); vbox.setSpacing(10)
        vbox.addWidget(QLabel('CONTROLS'))

        self.lbl_status = QLabel(
            """1) Click Map -> CAR
2) Click Map -> TARGET(S)
Right-click when done

Cyan = planned route
Red = trail
White = heading
Red rays = sensors
Stuck => escape then restart (Target #1)

Hybrid road detector: works on binary + colored maps"""
        )
        self.lbl_status.setStyleSheet(f"background-color: {C_INFO_BG.name()}; padding: 10px; border-radius: 5px;")
        vbox.addWidget(self.lbl_status)

        self.lbl_stuck = QLabel('Stuck count: 0')
        self.lbl_stuck.setStyleSheet(f"background-color: {C_PANEL.name()}; padding: 8px; border-radius: 5px;")
        vbox.addWidget(self.lbl_stuck)

        self.btn_run = QPushButton('▶ START (Space)')
        self.btn_run.setCheckable(True)
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.toggle)
        vbox.addWidget(self.btn_run)

        self.btn_reset = QPushButton('↺ RESET ALL (R)')
        self.btn_reset.clicked.connect(self.full_reset)
        vbox.addWidget(self.btn_reset)

        self.btn_load = QPushButton('📂 LOAD MAP')
        self.btn_load.clicked.connect(self.load_map_dialog)
        vbox.addWidget(self.btn_load)

        vbox.addWidget(QLabel('LOGS'))
        self.log_console = QTextEdit(); self.log_console.setReadOnly(True)
        vbox.addWidget(self.log_console)

        main_layout.addWidget(panel)

        # Scene
        self.scene = QGraphicsScene()
        self.map_pixmap_item = None

        self.view = QGraphicsView(self.scene)
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setStyleSheet(f"border: 2px solid {C_PANEL.name()}; background-color: {C_BG_DARK.name()}")
        self.view.mousePressEvent = self.on_click
        main_layout.addWidget(self.view)

        self.sim_timer = QTimer(); self.sim_timer.timeout.connect(self.game_loop)
        self.setup_state = 0

        # overlays
        self.car_item = CarItem()
        self.target_items = []

        self.path_item = QGraphicsPathItem(); self.path_item.setZValue(20)
        self.path_item.setPen(QPen(C_PLANNED_PATH, 3))
        self.scene.addItem(self.path_item)

        self.trail_item = QGraphicsPathItem(); self.trail_item.setZValue(15)
        self.trail_item.setPen(QPen(C_TRAIL, 2))
        self.scene.addItem(self.trail_item)
        self.trail_points = deque(maxlen=TRAIL_MAX_POINTS)
        self._trail_frame = 0

        self.heading_item = QGraphicsLineItem(); self.heading_item.setZValue(95)
        self.heading_item.setPen(QPen(QColor(255, 255, 255, 200), 2))
        self.scene.addItem(self.heading_item)

        # sensors
        self.sensor_items = []
        self._ensure_sensor_items()

        # log aggregation
        self._last_log_key = None
        self._last_log_count = 0

        # stuck counter
        self.stuck_count = 0

        self.brain = None
        self.map_img = None

        self.setup_map(DEFAULT_MAP)

    # -----------------
    # Logging
    # -----------------
    def log(self, html: str):
        self.log_console.append(html)
        sb = self.log_console.verticalScrollBar(); sb.setValue(sb.maximum())

    def log_repeat(self, key: str, html: str):
        if key == self._last_log_key:
            self._last_log_count += 1
            msg = f"{html} <span style='color:#D8DEE9'>(x{self._last_log_count})</span>"
            cursor = self.log_console.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock, QTextCursor.MoveMode.MoveAnchor)
            cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
            cursor.insertHtml(msg)
            self.log_console.setTextCursor(cursor)
        else:
            self._last_log_key = key
            self._last_log_count = 1
            self.log(html)

    # -----------------
    # Sensors
    # -----------------
    def _ensure_sensor_items(self):
        if len(self.sensor_items) == len(SENSOR_ANGLES):
            return
        for it in self.sensor_items:
            try:
                if it.scene() == self.scene:
                    self.scene.removeItem(it)
            except Exception:
                pass
        self.sensor_items = []
        pen = QPen(SENSOR_PEN, 2)
        for _ in SENSOR_ANGLES:
            li = QGraphicsLineItem(); li.setZValue(90)
            li.setPen(pen)
            self.scene.addItem(li)
            self.sensor_items.append(li)

    # -----------------
    # Map load (Option 2: resize)
    # -----------------
    def setup_map(self, path: str):
        if not os.path.exists(path):
            self.log(f"<font color='{C_FAILURE.name()}'><b>Map not found:</b> {path}</font>")
            return

        qimg = QImage(path).convertToFormat(QImage.Format.Format_RGB32)
        mode = Qt.TransformationMode.SmoothTransformation if USE_SMOOTH_SCALING else Qt.TransformationMode.FastTransformation
        qimg = qimg.scaled(MAX_MAP_W, MAX_MAP_H, Qt.AspectRatioMode.KeepAspectRatio, mode)
        self.map_img = qimg

        if self.map_pixmap_item is not None:
            try:
                if self.map_pixmap_item.scene() == self.scene:
                    self.scene.removeItem(self.map_pixmap_item)
            except RuntimeError:
                pass
            self.map_pixmap_item = None

        self.map_pixmap_item = self.scene.addPixmap(QPixmap.fromImage(self.map_img))
        self.map_pixmap_item.setZValue(0)

        # reset overlays
        self.path_item.setPath(QPainterPath())
        self.trail_item.setPath(QPainterPath())
        self.trail_points.clear(); self._trail_frame = 0
        self.heading_item.setLine(0, 0, 0, 0)

        self._ensure_sensor_items()
        for s in self.sensor_items:
            s.setLine(0, 0, 0, 0)

        # reset counters
        self.stuck_count = 0
        self.lbl_stuck.setText('Stuck count: 0')

        # create brain
        self.brain = NavBrain(self.map_img)

        rect = self.map_pixmap_item.boundingRect()
        self.scene.setSceneRect(rect)
        self.view.resetTransform()
        self.view.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

        self.log(f"Map loaded (resized): {self.map_img.width()}x{self.map_img.height()} | grid={self.brain.planner.gw}x{self.brain.planner.gh} cell={CELL}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.map_pixmap_item is not None:
            self.view.fitInView(self.map_pixmap_item.boundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def load_map_dialog(self):
        f, _ = QFileDialog.getOpenFileName(self, 'Select Your Map', '', 'Images (*.png *.jpg *.jpeg)')
        if f:
            self.full_reset()
            self.setup_map(f)

    # -----------------
    # Click handling
    # -----------------
    def on_click(self, event):
        pt = self.view.mapToScene(event.pos())
        if self.setup_state == 0:
            self.brain.set_start_pos(pt)
            if self.car_item.scene() is None:
                self.scene.addItem(self.car_item)
            self.car_item.setPos(self.brain.car_pos)
            self.setup_state = 1
            self.trail_points.clear(); self.trail_item.setPath(QPainterPath())

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
                self.log(f"Target #{num} added at ({pt.x():.0f}, {pt.y():.0f})")
                self.lbl_status.setText(f"Targets: {num} | Right-click to finish")

            elif event.button() == Qt.MouseButton.RightButton:
                if len(self.brain.targets) > 0:
                    self.setup_state = 2
                    self.btn_run.setEnabled(True)
                    self.brain.reset_full()
                    self.update_path_visual(); self.update_visuals()

    # -----------------
    # Visuals
    # -----------------
    def update_path_visual(self):
        if not self.brain.route_cells:
            self.path_item.setPath(QPainterPath())
            return
        pts = [self.brain.planner.to_point(x, y) for x, y in self.brain.route_cells]
        if not pts:
            self.path_item.setPath(QPainterPath())
            return
        path = QPainterPath(); path.moveTo(pts[0])
        for p in pts[1:]:
            path.lineTo(p)
        self.path_item.setPath(path)

    def update_trail(self):
        self._trail_frame += 1
        if self._trail_frame % TRAIL_DOWNSAMPLE != 0:
            return
        p = QPointF(self.brain.car_pos.x(), self.brain.car_pos.y())
        if not self.trail_points:
            self.trail_points.append(p)
            tr = QPainterPath(); tr.moveTo(p)
            self.trail_item.setPath(tr)
            return
        last = self.trail_points[-1]
        if math.hypot(p.x() - last.x(), p.y() - last.y()) < 0.5:
            return
        self.trail_points.append(p)
        tr = self.trail_item.path()
        if tr.isEmpty():
            tr.moveTo(self.trail_points[0])
        tr.lineTo(p)
        self.trail_item.setPath(tr)

    def update_visuals(self):
        self.car_item.setPos(self.brain.car_pos)
        self.car_item.setRotation(self.brain.car_angle)

        rad = math.radians(self.brain.car_angle)
        hx = self.brain.car_pos.x() + math.cos(rad) * HEADING_LEN
        hy = self.brain.car_pos.y() + math.sin(rad) * HEADING_LEN
        self.heading_item.setLine(self.brain.car_pos.x(), self.brain.car_pos.y(), hx, hy)

        self._ensure_sensor_items()
        for i, rel in enumerate(SENSOR_ANGLES):
            ang = self.brain.car_angle + rel
            rad2 = math.radians(ang)
            sx = self.brain.car_pos.x(); sy = self.brain.car_pos.y()
            ex = sx + math.cos(rad2) * SENSOR_DIST
            ey = sy + math.sin(rad2) * SENSOR_DIST
            self.sensor_items[i].setLine(sx, sy, ex, ey)

        for i, t in enumerate(self.target_items):
            t.set_active(i == self.brain.current_target_idx)
        self.scene.update()

    # -----------------
    # Controls
    # -----------------
    def full_reset(self):
        self.sim_timer.stop()
        self.btn_run.setChecked(False)
        self.btn_run.setEnabled(False)
        self.btn_run.setText('▶ START (Space)')
        self.setup_state = 0

        self.stuck_count = 0
        self.lbl_stuck.setText('Stuck count: 0')

        if self.car_item.scene() == self.scene:
            self.scene.removeItem(self.car_item)
        for t in self.target_items:
            if t.scene() == self.scene:
                self.scene.removeItem(t)
        self.target_items = []

        self.path_item.setPath(QPainterPath())
        self.trail_item.setPath(QPainterPath())
        self.trail_points.clear(); self._trail_frame = 0
        self.heading_item.setLine(0, 0, 0, 0)

        if self.brain is not None:
            self.brain.targets = []
            self.brain.current_target_idx = 0
            self.brain.targets_reached = 0

    def toggle(self):
        if self.btn_run.isChecked():
            self.sim_timer.start(16)
            self.btn_run.setText('⏸ PAUSE')
        else:
            self.sim_timer.stop()
            self.btn_run.setText('▶ RESUME')

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space and self.setup_state == 2:
            self.btn_run.click()
        elif event.key() == Qt.Key.Key_R:
            self.full_reset()
        elif event.key() == Qt.Key.Key_Escape:
            if self.sim_timer.isActive():
                self.btn_run.setChecked(False)
                self.toggle()

    # -----------------
    # Main loop (safe)
    # -----------------
    def game_loop(self):
        if self.setup_state != 2:
            return

        try:
            _, reason = self.brain.step()
        except Exception as e:
            self._last_log_key = None
            self._last_log_count = 0
            self.log(f"<font color='{C_FAILURE.name()}'><b>❌ Runtime error:</b> {type(e).__name__}: {e}</font>")
            self.sim_timer.stop()
            self.btn_run.setChecked(False)
            self.btn_run.setText('⛔ ERROR')
            return

        self.update_trail()

        if reason == 'escaping':
            self.update_visuals()
            return

        if reason == 'escape_success':
            self._last_log_key = None
            self._last_log_count = 0
            self.log(f"<font color='{C_ACCENT.name()}'><b>✅ ESCAPED: replanning</b></font>")
            self.update_path_visual()

        if reason == 'no_path':
            self._last_log_key = None
            self._last_log_count = 0
            self.log(f"<font color='{C_FAILURE.name()}'><b>❌ NO PATH to target {self.brain.current_target_idx+1}. Stop.</b></font>")
            self.sim_timer.stop()
            self.btn_run.setChecked(False)
            self.btn_run.setEnabled(False)
            self.btn_run.setText('⛔ NO PATH')
            return

        if reason in ('off_road', 'out_of_bounds'):
            self.brain.record_crash(self.brain.last_crash_pos)
            self.log_repeat(reason, f"<font color='{C_FAILURE.name()}'><b>⚠️ {reason.upper()} (alternate route)</b></font>")
            self.brain.respawn_keep_progress()
            self.update_path_visual()

        if reason == 'stuck':
            self.stuck_count += 1
            self.lbl_stuck.setText(f'Stuck count: {self.stuck_count}')
            self.brain.record_crash(self.brain.last_crash_pos)
            self._last_log_key = None
            self._last_log_count = 0
            self.log(f"<font color='{C_FAILURE.name()}'><b>↺ STUCK #{self.stuck_count}: Restarting mission from start (Target #1)</b></font>")
            self.brain.restart_mission()
            self.update_path_visual()

        if reason == 'target_reached':
            self._last_log_key = None
            self._last_log_count = 0
            t = self.brain.targets_reached
            total = len(self.brain.targets)
            self.log(f"<font color='{C_ACCENT.name()}'><b>🎯 Target {t} reached! Moving to {t+1}/{total}</b></font>")
            self.update_path_visual()

        if reason == 'completed':
            self._last_log_key = None
            self._last_log_count = 0
            total = len(self.brain.targets)
            self.log(f"<font color='{C_SUCCESS.name()}'><b>✅ COMPLETED: {total}/{total} targets</b></font>")
            self.lbl_status.setText(f"COMPLETED: {total}/{total} targets")
            self.sim_timer.stop()
            self.btn_run.setChecked(False)
            self.btn_run.setEnabled(False)
            self.btn_run.setText('✅ COMPLETED')
            self.update_visuals()
            return

        if self.brain.route_cells is None or self.brain.replan_counter == 1:
            self.update_path_visual()

        self.update_visuals()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = NeuralNavApp()
    win.show()
    sys.exit(app.exec())
