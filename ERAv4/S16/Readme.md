# City Map Navigation Assignment — README

This project is a **PyQt6-based 2D navigation simulator** that drives a small “car” icon along **road pixels** on a map image using **A\*** planning + a **pure-pursuit** follower.

It supports **multiple targets**, a **restart-on-stuck** policy (with one local escape attempt), and a UI that lets you load different maps.

---

## 1) What you get

- **Click-to-set start** (car position)
- **Click-to-add targets** (1…N)
- **A\*** planning on a grid
- **Road-following** using a lookahead point (pure pursuit)
- **Crash avoidance / re-route** using a penalty heatmap
- **Stuck detection** + **escape then restart** from Target #1
- **Stuck counter** (how many restarts due to stuck)
- **Sensor rays** (red) for visual debugging
- **Option 2 map normalization**: any loaded image is resized to fit inside a max width/height

---

## 2) Requirements

### Software
- **Python 3.10+** recommended
- OS: macOS / Linux / Windows

### Python packages
Install dependencies:

```bash
pip install pyqt6 numpy
```

> If you plan to analyze images or extend the project, you may also install:
> `pip install pillow scipy`

---

## 3) Run the simulator

Example (v10.1 hybrid build):

```bash
python citymap_assignment_fixed_v10_1_hybrid_both_maps.py
```

If your code file has a different name, run that file instead.

---

## 4) How to use (Controls)

### Step-by-step
1. **Left-click** on the map to place the **CAR (start position)**.
2. **Left-click** to add **TARGETS** (Target #1, #2, …).
3. **Right-click** when you are done adding targets.
4. Press **START** button (or **Space**) to begin.

### Keyboard shortcuts
- **Space** → Start / Pause
- **R** → Reset all (clear car + targets)
- **Esc** → Pause

### Visual legend
- **Cyan path** → planned A\* route
- **Red trail** → traveled path
- **White heading line** → car heading direction
- **Red rays** → sensor visualization

---

## 5) Map formats supported

### A) Road-only / binary map (recommended)
- White roads (near #FFFFFF)
- Dark background
- Works best and fastest

### B) Colored city map with white roads
- Roads are white/light but the rest of the map may be colored (blue/orange etc.)
- Works in the **hybrid** build that uses HSV+RGB road detection

> Tip: If a map fails, it is usually because the “road detector” cannot reliably distinguish roads from background.

---

## 6) Option 2 Map Normalization (auto resize)

To keep UI consistent, the program can resize any loaded map to:

```python
MAX_MAP_W = 1200
MAX_MAP_H = 800
```

- Aspect ratio is preserved.
- This makes different maps “fit” inside the same view and keeps simulation scale stable.

If you want more detail (bigger internal map), increase these values.

---

## 7) Parameters you can tune

### Road detection
For road-only binary maps you can use strict thresholds.
For colored maps you typically need hybrid thresholds.

Typical hybrid knobs:

```python
ROAD_V_THR   = 0.78  # brightness (V in HSV)
ROAD_S_THR   = 0.28  # saturation (S in HSV)
ROAD_MIN_THR = 0.85  # min(R,G,B) fallback
```

### Car road-check footprint
To avoid false OFF_ROAD on thick or anti-aliased roads:

```python
ROAD_FOOTPRINT_R = 3  # sample radius in pixels
ROAD_MIN_HITS    = 5  # min on-road hits out of 9 samples
```

If roads are very thin: try `ROAD_FOOTPRINT_R=2`.
If roads are thick: try `ROAD_FOOTPRINT_R=4`.

### A\* planner grid resolution

```python
CELL = 6
```

- Lower CELL (e.g., 4–5) = more accurate, slower.
- Higher CELL (e.g., 8) = faster, less accurate.

### Replanning frequency

```python
REPLAN_EVERY_N = 12
```

Lower value = replans more often (more robust), higher CPU.

### Stuck handling

```python
STUCK_SECONDS = 20.0
STUCK_ANCHOR_RADIUS = 30.0

ESCAPE_SECONDS = 4.0
ESCAPE_TURN    = 30.0
ESCAPE_SPEED   = 0.95
```

- Reduce `STUCK_SECONDS` to restart sooner.
- Increase `STUCK_ANCHOR_RADIUS` if position jitter prevents stuck detection.

---

## 8) Stuck logic (what happens)

1. If the car is considered **stuck**, it **tries one escape attempt** (turn/move).
2. If it **escapes**, it replans and continues.
3. If escape **fails**, it triggers **STUCK** and the simulator:
   - increments **Stuck count**
   - calls **restart_mission()**
   - resets to **start** and **Target #1**

This matches the requirement: *“If it didn’t reach the target and it is stuck, restart from the first place.”*

---

## 9) Troubleshooting

### A) “NO PATH” appears
- Your road detector likely marked the road as non-walkable.
- Try:
  - lowering thresholds (`ROAD_V_THR`, `ROAD_MIN_THR`)
  - reducing `CELL`
  - using a road-only binary map

### B) Car spins or oscillates near corners
- Increase `LOOKAHEAD_DIST` slightly.
- Reduce `MAX_TURN_PER_TICK`.

### C) Too many OFF_ROAD events
- Increase `ROAD_MIN_HITS` (more strict) **OR** adjust thresholds.
- Increase `ROAD_FOOTPRINT_R` for thick roads.

### D) Map looks zoomed/cropped
- Use “fit-to-view” behavior (if enabled) or increase `MAX_MAP_W/MAX_MAP_H`.

---

## 10) Assignment checklist (suggested)

- [ ] Load a map image and generate a walkable grid.
- [ ] Place start and multiple targets.
- [ ] Plan A\* route from start to target.
- [ ] Follow the route with smooth turning.
- [ ] Detect off-road/out-of-bounds and replan.
- [ ] Detect stuck loops and reset mission.
- [ ] Maintain progress across multiple targets.

---

## 11) File structure (typical)

```
project/
  citymap_assignment_fixed_v10_1_hybrid_both_maps.py
  City_Image_1_converted.png
  City_Map.png
  README.md
```

---

## 12) Notes

- For the most reliable behavior, use **road-only binary maps**.
- Colored maps work best with hybrid detection and multi-sample planner grid.

---

If you want, tell me which exact script name you are currently using (v10.0, v10.1, etc.), and I can tailor this README to match it exactly (including the precise parameter names and defaults).
