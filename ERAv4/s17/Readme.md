"""
NeuralNav TD3 Triple Exploration Reward + Any-Order Targets + Big Targets + Trail Fix
=============================================================================================

What you requested
------------------
1) Keep exploration memory (visited grid) persistent across respawns ✅
2) Triple the reward for exploring new path ✅
3) If the agent reaches Target #2 or #3 first, count it (any-order targets) ✅

Key changes
---------------------
A) Exploration / novelty reward increased ~3x
   - NOVELTY_BONUS: 0.35 -> 1.05
   - NOVELTY_MAX  : 0.35 -> 1.05
   - Still decays with revisits to prevent infinite reward farming.

B) Any-order targets
   - Targets are treated as a SET.
   - If the car comes within GOAL_RADIUS of ANY unreached target:
       * mark that target reached
       * give a reward
       * choose the next active target as the NEAREST unreached
   - If all targets reached -> done True

C) Big targets for screenshots
   - Active radius: 16
   - Inactive radius: 12
   - Reach radius (GOAL_RADIUS): 24

D) Trail fix (no Qt deleted item crash)
   - Trail item is recreated safely after scene.clear()

Run
---
python citymap_TD3_v3_5.py

Files
-----
- car_topdown.png  (optional, top-down sprite; fallback rectangle if missing)
"""
