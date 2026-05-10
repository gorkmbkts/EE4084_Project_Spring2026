# CARLA Route Selection and Ground Truth Following Project

This project is a modular CARLA + pygame starter architecture for step-by-step development.

Current active behavior:
- Connect to CARLA.
- Spawn a vehicle with CARLA autopilot disabled.
- Attach an RGB camera.
- Show camera feed in a pygame window.
- Generate future waypoints from map + vehicle pose.
- Project waypoints into camera image space.
- Draw waypoint overlay inside pygame.
- Drive the vehicle manually with keyboard input.
- Show a top-down CARLA road network panel.
- Select A and B route endpoints by clicking the top-down panel.
- Snap A and B to CARLA drivable waypoints.
- Generate a global route using CARLA's `GlobalRoutePlanner` when available.
- Fall back to a topology-graph planner if the CARLA agents planner is unavailable.
- Teleport the ego vehicle to A after a valid A/B route is created.
- Track route progress from CARLA ground truth vehicle pose.
- Follow the route with a Pure Pursuit + proportional speed controller that slows down for large steering demand.

Not implemented yet (intentionally):
- GNSS/IMU/LiDAR localization.
- Kalman, EKF, or UKF localization logic.
- Sensor fusion.
- Lane detection.
- Traffic light handling.
- Obstacle avoidance.
- Behavior planning.

## Keyboard Controls
- `W`: throttle in manual mode
- `S`: brake in manual mode
- `A`: steer left in manual mode
- `D`: steer right in manual mode
- `Space`: hand brake in manual mode
- `M`: manual driving mode
- `P`: autonomous route-following mode when a valid route exists
- `T`: toggle top-down map selection panel
- `R`: reset A/B selections and clear route
- `C`: clear route only
- `G`: regenerate route from selected A/B
- `ESC`: quit

## Mouse Controls
- First left click inside the top-down panel selects A.
- Second left click selects B, generates a route, teleports the ego vehicle to A, and starts autonomous driving to B.
- A later left click starts a new A/B selection cycle.
- Mouse wheel zooms the top-down panel.
- Right or middle mouse drag pans the top-down panel.

## Run
1. Start CARLA server (`localhost:2000` by default).
2. Create and activate your Python environment.
3. Install requirements:
   - `pip install -r requirements.txt`
4. Run:
   - `python main.py`

## Notes
- This stage intentionally uses `vehicle.get_transform()` and `vehicle.get_velocity()` as the state source.
- The tracker and controller depend on the `EgoState` abstraction, so the state provider can later be replaced with sensor-based estimated pose.
- The project tries to import `carla` normally first.
- If that fails, it auto-adds CARLA wheels/eggs from:
  - `./CARLA_0.9.16/PythonAPI/carla/dist`
  - `../CARLA_0.9.16/PythonAPI/carla/dist`
- It also auto-adds CARLA's `PythonAPI/carla` path so `agents.navigation.global_route_planner` can be imported.
