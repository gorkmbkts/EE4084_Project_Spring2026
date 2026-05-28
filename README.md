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
1. Create and activate your Python environment.
2. Install requirements:
   - `pip install -r requirements.txt`
3. Run:
   - `python main.py`

`main.py` will use an existing CARLA server on `localhost:2000` if one is already running. If not, it can launch the local CARLA executable automatically.

## Startup map selection
Running `main.py` opens a pre-dashboard startup screen before vehicles, sensors, filters, route tools, or benchmark tools are initialized.

Startup flow:
- Checks whether CARLA RPC is already responding on `localhost:2000`.
- If CARLA is not running and `CARLA.auto_launch` is enabled, launches `CarlaUE4.exe`.
- Waits until CARLA RPC is responsive.
- Lists maps from `client.get_available_maps()` when available.
- Lets you select a map, load it safely, or press `U` to use the currently loaded map.
- Starts the dashboard only after `world`, `world_map`, and `blueprint_library` are valid.

Executable discovery:
- Uses `CARLA.executable_path` in `config/settings.py` when set.
- Otherwise searches:
  - `./CARLA_0.9.16/CarlaUE4.exe`
  - `../CARLA_0.9.16/CarlaUE4.exe`
  - `./CARLA_0.9.16/WindowsNoEditor/CarlaUE4.exe`
  - `../CARLA_0.9.16/WindowsNoEditor/CarlaUE4.exe`

Startup chooser controls:
- `Up` / `Down`: move selection
- Mouse wheel: scroll map list
- Left click: select map
- `Enter`: load selected map
- `U`: use currently loaded map
- `R`: refresh available maps
- `ESC`: quit safely

Saved routes are map-specific. Routes are stored with their CARLA map name, and the Route tab only exposes routes compatible with the active map. `TownXX` and `TownXX_Opt` variants are treated as compatible for saved route use where practical. To create a route for a new map, select that map in the startup chooser, enable test route mode in the Route tab, choose A/B endpoints on the 2D map, and save the route.

Known limitations:
- Packaged CARLA assets may not be visible in GitHub.
- `client.get_available_maps()` is the authoritative map list when CARLA is running.
- If map listing fails, the chooser still allows the current map and common fallback map names.
- Custom maps may have invalid GNSS georeference metadata; the dashboard will show GNSS projection as unavailable instead of crashing.
- If CARLA fails to launch, verify `CARLA.executable_path`, the local `CARLA_0.9.16` folder location, and port `2000` availability.

## Legacy Manual Startup
If `CARLA.auto_launch` is set to `False`, start CARLA server (`localhost:2000` by default) before running `main.py`.

1. Create and activate your Python environment.
2. Install requirements:
   - `pip install -r requirements.txt`
3. Run:
   - `python main.py`

## Notes
- This stage intentionally uses `vehicle.get_transform()` and `vehicle.get_velocity()` as the state source.
- The tracker and controller depend on the `EgoState` abstraction, so the state provider can later be replaced with sensor-based estimated pose.
- The project tries to import `carla` normally first.
- If that fails, it auto-adds CARLA wheels/eggs from:
  - `./CARLA_0.9.16/PythonAPI/carla/dist`
  - `../CARLA_0.9.16/PythonAPI/carla/dist`
- It also auto-adds CARLA's `PythonAPI/carla` path so `agents.navigation.global_route_planner` can be imported.
