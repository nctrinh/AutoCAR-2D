"""
RL environment for AutoCAR-2D.

DESIGN GOAL (thay thế hoàn toàn cho environment.py cũ)
--------------------------------------------------------
Thay vì để RL tự học điều hướng toàn cục (việc mà A* / RRT đã làm rất tốt
và rẻ hơn nhiều so với train một policy), RL trong thiết kế này chỉ đóng
vai trò "Local Path-Tracking Controller":

    A* / RRT  --Path(waypoints)-->  RL Agent  --(accel, steer)-->  Vehicle

Agent không nhận toạ độ goal tuyệt đối. Nó chỉ nhận:
    - Trạng thái động học của xe (vận tốc, góc lái hiện tại)
    - Sai số so với PATH đã có sẵn: cross-track error, heading error,
      độ cong phía trước (giống AdaptivePurePursuitController)
    - LIDAR để phản ứng với vật cản KHÔNG có trong path gốc (vật cản
      động, hoặc sai lệch giữa map lúc planning và thực tế)

=> RL học một hàm điều khiển "mềm dẻo hơn PID/Pure Pursuit" (không cần
   tay tune gain, tự thích nghi với nhiễu động lực học, tự né vật cản
   bất ngờ) nhưng vẫn tận dụng toàn bộ phần Planning đã có, và có thể so
   sánh trực tiếp (apple-to-apple) với PIDController / PurePursuitController
   trên cùng một Path.

Điểm khác biệt so với bản cũ (đã xoá):
    - Không còn "goal tuyệt đối trong observation" -> không cần agent tự
      học điều hướng toàn cục, giảm không gian khám phá, hội tụ nhanh hơn.
    - Reward chính là "tiến độ dọc theo path" (progress-along-path) thay
      vì "khoảng cách Euclid tới goal" -> tránh trường hợp agent đi xuyên
      tường tắt (cắt góc) mà A* đã cố tình tránh.
    - Có thể tái sử dụng cùng một Path cho cả 3 bộ điều khiển
      (PID / PurePursuit / RL) để benchmark công bằng.
    - Sửa lỗi mutable default argument (Vehicle() dùng chung instance).

FOG-OF-WAR / ONLINE REPLANNING (tuỳ chọn, bật bằng fog_of_war=True)
--------------------------------------------------------------------
Mặc định env giả định map hoàn toàn quan sát được (path A* tính 1 lần lúc
reset() trên full map). Đây là giả định KHÔNG đúng với kịch bản thực tế:
"ban đầu chỉ biết vùng LIDAR quét được, vừa đi vừa quét thêm, phát hiện
vật cản mới thì phải replan".

Khi fog_of_war=True, env mô phỏng đúng luồng đó:
    1. Có một `internal_map` riêng, BAN ĐẦU KHÔNG CÓ OBSTACLE nào (mù mờ).
    2. Mỗi step(): LIDAR quét trên map THẬT (self.map_env) như bình thường,
       nhưng bất kỳ tia nào chạm vật cản sẽ "ghi" obstacle đó vào
       internal_map (agent chỉ biết obstacle nó đã thực sự nhìn thấy).
    3. Nếu đoạn path phía trước (vài waypoint kế tiếp) bị chặn bởi obstacle
       vừa phát hiện -> gọi lại AStarPlanner trên internal_map để replan
       từ vị trí hiện tại tới goal cuối cùng, thay self.path bằng path mới.
    4. RL KHÔNG biết việc replan vừa xảy ra -- nó luôn chỉ nhìn
       cross-track-error/heading-error/curvature so với self.path TẠI THỜI
       ĐIỂM HIỆN TẠI. Observation/reward không có tín hiệu "path vừa đổi".
       Điều này giữ đúng phân chia vai trò: A* lo "biết gì, đi đường nào",
       RL chỉ lo "bám path hiện tại cho tốt" -- nhất quán với vai trò local
       tracker đã thiết kế ở phần đầu file.
    5. progress_dist được tính lại dựa trên path mới nhưng neo theo vị trí
       xe hiện tại, nên không có bước nhảy đột ngột trong reward khi replan.
"""

import math
import sys
from pathlib import Path as _Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.append(str(_Path(__file__).parent.parent.parent))

from src.core.vehicle import Vehicle, VehicleConfig
from src.core.map import Map2D, CircleObstacle
from src.planning.base_planner import Path as PlannedPath, PathPoint
from src.planning.a_star import AStarPlanner
from src.control import path_geometry


class PathTrackingEnv(gym.Env):
    """
    Gymnasium environment: RL agent bám theo một Path có sẵn (từ A*/RRT).

    Observation (tất cả đã normalize về khoảng xấp xỉ [-1, 1]):
        [0] velocity / max_velocity
        [1] steering_angle / max_steering_angle
        [2] cross_track_error / cte_norm        (lệch trái/phải so với path)
        [3] heading_error / pi                  (lệch hướng so với path)
        [4] curvature_ahead (đã scale)           (độ cong phía trước)
        [5] progress_ratio                       (đã đi được bao nhiêu % path)
        [6:] lidar readings (N tia, đã normalize [0,1])
    Total dims = 6 + num_lidar_rays

    Action:
        Continuous [acceleration, steering_angle], normalized [-1, 1],
        được scale theo VehicleConfig giống bản gốc.

    Reward:
        + progress dọc theo path (không phải khoảng cách thẳng tới goal)
        - cross-track error lớn
        - heading error lớn
        - va chạm (terminal, phạt nặng)
        + hoàn thành path (terminal, thưởng lớn)
        - time penalty nhẹ để khuyến khích đi nhanh
        - phạt lidar khi quá gần vật cản (an toàn, giữ nguyên tinh thần bản gốc)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        map_env: Optional[Map2D] = None,
        path: Optional[PlannedPath] = None,
        vehicle: Optional[Vehicle] = None,
        max_steps: int = 1000,
        num_lidar_rays: int = 16,
        lidar_range: float = 20.0,
        goal_threshold: float = 3.0,
        cte_fail_threshold: float = 8.0,
        render_mode: Optional[str] = None,
        fog_of_war: bool = False,
        replan_lookahead_wp: int = 5,
        replan_grid_resolution: float = 1.0,
        replan_obstacle_radius: float = 0.5,
        lidar_grid_resolution: float = 0.5,
    ):
        """
        Args:
            map_env: Map2D (tạo map + path mặc định nếu None)
            path: Path đã được A*/RRT sinh ra. Nếu None, tự chạy AStarPlanner
                  trên map_env.start -> map_env.goal khi reset().
            vehicle: instance Vehicle dùng riêng cho env này (KHÔNG share
                     giữa các env, tránh bug mutable-default của bản cũ).
            max_steps: số bước tối đa một episode.
            num_lidar_rays / lidar_range: cấu hình cảm biến LIDAR.
            goal_threshold: khoảng cách tới waypoint cuối để coi là "tới đích".
            cte_fail_threshold: nếu lệch khỏi path quá xa -> kết thúc sớm
                     (agent đang đi lạc hoàn toàn khỏi path được giao).
            render_mode: 'human' | 'rgb_array' | None
            fog_of_war: nếu True, path KHÔNG được tính trên full map. Agent
                     chỉ biết obstacle đã thực sự "nhìn thấy" qua LIDAR, và
                     env tự động replan bằng A* mỗi khi đoạn path phía
                     trước bị chặn bởi obstacle mới phát hiện. RL không
                     nhận biết việc replan này (xem docstring đầu file).
            replan_lookahead_wp: số waypoint phía trước current_wp_idx được
                     kiểm tra xem có bị obstacle mới chặn không.
            replan_grid_resolution: grid resolution dùng cho A* khi replan
                     (không cần trùng với lúc plan path gốc).
            replan_obstacle_radius: bán kính CircleObstacle được thêm vào
                     internal_map cho mỗi điểm chạm LIDAR mới phát hiện.
            lidar_grid_resolution: resolution của occupancy grid dùng để
                     raycast LIDAR (rasterize map_env MỘT LẦN lúc khởi tạo
                     thay vì loop qua từng obstacle bằng numpy mỗi tia mỗi
                     step -- đây là bottleneck chính khi train RL).
        """
        super().__init__()

        self.map_env = map_env or self._create_default_map()
        self.vehicle = vehicle if vehicle is not None else Vehicle()
        self.max_steps = max_steps
        self.num_lidar_rays = num_lidar_rays
        self.lidar_range = lidar_range
        self.goal_threshold = goal_threshold
        self.cte_fail_threshold = cte_fail_threshold
        self.render_mode = render_mode

        self.fog_of_war = fog_of_war
        self.replan_lookahead_wp = replan_lookahead_wp
        self.replan_grid_resolution = replan_grid_resolution
        self.replan_obstacle_radius = replan_obstacle_radius
        self.internal_map: Optional[Map2D] = None
        self._known_obstacle_points: list = []  # tránh add trùng obstacle
        self.replan_count = 0  # số lần đã replan trong episode hiện tại (để log/debug)

        self._external_path = path  # có thể None -> auto-plan khi reset
        self.path: Optional[PlannedPath] = None

        self.steps = 0
        self.total_reward = 0.0
        self.current_wp_idx = 0
        self.prev_progress_dist = 0.0

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        obs_dim = 6 + self.num_lidar_rays
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Chuẩn hoá cross-track error theo bề rộng map để obs không phụ
        # thuộc quá nhiều vào scale tuyệt đối của từng map trong curriculum.
        self._cte_norm = max(self.map_env.width, self.map_env.height) * 0.1

        # Rasterize map_env MỘT LẦN thành occupancy grid để raycast LIDAR
        # bằng tra bảng O(1) thay vì loop obstacles + numpy scalar ops mỗi
        # điểm mỗi tia mỗi step. map_env không đổi trong suốt vòng đời env
        # (kể cả khi fog_of_war=True -- chỉ internal_map thay đổi, LIDAR
        # luôn quét trên map THẬT), nên grid này chỉ cần build đúng 1 lần.
        self._lidar_grid_resolution = lidar_grid_resolution
        self._lidar_occ_grid = self._rasterize_occupancy(
            self.map_env, self._lidar_grid_resolution
        )

        self._lidar_cache = None
        self._lidar_cache_key = None

        self.renderer = None

    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _rasterize_occupancy(map_env: Map2D, resolution: float) -> np.ndarray:
        """Rasterize map_env's TRUE geometry (safety_margin=0, giống hệt
        semantics gốc của LIDAR raycasting) thành một bool grid, build một
        lần duy nhất. Đây là bước biến is_collision() (loop obstacles) thành
        một tra bảng O(1) cho mỗi điểm LIDAR ghé qua."""
        grid_w = int(np.ceil(map_env.width / resolution))
        grid_h = int(np.ceil(map_env.height / resolution))
        grid = np.zeros((grid_w, grid_h), dtype=bool)
        for gx in range(grid_w):
            wx = (gx + 0.5) * resolution
            for gy in range(grid_h):
                wy = (gy + 0.5) * resolution
                grid[gx, gy] = map_env.is_collision(wx, wy, 0)
        return grid

    def _create_default_map(self) -> Map2D:
        map_env = Map2D(width=100, height=100)
        for _ in range(5):
            x = np.random.uniform(20, 80)
            y = np.random.uniform(20, 80)
            radius = np.random.uniform(3, 8)
            map_env.add_obstacle(CircleObstacle(x, y, radius))
        map_env.set_start(10, 10)
        map_env.set_goal(90, 90)
        return map_env

    def set_path(self, path: PlannedPath) -> None:
        """Cho phép script training gán path cụ thể (ví dụ path đã smooth)."""
        self._external_path = path

    def _plan_path(self) -> PlannedPath:
        """Tự sinh path bằng A* nếu không có path được truyền vào."""
        if not (self.map_env.start and self.map_env.goal):
            raise ValueError(
                "map_env cần set_start()/set_goal() hoặc phải truyền path= "
                "trực tiếp cho PathTrackingEnv."
            )
        planner = AStarPlanner(self.map_env, grid_resolution=1.0)
        path = planner.plan(self.map_env.start, self.map_env.goal, info=False)
        if path is None:
            raise RuntimeError(
                "A* không tìm được đường đi cho map hiện tại; kiểm tra lại "
                "start/goal/obstacles trước khi train RL."
            )
        return path

    def _create_empty_internal_map(self) -> Map2D:
        """internal_map ban đầu KHÔNG có obstacle nào -- agent 'mù' toàn bộ
        map ngoại trừ vị trí start/goal, giống robot thật chưa quét gì."""
        internal = Map2D(
            width=self.map_env.width,
            height=self.map_env.height,
            safety_margin=self.map_env.safety_margin,
        )
        internal.set_start(*self.map_env.start)
        internal.set_goal(*self.map_env.goal)
        return internal

    def _plan_path_fog_of_war(self) -> PlannedPath:
        """
        Path ban đầu khi fog_of_war=True: chỉ dùng những gì internal_map
        biết (ban đầu là KHÔNG obstacle nào) -- vì vậy path đầu tiên thường
        là đường thẳng-ish tới goal, sẽ được _maybe_replan() sửa dần khi xe
        di chuyển và LIDAR phát hiện obstacle thật.
        """
        planner = AStarPlanner(self.internal_map, grid_resolution=self.replan_grid_resolution)
        path = planner.plan(self.map_env.start, self.map_env.goal, info=False)
        if path is None:
            raise RuntimeError(
                "A* (fog-of-war, internal_map trống) không tìm được đường đi "
                "-- kiểm tra lại start/goal có nằm trong map hợp lệ không."
            )
        return path

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self.replan_count = 0
        self._known_obstacle_points = []

        if self.fog_of_war and self._external_path is None:
            self.internal_map = self._create_empty_internal_map()
            self.path = self._plan_path_fog_of_war()
        else:
            self.internal_map = None
            self.path = self._external_path or self._plan_path()

        self._path_cum_dist = self._compute_cumulative_distances(self.path)

        start = self.path.points[0]
        theta0 = start.theta
        if theta0 is None:
            # heading ban đầu hướng theo đoạn path đầu tiên
            if len(self.path.points) > 1:
                nxt = self.path.points[1]
                theta0 = np.arctan2(nxt.y - start.y, nxt.x - start.x)
            else:
                theta0 = 0.0

        if options and options.get("random_heading"):
            theta0 = np.random.uniform(-np.pi, np.pi)

        self.vehicle.reset(x=start.x, y=start.y, theta=theta0)

        self.steps = 0
        self.total_reward = 0.0
        self.current_wp_idx = 0
        self.prev_progress_dist = 0.0

        self._lidar_cache = None
        self._lidar_cache_key = None

        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.clip(action, -1.0, 1.0)
        acceleration = action[0] * self.vehicle.config.max_acceleration
        steering = action[1] * self.vehicle.config.max_steering_angle

        self.vehicle.update(acceleration, steering)

        lidar = self._get_lidar_readings()

        if self.fog_of_war:
            self._update_internal_map_from_lidar(lidar)
            self._maybe_replan()

        cte, heading_err, curvature, progress_dist, progress_ratio = (
            self._compute_path_metrics()
        )

        terminated = False
        truncated = False
        is_collision = False
        is_goal_reached = False

        for corner in self.vehicle.get_corners():
            if self.map_env.is_collision(corner[0], corner[1], 0):
                is_collision = True
                terminated = True
                break

        last_wp = self.path.points[-1]
        if self.vehicle.distance_to(last_wp.x, last_wp.y) < self.goal_threshold:
            is_goal_reached = True
            terminated = True

        if abs(cte) > self.cte_fail_threshold:
            # Agent đi lạc quá xa khỏi path được giao -> dừng sớm, phạt vừa
            # phải (không phạt nặng như va chạm, vì đây có thể là lúc agent
            # đang né một vật cản không có trong path gốc).
            truncated = True

        reward = self._calculate_reward(
            action=action,
            lidar=lidar,
            cte=cte,
            heading_err=heading_err,
            progress_dist=progress_dist,
            is_collision=is_collision,
            is_goal_reached=is_goal_reached,
        )

        self.steps += 1
        if self.steps >= self.max_steps:
            truncated = True

        self.total_reward += reward
        self.prev_progress_dist = progress_dist

        observation = self._build_obs_vector(
            cte, heading_err, curvature, progress_ratio, lidar
        )
        info = self._get_info()
        info.update(
            {
                "lidar_data": lidar,
                "cross_track_error": cte,
                "heading_error": heading_err,
                "is_collision": is_collision,
                "is_goal_reached": is_goal_reached,
            }
        )
        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Path-tracking geometry (dùng chung logic với AdaptivePurePursuit)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_cumulative_distances(path: PlannedPath) -> np.ndarray:
        pts = path.to_array()
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        return np.concatenate(([0.0], np.cumsum(seg)))

    def _compute_path_metrics(self) -> Tuple[float, float, float, float, float]:
        """
        Trả về (dùng chung geometry với AdaptivePurePursuitController, xem
        src/control/path_geometry.py -- đảm bảo so sánh RL vs classical
        apples-to-apples):
            cte: cross-track error có dấu (âm = lệch phải, dương = lệch trái)
            heading_err: sai lệch hướng so với hướng đoạn path hiện tại [-pi, pi]
            curvature: độ cong ước lượng phía trước (giống AdaptivePurePursuit)
            progress_dist: khoảng cách dọc theo path đã đi được (m)
            progress_ratio: progress_dist / tổng chiều dài path, trong [0, 1]
        """
        pos = self.vehicle.get_position()
        theta = self.vehicle.state.theta
        pts = self.path.to_array()

        self.current_wp_idx = path_geometry.closest_segment_index(
            pts, pos, search_start=self.current_wp_idx
        )
        cte, heading_err, t, seg_len = path_geometry.path_tracking_error(
            pts, pos, theta, self.current_wp_idx
        )
        curvature = path_geometry.estimate_curvature(self.current_wp_idx, pts)

        i = min(self.current_wp_idx, len(pts) - 2)
        progress_dist = float(self._path_cum_dist[i] + t * seg_len)
        total_len = self._path_cum_dist[-1] if self._path_cum_dist[-1] > 0 else 1.0
        progress_ratio = float(np.clip(progress_dist / total_len, 0.0, 1.0))

        return cte, heading_err, curvature, progress_dist, progress_ratio

    # ------------------------------------------------------------------ #
    # Fog-of-war: cập nhật internal_map từ LIDAR + replan khi path bị chặn
    # ------------------------------------------------------------------ #
    def _update_internal_map_from_lidar(self, lidar: np.ndarray) -> None:
        """Với mỗi tia LIDAR chạm vật cản (reading < 1.0), 'ghi nhớ' điểm đó
        vào internal_map bằng một CircleObstacle nhỏ. Đây chính là bước
        biến 'điểm quét được' thành 'ô nhớ trong map nội bộ', y hệt cách
        FogOfWarDriver bản gốc của bạn từng làm ở tầng evaluate script."""
        pos = self.vehicle.get_position()
        theta = self.vehicle.state.theta
        ray_angles = theta + np.linspace(0, 2 * np.pi, self.num_lidar_rays, endpoint=False)

        for i, angle in enumerate(ray_angles):
            if lidar[i] >= 1.0:
                continue  # tia này không chạm gì trong tầm quét
            dist = float(lidar[i]) * self.lidar_range
            ox = pos[0] + dist * np.cos(angle)
            oy = pos[1] + dist * np.sin(angle)

            if self._is_newly_discovered(ox, oy):
                self.internal_map.add_obstacle(
                    CircleObstacle(ox, oy, radius=self.replan_obstacle_radius)
                )
                self._known_obstacle_points.append((ox, oy))

    def _is_newly_discovered(self, x: float, y: float, merge_radius: float = 1.5) -> bool:
        """Tránh spam hàng trăm CircleObstacle chồng nhau cho cùng một vật
        cản khi xe đứng gần nó nhiều step liên tiếp."""
        for (kx, ky) in self._known_obstacle_points:
            if np.hypot(kx - x, ky - y) < merge_radius:
                return False
        return True

    def _maybe_replan(self) -> None:
        """Kiểm tra xem vài waypoint phía trước có bị obstacle mới phát
        hiện chặn không; nếu có, replan bằng A* trên internal_map (map mà
        agent 'biết' tính tới thời điểm hiện tại) từ vị trí xe -> goal cuối."""
        if self.path is None or len(self.path.points) == 0:
            return

        end_idx = min(self.current_wp_idx + self.replan_lookahead_wp, len(self.path.points))
        blocked = False
        for wp in self.path.points[self.current_wp_idx:end_idx]:
            if self.internal_map.is_collision(wp.x, wp.y):
                blocked = True
                break

        if not blocked:
            return

        pos = self.vehicle.get_position()
        goal = self.map_env.goal
        planner = AStarPlanner(self.internal_map, grid_resolution=self.replan_grid_resolution)
        new_path = planner.plan(pos, goal, info=False)

        if new_path is not None and len(new_path.points) >= 2:
            self.path = new_path
            self._path_cum_dist = self._compute_cumulative_distances(self.path)
            self.current_wp_idx = 0
            self.replan_count += 1
        # Nếu A* không tìm được đường trên internal_map hiện tại (thông tin
        # còn thiếu), giữ nguyên path cũ và thử lại ở step kế tiếp khi biết
        # thêm thông tin -- tránh crash episode chỉ vì 1 lần replan thất bại.

    # ------------------------------------------------------------------ #
    # Observation / reward
    # ------------------------------------------------------------------ #
    def _get_observation(self) -> np.ndarray:
        lidar = self._get_lidar_readings()
        cte, heading_err, curvature, _, progress_ratio = self._compute_path_metrics()
        return self._build_obs_vector(cte, heading_err, curvature, progress_ratio, lidar)

    def _build_obs_vector(
        self,
        cte: float,
        heading_err: float,
        curvature: float,
        progress_ratio: float,
        lidar: np.ndarray,
    ) -> np.ndarray:
        v = self.vehicle.state.velocity
        steering = self.vehicle.state.steering_angle

        base = np.array(
            [
                v / self.vehicle.config.max_velocity,
                steering / self.vehicle.config.max_steering_angle,
                np.clip(cte / self._cte_norm, -3.0, 3.0),
                heading_err / np.pi,
                np.clip(curvature * 10.0, 0.0, 1.0),  # scale để rơi vào ~[0,1]
                progress_ratio,
            ],
            dtype=np.float32,
        )
        return np.concatenate([base, lidar]).astype(np.float32)

    def _get_lidar_readings(self) -> np.ndarray:
        """
        Raycast bằng tra bảng trên occupancy grid đã rasterize sẵn
        (self._lidar_occ_grid) thay vì loop obstacles + numpy scalar ops
        mỗi điểm mỗi tia mỗi step -- đây từng là bottleneck chính khi train
        (num_lidar_rays x range_steps x num_obstacles lệnh numpy MỖI step).
        Dùng math.* (không phải np.*) cho scalar trig vì numpy có overhead
        dispatch đáng kể trên từng lệnh scalar riêng lẻ.
        """
        pos = self.vehicle.get_position()
        theta = self.vehicle.state.theta
        cache_key = (round(pos[0], 3), round(pos[1], 3), round(theta, 3))
        if self._lidar_cache is not None and self._lidar_cache_key == cache_key:
            return self._lidar_cache

        readings = np.ones(self.num_lidar_rays, dtype=np.float32)
        grid = self._lidar_occ_grid
        grid_w, grid_h = grid.shape
        res = self._lidar_grid_resolution
        map_w, map_h = self.map_env.width, self.map_env.height
        step_size = 1.0
        max_steps = int(self.lidar_range / step_size)
        angle_increment = 2.0 * math.pi / self.num_lidar_rays

        for i in range(self.num_lidar_rays):
            angle = theta + i * angle_increment
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            for step in range(1, max_steps + 1):
                dist = step * step_size
                rx = pos[0] + dist * cos_a
                ry = pos[1] + dist * sin_a
                # Bounds check on world coords first (matches Map2D.is_collision
                # semantics exactly) -- avoids int() truncating a negative
                # out-of-bounds coordinate into a spuriously "valid" grid index 0.
                if not (0.0 <= rx <= map_w and 0.0 <= ry <= map_h):
                    readings[i] = dist / self.lidar_range
                    break
                gx = min(int(rx / res), grid_w - 1)
                gy = min(int(ry / res), grid_h - 1)
                if grid[gx, gy]:
                    readings[i] = dist / self.lidar_range
                    break

        self._lidar_cache = readings
        self._lidar_cache_key = cache_key
        return readings

    def _calculate_reward(
        self,
        action: np.ndarray,
        lidar: np.ndarray,
        cte: float,
        heading_err: float,
        progress_dist: float,
        is_collision: bool,
        is_goal_reached: bool,
    ) -> float:
        if is_collision:
            return -50.0
        if is_goal_reached:
            return 50.0

        reward = 0.0

        # 1. Progress dọc theo PATH (không phải khoảng cách thẳng tới goal)
        #    -> agent bị "ép" đi theo hình dạng path mà A*/RRT đã tính,
        #    thay vì học cách cắt góc xuyên vật cản.
        progress = progress_dist - self.prev_progress_dist
        reward += progress * 2.0

        # 2. Phạt lệch khỏi path (cross-track error) và lệch hướng.
        #    Khi lidar phát hiện vật cản gần, giảm phạt cross-track error:
        #    nếu không, agent bị phạt CTE ngay lúc nó cần rời path để né vật
        #    cản -- hai tín hiệu reward triệt tiêu lẫn nhau đúng lúc quan
        #    trọng nhất. heading_error vẫn phạt bình thường (hướng đi lệch
        #    không phải là hành vi né vật cản hợp lệ).
        min_lidar = float(np.min(lidar))
        avoidance_urgency = np.clip((0.4 - min_lidar) / 0.4, 0.0, 1.0)
        reward -= 0.15 * abs(cte) * (1.0 - avoidance_urgency)
        reward -= 0.3 * abs(heading_err)

        # 3. An toàn: phạt khi lidar phát hiện vật cản gần (giữ tinh thần bản gốc)
        if min_lidar < 0.4:
            reward -= 10.0 * (0.4 - min_lidar)

        # 4. Time penalty nhẹ, khuyến khích đi nhanh
        reward -= 0.5

        # 5. Khuyến khích không đứng yên (tránh chính sách "đứng im cho an toàn")
        if self.vehicle.state.velocity < 0.1:
            reward -= 1.0

        # 6. Phạt nhẹ hành động giật cục (điều khiển mượt hơn)
        reward -= 0.02 * float(np.sum(np.square(action)))

        return reward

    def _get_info(self) -> Dict[str, Any]:
        pos = self.vehicle.get_position()
        last_wp = self.path.points[-1] if self.path else None
        info = {
            "steps": self.steps,
            "total_reward": self.total_reward,
            "position": pos,
            "current_waypoint_idx": self.current_wp_idx,
            "path_length": self.path.length if self.path else 0.0,
            "replan_count": self.replan_count,
        }
        if last_wp is not None:
            info["distance_to_goal"] = self.vehicle.distance_to(last_wp.x, last_wp.y)
        return info

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def render(self):
        if self.render_mode is None:
            return None
        if self.render_mode == "human":
            return self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()

    def _render_human(self):
        if self.renderer is None:
            from src.simulation.renderer import Renderer

            self.renderer = Renderer(
                screen_width=800,
                screen_height=800,
                world_width=self.map_env.width,
                world_height=self.map_env.height,
                caption="RL Path Tracking Training",
            )

        import pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        self.renderer.clear()
        self.renderer.draw_grid()
        self.renderer.draw_map(self.map_env)

        if self.path is not None and len(self.path.points) > 1:
            screen_points = [
                self.renderer.world_to_screen(p.x, p.y) for p in self.path.points
            ]
            pygame.draw.lines(self.renderer.screen, (255, 255, 0), False, screen_points, 2)

        if self.map_env.start and self.map_env.goal:
            self.renderer.draw_start_goal(self.map_env.start, self.map_env.goal)

        self.renderer.draw_vehicle(self.vehicle)
        self.renderer.update()

    def _render_rgb_array(self):
        if self.renderer is None:
            from src.simulation.renderer import Renderer

            self.renderer = Renderer(
                screen_width=400,
                screen_height=400,
                world_width=self.map_env.width,
                world_height=self.map_env.height,
            )

        self.renderer.clear()
        self.renderer.draw_map(self.map_env)
        self.renderer.draw_vehicle(self.vehicle)

        import pygame

        img = pygame.surfarray.array3d(self.renderer.screen)
        return np.transpose(img, (1, 0, 2))

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None