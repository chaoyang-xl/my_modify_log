#修改了 value_map.py 中的 sort_waypoints 函数，新增了 compute_information_gain 函数，并在 sort_waypoints 中结合了原有的 value map 中的值和信息增益（information gain）来计算最终的分数。
def sort_waypoints(
        self, waypoints: np.ndarray, radius: float, reduce_fn: Optional[Callable] = None
    ) -> Tuple[np.ndarray, List[float]]:
        """Selects the best waypoint from the given list of waypoints.

        Args:
            waypoints (np.ndarray): An array of 2D waypoints to choose from.
            radius (float): The radius in meters to use for selecting the best waypoint.
            reduce_fn (Callable, optional): The function to use for reducing the values
                within the given radius. Defaults to np.max.

        Returns:
            Tuple[np.ndarray, List[float]]: A tuple of the sorted waypoints and
                their corresponding values.
        """
        radius_px = int(radius * self.pixels_per_meter)

        def get_value(point: np.ndarray) -> Union[float, Tuple[float, ...]]:
            x, y = point
            px = int(-x * self.pixels_per_meter) + self._episode_pixel_origin[0]
            py = int(-y * self.pixels_per_meter) + self._episode_pixel_origin[1]
            point_px = (self._value_map.shape[0] - px, py)
            all_values = [
                pixel_value_within_radius(self._value_map[..., c], point_px, radius_px)
                for c in range(self._value_channels)
            ]
            if len(all_values) == 1:
                return all_values[0]
            return tuple(all_values)

        #values = [get_value(point) for point in waypoints]
        #新增候选点的价值计算方式，结合原有的 value map 中的值和信息增益（information gain）
        raw_values = [get_value(point) for point in waypoints]

        if self._value_channels > 1:
            assert reduce_fn is not None, "Must provide a reduction function when using multiple value channels."
            reduced_values = reduce_fn(raw_values)
        else:
            reduced_values = raw_values
            
        alpha = 1.0
        beta = 0.5
        # 获取最大可能视野的像素数，用于归一化
        if len(self._confidence_masks) > 0:
            fov, max_depth = next(reversed(self._confidence_masks.keys()))
        else:
            fov = np.deg2rad(float(os.environ.get("INFO_GAIN_FOV_DEG", "79")))
            max_depth = float(os.environ.get("INFO_GAIN_MAX_DEPTH", "5.0"))

        blank_cone = self._get_blank_cone_mask(fov, max_depth)
        max_possible_gain = np.count_nonzero(blank_cone)  

        final_scores = []
        for i, point in enumerate(waypoints):
            v = reduced_values[i]
            ig = self.compute_information_gain(point)
            
            # 将 IG 归一化到 [0, 1] 区间
            ig_normalized = ig / (max_possible_gain + 1e-5) 
            
            score = alpha * v + beta * ig_normalized
            final_scores.append(score)
      


        # Use np.argsort to get the indices of the sorted values
        sorted_inds = np.argsort([-s for s in final_scores])  # 降序
        sorted_values = [final_scores[i] for i in sorted_inds]
        sorted_frontiers = np.array([waypoints[i] for i in sorted_inds])

        return sorted_frontiers, sorted_values
#新增函数
def compute_information_gain(self, point: np.ndarray) -> int:
        """Estimate exploration gain at a frontier point.

        The gain is the number of currently unknown pixels (where
        ``self._obstacle_map.explored_area == 0``) covered by the sensor FOV when
        centered at ``point``.
        """
        if self._obstacle_map is None:
            raise ValueError("compute_information_gain requires an obstacle_map.")

        # Reuse known sensor settings when possible; otherwise use common defaults.
        if len(self._confidence_masks) > 0:
            fov, max_depth = next(reversed(self._confidence_masks.keys()))
        else:
            fov = np.deg2rad(float(os.environ.get("INFO_GAIN_FOV_DEG", "79")))
            max_depth = float(os.environ.get("INFO_GAIN_MAX_DEPTH", "5.0"))

        x, y = point
        px = int(-x * self.pixels_per_meter) + self._episode_pixel_origin[0]
        py = int(-y * self.pixels_per_meter) + self._episode_pixel_origin[1]
        point_px = (self._value_map.shape[0] - px, py)

        explored_area = self._obstacle_map.explored_area.astype(bool)
        unknown_mask = np.logical_not(explored_area)

        # Frontier points do not include heading, so evaluate multiple headings and
        # return the best (optimistic) gain.
        blank_cone = self._get_blank_cone_mask(fov, max_depth).astype(np.uint8)
        test_yaws = np.linspace(0.0, 2 * np.pi, 16, endpoint=False)

        best_gain = 0
        for yaw in test_yaws:
            oriented_cone = rotate_image(blank_cone, -yaw)
            fov_mask = np.zeros_like(explored_area, dtype=np.uint8)
            fov_mask = place_img_in_img(fov_mask, oriented_cone, point_px[0], point_px[1])
            gain = int(np.count_nonzero(np.logical_and(fov_mask > 0, unknown_mask)))
            if gain > best_gain:
                best_gain = gain

        return best_gain


#4.13 修改 ITMPolicyV2 中的 _sort_frontiers_by_value 函数，
# 新增了距离惩罚项，并重新根据带有距离惩罚的分数进行排序。
class ITMPolicyV2(BaseITMPolicy):
    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
        self._pre_step(observations, masks)
        self._update_value_map()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        # 这里调用的 sort_waypoints 已经包含了 Value + IG
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 0.5)
        #引入距离惩罚
        gamma = 0.05 # 距离惩罚系数 (需要微调)
        robot_xy = self._observations_cache["robot_xy"]
        final_scores = []
        for i, frontier in enumerate(sorted_frontiers):
            # 计算机器人到 frontier 的欧氏距离
            dist = np.linalg.norm(frontier - robot_xy)
            
            # Value_IG_Score - gamma * Distance
            penalized_score = sorted_values[i] - gamma * dist
            final_scores.append(penalized_score)
        # 重新根据带有距离惩罚的分数排序
        resort_inds = np.argsort([-s for s in final_scores])
        final_frontiers = np.array([sorted_frontiers[i] for i in resort_inds])
        final_sorted_values = [final_scores[i] for i in resort_inds]
        
        return final_frontiers, final_sorted_values

