# 4.13 修改代码
联合效用函数 
$Utility = α * TargetValue + β * InfoGain - γ * Distance$


语义相似度+IG信息增益

vlfm/mapping/value_map.py       修改sort_waypoints函数 新增计算信息增益函数

距离惩罚

vlfm/policy/itm_policy.py       ITMPolicyV2类


打开地同同步策略

config/experiments/vlfm_objectnav_hm3d.yaml

在配置文件中寻找类似策略参数的位置



RL:
  POLICY:
    name: "ITMPolicyV2"
    sync_explored_areas: True  # 在这里设为 True





---调试信息打印 在 sort_waypoints 的末尾，return 之前加上这些打印代码 ---
        
        # 打印排名前 3 的边界点信息，用于 Debug
        print(f"\n--- 评估了 {len(waypoints)} 个候选边界点 ---")
        for idx in range(min(3, len(sorted_inds))):
            original_idx = sorted_inds[idx]
            v = reduced_values[original_idx]
            ig = self.compute_information_gain(waypoints[original_idx])
            ig_normalized = ig / (max_possible_gain + 1e-5)
            final_score = sorted_values[idx]
            
            print(f"Top {idx+1}: 坐标({waypoints[original_idx][0]:.2f}, {waypoints[original_idx][1]:.2f}) "
                  f"| 语义Value: {v:.3f} | 信息增益IG(归一化): {ig_normalized:.3f} "
                  f"| 融合得分: {final_score:.3f}")
        print("-----------------------------------")

        return sorted_frontiers, sorted_values


修改 vlfm/policy/itm_policy.py 中的 _get_best_frontier 方法，让它把加上了距离惩罚后的真实状态显示在画面上


    def _get_best_frontier(
            self,
            observations: Union[Dict[str, Tensor], "TensorDict"],
            frontiers: np.ndarray,
        ) -> Tuple[np.ndarray, float]:
            # ... (前面的代码保持不变) ...

            best_frontier = sorted_pts[best_frontier_idx]
            best_value = sorted_values[best_frontier_idx]
            self._acyclic_enforcer.add_state_action(robot_xy, best_frontier, top_two_values)
            self._last_value = best_value
            self._last_frontier = best_frontier
            
            # === 修改这里的 DEBUG_INFO ===
            # 计算一下当前选定点距离机器人的真实距离
            dist_to_best = np.linalg.norm(best_frontier - robot_xy)
            
            # 将我们修改后的综合得分和距离渲染到视频画面上
            debug_str = f" Score: {best_value:.3f} | Dist: {dist_to_best:.2f}m"
            os.environ["DEBUG_INFO"] += debug_str
            print(f"最终决定前往: 坐标({best_frontier[0]:.2f}, {best_frontier[1]:.2f}) - {debug_str}")

            return best_frontier, best_value


# 4.14 解决假阳性问题

vlfm/mapping/object_point_cloud_map.py中     _extract_object_cloud方法

        final_mask = cv2.erode(final_mask, None, iterations=self._erosion_size)  # type: ignore

        # Filter invalid or unreliable depth regions before 3D projection.
        depth_m = depth * (max_depth - min_depth) + min_depth
        min_valid_depth_m = min_depth + 0.05
        max_valid_depth_m = max_depth * 0.98
        depth_valid_mask = (depth > 0.0) & (depth_m > min_valid_depth_m) & (depth_m < max_valid_depth_m)
        final_mask = ((final_mask > 0) & depth_valid_mask).astype(np.uint8) * 255

        if np.count_nonzero(final_mask) == 0:
            return np.array([])
        #####################
        valid_depth = depth.copy()
        
_get_closest_point方法

def _get_closest_point(self, cloud: np.ndarray, curr_position: np.ndarray) -> np.ndarray:
        ndim = curr_position.shape[0]
        if len(cloud) == 0:
            if ndim == 2:
                return np.array([curr_position[0], curr_position[1], 0.5, 0.0], dtype=np.float32)
            return np.array([curr_position[0], curr_position[1], curr_position[2], 0.0], dtype=np.float32)

        
        if self.use_dbscan:
            # Use a robust anchor from the nearest subset to suppress single-point outliers.
            distances = np.linalg.norm(cloud[:, :ndim] - curr_position, axis=1)
            num_near = max(10, int(0.2 * len(cloud)))
            num_near = min(num_near, len(cloud))
            if num_near == len(cloud):
                near_cloud = cloud
            else:
                near_indices = np.argpartition(distances, num_near - 1)[:num_near]
                near_cloud = cloud[near_indices]

            anchor_xyz = np.median(near_cloud[:, :3], axis=0)
            anchor_within_range = 1.0 if np.any(near_cloud[:, -1] == 1) else float(near_cloud[0, -1])
            closest_point = np.concatenate((anchor_xyz, np.array([anchor_within_range], dtype=np.float32)))
        else:
        ########################################






