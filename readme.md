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


    def _extract_object_cloud(
        self,
        depth: np.ndarray,
        object_mask: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> np.ndarray:
        final_mask = object_mask * 255
        final_mask = cv2.erode(final_mask, None, iterations=self._erosion_size)
        
        # === 核心逻辑：深度补全 (Depth Inpainting) ===
        valid_depth = depth.copy()
        physical_depth = depth * (max_depth - min_depth) + min_depth
        
        # 1. 找到掩码内“真正有效的物体深度”（排除极近的地板噪点和无效黑洞）
        valid_pixel_mask = (final_mask > 0) & (depth > 0.0) & (physical_depth > 0.3)
        valid_pixels = depth[valid_pixel_mask]
        
        # 2. 计算物体的真实深度中位数（比如电视边框的深度）
        if len(valid_pixels) > 0:
            fill_value = np.median(valid_pixels)
        else:
            fill_value = 1.0  # 兜底
            
        # 3. 将电视的“黑色空洞”用它自己的真实边框深度填满！
        mask_holes = (final_mask > 0) & (depth == 0.0)
        valid_depth[mask_holes] = fill_value
        
        # 4. 彻底抛弃那些贴脸的地板溢出噪点
        depth_valid_mask = (physical_depth > 0.3) | mask_holes
        final_mask = (final_mask * depth_valid_mask).astype(np.uint8)
        
        # 将归一化深度还原为物理米制
        valid_depth = valid_depth * (max_depth - min_depth) + min_depth
        
        cloud = get_point_cloud(valid_depth, final_mask, fx, fy)
        cloud = get_random_subarray(cloud, 5000)
        if self.use_dbscan:
            cloud = open3d_dbscan_filtering(cloud)

        return cloud

    def _get_closest_point(self, cloud: np.ndarray, curr_position: np.ndarray) -> np.ndarray:
        ndim = curr_position.shape[0]
        
        # 空数组安全保护
        if len(cloud) == 0:
            if ndim == 2:
                return np.array([curr_position[0], curr_position[1], 0.5, 0.0], dtype=np.float32)
            return np.array([curr_position[0], curr_position[1], curr_position[2], 0.0], dtype=np.float32)

        # 计算所有点到机器人的距离
        dists = np.linalg.norm(cloud[:, :ndim] - curr_position, axis=1)
        
        if self.use_dbscan:
            # 此时云团已经被 DBSCAN 清理得非常干净了。
            # 为了防止最后残留一两个边缘点，我们跳过离得最近的 5% 的点，
            # 并挑选一个【真实存在的点】作为目标，绝不拼凑坐标！
            sorted_idx = np.argsort(dists)
            skip_idx = min(int(len(cloud) * 0.05), len(cloud) - 1)
            closest_point = cloud[sorted_idx[skip_idx]]
        else:
            closest_point = cloud[np.argmin(dists)]
            
        return closest_point






