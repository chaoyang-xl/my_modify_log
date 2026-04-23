# 4.13 修改代码
联合效用函数 
$Utility = α * TargetValue + β * InfoGain - γ * Distance$


语义相似度+IG信息增益

vlfm/mapping/value_map.py       修改sort_waypoints函数 新增计算信息增益函数

距离惩罚

vlfm/policy/itm_policy.py       ITMPolicyV2类


打开地同同步策略

vlfm/policy/itm_policy.py中

修改  sync_explored_areas: bool = True,


---调试信息打印 ---
        

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
            
            # 增加 debug 信息，显示 best_value 和 robot 到 best_frontier 的距离
            dist_to_best = np.linalg.norm(best_frontier - robot_xy)
            debug_info = f"Score: {best_value*100:.2f}%, Distance: {dist_to_best:.2f}. "
            os.environ["DEBUG_INFO"] += debug_info
            print(debug_info)

            return best_frontier, best_value




