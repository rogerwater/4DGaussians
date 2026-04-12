import numpy as np
from mpc.constraint_utils import project_joint_angles_torch, check_angular_velocity_constraint
import time
from hydra.utils import instantiate

from mpc.optimizer import Optimizer
from mpc.utils import ObservationList, write_moviepy_gif


class CEMOptimizer(Optimizer):
    # Based on https://github.com/google-research/pddm/blob/06b88cdbafa7fc35451b3feb3e01aa6726113f72/pddm/policies/cem.py

    def __init__(
        self,
        sampler,
        model,
        objective,
        a_dim,
        horizon,
        num_samples,
        elites_frac,
        opt_iters,
        log_every=1,
        init_std=0.5,
        init_mean=None,
        alpha=0,
        verbose=False,
        round_gripper_action=False,
        save_all_iterations=False,  # 默认False：只保存最后一次迭代
        include_score_in_gif=False,  
        action_delta_penalty_weight=0.5,  # 超限动作惩罚权重
        action_delta_max_deg=30.0,  # 关节最大角度变化（度）
    ):
        self.obj_fn = objective
        self.sampler = sampler
        self.model = model
        self.horizon = horizon
        self.a_dim = a_dim
        self.num_samples = num_samples
        self.elites_frac = elites_frac
        self.opt_iters = opt_iters
        self.log_every = log_every
        self.alpha = alpha  # Controls mean/variance update rate
        self.init_std = np.array(init_std)
        self.init_mean = np.array(init_mean)
        self.lower_bound, self.upper_bound = -1, 1
        self.verbose = verbose
        self.round_gripper_action = round_gripper_action
        self.save_all_iterations = save_all_iterations  # 保存所有迭代还是只保存最后一次
        self.include_score_in_gif = include_score_in_gif  # GIF中是否包含score文本
        self.action_delta_penalty_weight = action_delta_penalty_weight
        self.action_delta_max_deg = action_delta_max_deg
        
        # Store latest optimization results for external access
        self.last_best_reward = None
        self.last_mean_reward = None
        self.last_rewards_history = []  # List of (best, mean, std) per iteration
    
    def _print_loss_breakdown(self, predictions, goal, rewards, iter_num, obs_history=None):
        """打印loss的各个组成部分详情"""
        print(f"    📊 Loss Breakdown (Iter {iter_num+1}):")
        
        # 获取CombinedObjective的各个子目标
        if hasattr(self.obj_fn, 'objectives'):
            for name, objective in self.obj_fn.objectives.items():
                # 转换goal到GPU（如果需要）
                import torch
                goal_dict = goal.data_dict if hasattr(goal, 'data_dict') else goal
                goal_gpu = {}
                for key, value in goal_dict.items():
                    if isinstance(value, np.ndarray):
                        goal_gpu[key] = torch.from_numpy(value).to(self.model.device)
                    elif isinstance(value, torch.Tensor):
                        goal_gpu[key] = value.to(self.model.device)
                    else:
                        goal_gpu[key] = value
                
                # 添加prev_rgb（如果需要）
                if "prev_rgb" not in goal_gpu and obs_history is not None:
                    prev_rgb = np.array(obs_history[self.model.base_prediction_modality])[-1]
                    goal_gpu["prev_rgb"] = torch.from_numpy(prev_rgb[None]).to(self.model.device)
                
                # 计算单个objective的reward
                try:
                    single_reward = objective(predictions, goal_gpu)
                    if isinstance(single_reward, torch.Tensor):
                        single_reward = single_reward.cpu().numpy()
                    
                    reward_mean = np.mean(single_reward)
                    reward_std = np.std(single_reward)
                    reward_max = np.max(single_reward)
                    reward_min = np.min(single_reward)
                    
                    print(f"      - {name:25s}: mean={reward_mean:8.4f}, std={reward_std:8.4f}, max={reward_max:8.4f}, min={reward_min:8.4f}")
                    
                    # 🔍 对ActionRegularizationObjective进行额外诊断
                    if name == 'action_regularization' and 'actions' in predictions:
                        try:
                            import torch
                            actions = predictions['actions']
                            if isinstance(actions, np.ndarray):
                                actions = torch.from_numpy(actions).float()
                            B, T, A = actions.shape
                            
                            # 计算角度变化统计
                            if T > 1 and A >= 12:
                                angle_deltas_all = []
                                for i in range(6):
                                    sin_idx = 2 * i
                                    cos_idx = 2 * i + 1
                                    prev_angle = torch.atan2(actions[:, :-1, sin_idx], actions[:, :-1, cos_idx])
                                    curr_angle = torch.atan2(actions[:, 1:, sin_idx], actions[:, 1:, cos_idx])
                                    angle_delta = curr_angle - prev_angle
                                    angle_delta = torch.atan2(torch.sin(angle_delta), torch.cos(angle_delta))
                                    angle_deltas_all.append(torch.abs(angle_delta))
                                
                                if angle_deltas_all:
                                    angle_deltas_tensor = torch.stack(angle_deltas_all, dim=-1)  # (B, T-1, 6)
                                    angle_deltas_flat = angle_deltas_tensor.reshape(-1)
                                    
                                    total_elements = angle_deltas_flat.numel()
                                    if total_elements > 0:
                                        print(f"        💡 Angle delta diagnostics:")
                                        print(f"           - Mean: {angle_deltas_flat.mean():.4f} rad ({torch.rad2deg(angle_deltas_flat.mean()):.2f}°)")
                                        print(f"           - Max:  {angle_deltas_flat.max():.4f} rad ({torch.rad2deg(angle_deltas_flat.max()):.2f}°)")
                                        print(f"           - Threshold: {objective.max_delta:.4f} rad ({np.rad2deg(objective.max_delta):.2f}°)")
                                        exceeds = (angle_deltas_flat > objective.max_delta).sum().item()
                                        percent = 100.0 * exceeds / total_elements if total_elements > 0 else 0.0
                                        print(f"           - Exceeds threshold: {exceeds}/{total_elements} ({percent:.1f}%)")
                        except Exception as diag_error:
                            print(f"        ⚠ Angle delta diagnostics failed: {str(diag_error)}")
                    
                except Exception as e:
                    print(f"      - {name:25s}: Error computing - {str(e)[:50]}")
        
        # 打印总reward统计
        if isinstance(rewards, torch.Tensor):
            rewards_np = rewards.cpu().numpy()
        else:
            rewards_np = rewards
        print(f"      - {'TOTAL':25s}: mean={np.mean(rewards_np):8.4f}, std={np.std(rewards_np):8.4f}, max={np.max(rewards_np):8.4f}, min={np.min(rewards_np):8.4f}")

    def _compute_action_delta_penalty(self, action_samples):
        """Compute penalty for joint angle deltas exceeding max threshold.

        Args:
            action_samples: (N, T, a_dim) actions in sin/cos form for first 12 dims.
        Returns:
            penalty: (N,) non-negative penalty per sample.
        """
        import numpy as np
        if hasattr(action_samples, "detach"):
            actions = action_samples.detach().cpu().numpy()
        else:
            actions = np.array(action_samples)

        if actions.ndim != 3 or actions.shape[-1] < 12:
            return np.zeros((actions.shape[0],), dtype=np.float32)

        max_delta_rad = np.deg2rad(self.action_delta_max_deg)
        _, penalty = check_angular_velocity_constraint(
            actions,
            action_t_prev=None,
            max_angular_velocity=max_delta_rad,
            start_idx=0,
            end_idx=12,
        )
        return penalty

    def update_dist(self, samples, scores, mu, var):
        # actions: array with shape [num_samples, time, action_dim]
        # scores: array with shape [num_samples]
        
        # Filter out NaN/Inf scores
        valid_mask = np.isfinite(scores.flatten())
        if not np.any(valid_mask):
            print("[CEM Warning] All scores are NaN/Inf! Keeping previous distribution.")
            return mu, var
        
        valid_samples = samples[valid_mask]
        valid_scores = scores.flatten()[valid_mask]
        
        num_elites = max(1, int(self.elites_frac * len(valid_scores)))
        indices = np.argsort(-valid_scores)[:num_elites]
        elite_samples = valid_samples[indices]
        n_mu = np.mean(elite_samples, axis=0)
        n_var = np.var(elite_samples, axis=0)
        new_mu = self.alpha * mu + (1 - self.alpha) * n_mu
        new_var = self.alpha * var + (1 - self.alpha) * n_var
        return new_mu, new_var

    def rounded_gripper_action(self, actions):
        # Gripper action is assumed to be in the last dimension
        actions[..., -1] = np.where(actions[..., -1] < 0, -1, 1)
        return actions

    def score_trajectories(
        self,
        new_action_samples,
        obs_history,
        state_history,
        action_history,
        goal,
        requires_grad=False,
    ):

        n_ctxt = self.model.num_context
        # Extract last n_ctxt context frames for conditioning
        action_history = action_history[-n_ctxt:]
        obs_history = obs_history[-n_ctxt:]
        state_history = state_history[-n_ctxt:]
        # Replicate context actions for all samples in batch
        context_actions = np.tile(
            np.array(action_history)[None], (new_action_samples.shape[0], 1, 1)
        )

        if requires_grad:
            import torch

            new_action_samples = project_joint_angles_torch(new_action_samples, start_idx=0, end_idx=12)
            if new_action_samples.shape[-1] >= 15:
                new_action_samples[..., 12:15] = torch.clamp(new_action_samples[..., 12:15], -1, 1)
            action_samples = torch.cat(
                (
                    torch.from_numpy(context_actions).to(new_action_samples),
                    new_action_samples,
                ),
                axis=1,
            )
        else:
            import torch
            new_action_samples_torch = torch.from_numpy(new_action_samples).float()
            new_action_samples_torch = project_joint_angles_torch(new_action_samples_torch, start_idx=0, end_idx=12)
            new_action_samples = new_action_samples_torch.cpu().numpy()
            if new_action_samples.shape[-1] >= 15:
                new_action_samples[..., 12:15] = np.clip(new_action_samples[..., 12:15], -1, 1)
            action_samples = np.concatenate(
                (context_actions, new_action_samples), axis=1
            )

        if self.round_gripper_action:
            action_samples = self.rounded_gripper_action(action_samples)

        # Prepare batch for model prediction
        batch = {
            "video": np.tile(
                np.array(obs_history[self.model.base_prediction_modality])[None],
                (new_action_samples.shape[0], 1, 1, 1, 1),
            ),
            "actions": action_samples,
            "state_obs": state_history,
        }

        # Run model prediction
        predictions = self.model(batch, grad_enabled=requires_grad)
        
        # 将goal转换到CUDA设备（提升objective计算性能）
        import torch
        
        # 处理ObservationList类型的goal
        if hasattr(goal, 'data_dict'):
            goal_dict = goal.data_dict
        else:
            goal_dict = goal
        
        goal_gpu = {}
        for key, value in goal_dict.items():
            if isinstance(value, np.ndarray):
                goal_gpu[key] = torch.from_numpy(value).to(self.model.device)
            elif isinstance(value, torch.Tensor):
                goal_gpu[key] = value.to(self.model.device)
            else:
                goal_gpu[key] = value

        # Add previous frame for perceptual loss (latest context frame)
        if "prev_rgb" not in goal_gpu:
            prev_rgb = np.array(obs_history[self.model.base_prediction_modality])[-1]
            goal_gpu["prev_rgb"] = torch.from_numpy(prev_rgb[None]).to(self.model.device)
        
        rewards = self.obj_fn(predictions, goal_gpu)

        # Apply action delta penalty if enabled
        if self.action_delta_penalty_weight > 0:
            penalty = self._compute_action_delta_penalty(action_samples)
            if isinstance(rewards, torch.Tensor):
                rewards = rewards - self.action_delta_penalty_weight * torch.from_numpy(penalty).to(rewards.device)[:, None, None]
            else:
                rewards = rewards - self.action_delta_penalty_weight * penalty[:, None, None]
        
        # 转换rewards为numpy（如果是torch tensor且不需要梯度）
        import torch
        if isinstance(rewards, torch.Tensor) and not requires_grad:
            rewards = rewards.cpu().numpy()
        
        return predictions, rewards, action_samples

    def perform_cem(
        self,
        t,
        log_dir,
        obs_history,
        state_history,
        action_history,
        goal,
        init_mean=None,
    ):
        # Reset rewards history for this planning step
        self.last_rewards_history = []
        
        # Initialize mean: use init_mean if provided, otherwise use last action from history
        if init_mean is not None:
            mu = np.zeros((self.horizon, self.a_dim))
            mu[: len(init_mean)] = init_mean
            mu[len(init_mean) :] = init_mean[-1]
        elif action_history is not None and len(action_history) > 0:
            # Use last action from history as starting point (current control state u)
            last_action = np.array(action_history[-1])
            mu = np.tile(last_action[None], (self.horizon, 1))
            if self.verbose:
                print(f"  Initializing CEM from current control u (from action_history)")
        else:
            mu = np.zeros((self.horizon, self.a_dim))
            if self.verbose:
                print(f"  WARNING: Initializing CEM from zero (no action history available)")
        var = np.tile((self.init_std**2)[None], (self.horizon, 1))

        for iter in range(self.opt_iters):
            lb_dist = mu - self.lower_bound
            ub_dist = self.upper_bound - mu
            constrained_var = np.minimum(
                np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var
            )
            new_action_samples = self.sampler.sample_actions(
                self.num_samples, mu, np.sqrt(constrained_var)
            )
            predictions, rewards, action_samples = self.score_trajectories(
                new_action_samples,
                obs_history,
                state_history,
                action_history,
                goal,
            )
            
            # 🔍 调试信息：打印loss的各个组成部分（仅在最后一次迭代）
            if self.verbose and iter == self.opt_iters - 1:
                self._print_loss_breakdown(predictions, goal, rewards, iter, obs_history)

            # 可视化逻辑：根据save_all_iterations决定保存策略
            should_save = False
            if self.save_all_iterations:
                # 保存所有迭代（旧行为）
                should_save = (log_dir is not None and t % self.log_every == 0)
                filename = f"{log_dir}/step_{t}_itr_{iter}_best_plan"
            else:
                # 只保存最后一次迭代（新行为，减少计算）
                should_save = (iter == self.opt_iters - 1 and log_dir is not None and t % self.log_every == 0)
                filename = f"{log_dir}/step_{t}_best_plan"
            
            if should_save:
                # 只取最优的1个样本进行可视化，减少计算量
                best_prediction_ind = np.argmax(rewards.flatten())
                
                # 安全地索引predictions
                best_pred_dict = {}
                for k, v in predictions.items():
                    if isinstance(v, list):
                        # 如果是list，检查是否为空
                        if len(v) > 0:
                            best_pred_dict[k] = v[best_prediction_ind] if isinstance(v[0], np.ndarray) else v
                    elif isinstance(v, np.ndarray):
                        best_pred_dict[k] = v[best_prediction_ind]
                    else:
                        best_pred_dict[k] = v
                
                best_pred = ObservationList(
                    best_pred_dict,
                    image_shape=(self.model.image_height, self.model.image_width)
                )
                best_reward = rewards[best_prediction_ind]
                
                # 准备goal用于可视化（去掉batch维度）
                goal_for_vis = ObservationList(
                    {k: v[0] if len(v.shape) > 3 else v for k, v in goal.data_dict.items()},
                    image_shape=goal.image_shape
                )
                
                self.log_best_plans(
                    filename,
                    [best_pred],
                    goal_for_vis,
                    [best_reward],
                    include_score=self.include_score_in_gif,  # 传递参数
                )
            
            if self.verbose and (iter == 0 or iter == self.opt_iters - 1 or iter % max(1, self.opt_iters // 3) == 0):
                # 转换为numpy如果是torch tensor
                import torch
                if isinstance(rewards, torch.Tensor):
                    rewards_np = rewards.cpu().numpy()
                else:
                    rewards_np = rewards
                
                best_r = np.max(rewards_np)
                mean_r = np.mean(rewards_np)
                std_r = np.std(rewards_np)
                print(f"    Iter {iter+1}/{self.opt_iters}: Best = {best_r:.6f}, Mean = {mean_r:.6f}, Std = {std_r:.6f}")
                
                # Store rewards history
                self.last_rewards_history.append({
                    'iteration': iter + 1,
                    'best': float(best_r),
                    'mean': float(mean_r),
                    'std': float(std_r)
                })

            n_ctxt = self.model.num_context
            # Update distribution using only the planned actions (excluding context)
            mu, var = self.update_dist(
                action_samples[:, n_ctxt:], rewards, mu, var
            )
        
        # Store final optimization results
        import torch
        if isinstance(rewards, torch.Tensor):
            rewards_np = rewards.cpu().numpy()
        else:
            rewards_np = rewards
        self.last_best_reward = float(np.max(rewards_np))
        self.last_mean_reward = float(np.mean(rewards_np))

        return action_samples, rewards

    def plan(
        self,
        t,
        log_dir,
        obs_history,
        state_history,
        action_history,
        goal,
        init_mean=None,
    ):
        """Plan optimal action sequence using CEM.
        
        Args:
            t: Current timestep
            log_dir: Directory for logging visualizations
            obs_history: Historical observations
            state_history: Historical states
            action_history: Historical actions (must have at least num_context elements)
            goal: Target observation to reach
            init_mean: Optional initial mean for action distribution
            
        Returns:
            Optimal action sequence for the planning horizon
        """
        n_ctxt = self.model.num_context
        action_samples, rewards = self.perform_cem(
            t, log_dir, obs_history, state_history, action_history, goal, init_mean
        )
        # Return best trajectory's planned actions (excluding context)
        return action_samples[np.argmax(rewards), n_ctxt:]
