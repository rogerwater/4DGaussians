"""
HexPlane Feature Analyzer
分析HexPlane特征表征质量，检测特征坍塌、饱和等问题
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')


class HexPlaneAnalyzer:
    """
    HexPlane特征分析器
    
    监控指标:
    1. 特征统计 (均值、方差、范围等)
    2. 特征坍塌检测 (样本间相似度)
    3. 特征有效秩 (维度利用率)
    4. 平面贡献度分析
    5. 控制信号可分性
    6. 梯度健康度
    """
    
    def __init__(
        self, 
        deformation_net: nn.Module,
        config: object,
        num_sample_positions: int = 2000,
        num_sample_actions: int = 50,
        enable_tsne: bool = False
    ):
        """
        Args:
            deformation_net: 形变网络 (包含HexPlane)
            config: 配置对象
            num_sample_positions: 空间采样点数
            num_sample_actions: 动作向量采样数
            enable_tsne: 是否启用t-SNE (计算量大)
        """
        self.deformation_net = deformation_net
        self.config = config
        self.num_sample_positions = num_sample_positions
        self.num_sample_actions = num_sample_actions
        self.enable_tsne = enable_tsne
        
        # 获取HexPlane
        if hasattr(deformation_net, 'deformation_net'):
            self.hexplane = deformation_net.deformation_net.grid
        elif hasattr(deformation_net, 'grid'):
            self.hexplane = deformation_net.grid
        else:
            raise AttributeError("Cannot find HexPlane in deformation network")
        
        # 获取AABB
        self.aabb = self.hexplane.get_aabb if hasattr(self.hexplane, 'get_aabb') else None
        
        print(f"[FeatureAnalyzer] Initialized")
        print(f"  - Sample positions: {num_sample_positions}")
        print(f"  - Sample actions: {num_sample_actions}")
        print(f"  - t-SNE enabled: {enable_tsne}")
    
    def sample_positions(self) -> torch.Tensor:
        """
        在AABB内均匀采样3D位置
        
        Returns:
            positions: [N, 3]
        """
        if self.aabb is not None:
            aabb = self.aabb
            xyz_min = aabb[0]
            xyz_max = aabb[1]
        else:
            # 默认范围
            xyz_min = torch.tensor([-1.0, -1.0, -1.0])
            xyz_max = torch.tensor([1.0, 1.0, 1.0])
        
        # 均匀采样
        positions = torch.rand(self.num_sample_positions, 3, device='cuda')
        positions = xyz_min + positions * (xyz_max - xyz_min)
        
        return positions
    
    def sample_actions(self, scene=None) -> torch.Tensor:
        """
        采样动作向量
        
        策略:
        - 使用训练集中的动作向量
        - 添加一些插值的动作向量 (测试泛化)
        
        Args:
            scene: 场景对象，包含训练数据
            
        Returns:
            actions: [M, action_dim]
        """
        action_dim = getattr(self.config, 'action_input_dim', 6)
        
        if scene is not None and hasattr(scene, 'getTrainCameras'):
            # 从训练集采样
            train_cameras = scene.getTrainCameras()
            if len(train_cameras) > 0:
                # 收集所有动作向量
                train_actions = []
                for cam in train_cameras:
                    if hasattr(cam, 'action_vec') and cam.action_vec is not None:
                        action = cam.action_vec
                        if isinstance(action, torch.Tensor):
                            train_actions.append(action.cpu())
                        else:
                            train_actions.append(torch.tensor(action))
                
                if len(train_actions) > 0:
                    train_actions = torch.stack(train_actions)
                    
                    # 随机选择一部分
                    n_train = min(self.num_sample_actions // 2, len(train_actions))
                    indices = torch.randperm(len(train_actions))[:n_train]
                    sampled_actions = train_actions[indices]
                    
                    # 生成插值动作向量
                    n_interp = self.num_sample_actions - n_train
                    if n_interp > 0:
                        interp_actions = []
                        for _ in range(n_interp):
                            idx1, idx2 = torch.randint(0, len(train_actions), (2,))
                            alpha = torch.rand(1)
                            interp = alpha * train_actions[idx1] + (1 - alpha) * train_actions[idx2]
                            interp_actions.append(interp)
                        interp_actions = torch.stack(interp_actions)
                        sampled_actions = torch.cat([sampled_actions, interp_actions], dim=0)
                    
                    return sampled_actions.cuda()
        
        # 备用: 随机采样
        actions = torch.randn(self.num_sample_actions, action_dim, device='cuda')
        # 归一化到合理范围 [-1, 1]
        actions = torch.tanh(actions)
        
        return actions
    
    @torch.no_grad()
    def extract_features(
        self, 
        positions: torch.Tensor, 
        actions: torch.Tensor,
        batch_size: int = 500
    ) -> Tuple[torch.Tensor, Dict]:
        """
        提取HexPlane特征
        
        Args:
            positions: [N, 3]
            actions: [M, action_dim]
            batch_size: 批量大小，避免OOM
            
        Returns:
            features: [N*M, feat_dim] - 所有组合的特征
            extras: dict - 额外信息 (平面特征等)
        """
        N = positions.shape[0]
        M = actions.shape[0]
        
        all_features = []
        all_plane_features = []
        
        # 编码动作向量
        if hasattr(self.deformation_net, 'action_encoder'):
            action_latents = []
            for i in range(0, M, batch_size):
                batch_action = actions[i:i+batch_size]
                latent = self.deformation_net.action_encoder(batch_action)
                action_latents.append(latent)
            action_latents = torch.cat(action_latents, dim=0)  # [M, 1]
        else:
            # 如果没有encoder，直接用第一个维度
            action_latents = actions[:, :1]
        
        # 对每个动作向量，提取所有位置的特征
        for m in range(M):
            action_latent = action_latents[m:m+1].expand(N, -1)  # [N, 1]
            
            batch_features = []
            batch_plane_features = []
            
            for i in range(0, N, batch_size):
                batch_pos = positions[i:i+batch_size]
                batch_action = action_latent[i:i+batch_size]
                
                # 提取HexPlane特征
                feat = self.hexplane(batch_pos, batch_action)
                batch_features.append(feat)
                
                # 尝试提取各平面的特征 (如果支持)
                if hasattr(self.hexplane, 'grid_coefs'):
                    plane_feats = self._extract_plane_features(batch_pos, batch_action)
                    batch_plane_features.append(plane_feats)
            
            features_for_ctrl = torch.cat(batch_features, dim=0)
            all_features.append(features_for_ctrl)
            
            if batch_plane_features:
                plane_feats_for_ctrl = {
                    k: torch.cat([pf[k] for pf in batch_plane_features], dim=0)
                    for k in batch_plane_features[0].keys()
                }
                all_plane_features.append(plane_feats_for_ctrl)
        
        # 合并所有特征 [M, N, D] -> [M*N, D]
        features = torch.stack(all_features, dim=0).reshape(-1, all_features[0].shape[-1])
        
        extras = {
            'plane_features': all_plane_features if all_plane_features else None,
            'action_latents': action_latents,
            'N': N,
            'M': M
        }
        
        return features, extras
    
    def _extract_plane_features(
        self, 
        positions: torch.Tensor, 
        action_latents: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """提取各个平面的特征"""
        plane_features = {}
        
        if not hasattr(self.hexplane, 'grid_coefs'):
            return plane_features
        
        # HexPlane有6个平面: XY, XZ, YZ, XT, YT, ZT
        plane_names = ['XY', 'XZ', 'YZ', 'XT', 'YT', 'ZT']
        
        # 归一化坐标
        if hasattr(self.hexplane, 'aabb'):
            aabb = self.hexplane.aabb
            pts = (positions - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0
        else:
            pts = positions
        
        # 构建4D坐标 [x, y, z, t]
        pts_4d = torch.cat([pts, action_latents], dim=-1)  # [N, 4]
        
        # 提取每个平面的特征
        coo_combs = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
        
        for i, (name, coo_comb) in enumerate(zip(plane_names, coo_combs)):
            if i < len(self.hexplane.grid_coefs):
                plane_grid = self.hexplane.grid_coefs[i]
                coords = pts_4d[:, coo_comb]
                
                # Grid sample
                from scene.hexplane import grid_sample_wrapper
                feat = grid_sample_wrapper(plane_grid, coords)
                plane_features[name] = feat
        
        return plane_features
    
    def compute_statistics(self, features: torch.Tensor) -> Dict[str, float]:
        """
        计算特征统计量
        
        Args:
            features: [N, D]
            
        Returns:
            stats: 统计指标字典
        """
        stats = {}
        
        # 基础统计
        stats['mean'] = features.mean().item()
        stats['std'] = features.std().item()
        stats['max'] = features.max().item()
        stats['min'] = features.min().item()
        stats['norm'] = features.norm(dim=-1).mean().item()
        
        # 稀疏度 (接近0的特征比例)
        threshold = 0.01
        stats['sparsity'] = (features.abs() < threshold).float().mean().item()
        
        # 死神经元 (某维度在所有样本都接近0)
        dim_max = features.abs().max(dim=0)[0]
        stats['dead_neurons'] = (dim_max < threshold).float().mean().item()
        
        # 饱和度 (极端值比例)
        saturation_threshold = 3.0
        stats['saturation'] = (features.abs() > saturation_threshold).float().mean().item()
        
        return stats
    
    def compute_similarity(self, features: torch.Tensor, sample_size: int = 500) -> Dict[str, float]:
        """
        计算样本间相似度 (检测特征坍塌)
        
        Args:
            features: [N, D]
            sample_size: 采样数量 (太多会很慢)
            
        Returns:
            similarity_metrics: 相似度指标
        """
        N = features.shape[0]
        if N > sample_size:
            indices = torch.randperm(N)[:sample_size]
            features = features[indices]
        
        # 归一化
        features_norm = torch.nn.functional.normalize(features, p=2, dim=-1)
        
        # 计算余弦相似度矩阵
        sim_matrix = torch.mm(features_norm, features_norm.t())
        
        # 排除对角线 (自己和自己的相似度)
        mask = ~torch.eye(sim_matrix.shape[0], dtype=torch.bool, device=sim_matrix.device)
        off_diag_sim = sim_matrix[mask]
        
        metrics = {
            'mean_similarity': off_diag_sim.mean().item(),
            'max_similarity': off_diag_sim.max().item(),
            'min_similarity': off_diag_sim.min().item(),
            'std_similarity': off_diag_sim.std().item(),
        }
        
        # 相似度矩阵 (用于可视化)
        metrics['similarity_matrix'] = sim_matrix.cpu().numpy()
        
        return metrics
    
    def compute_effective_rank(self, features: torch.Tensor) -> Dict[str, float]:
        """
        计算特征有效秩
        
        Args:
            features: [N, D]
            
        Returns:
            rank_metrics: 秩相关指标
        """
        # 中心化
        features_centered = features - features.mean(dim=0, keepdim=True)
        
        # SVD
        try:
            U, S, V = torch.svd(features_centered)
            singular_values = S.cpu().numpy()
            
            # 归一化奇异值
            singular_values = singular_values / (singular_values.sum() + 1e-10)
            
            # 有效秩 (Shannon entropy)
            entropy = -(singular_values * np.log(singular_values + 1e-10)).sum()
            effective_rank = np.exp(entropy)
            
            # 特征利用率
            feat_dim = features.shape[1]
            utilization = effective_rank / feat_dim
            
            # 90%能量的维度数
            cumsum = np.cumsum(singular_values)
            n_components_90 = np.searchsorted(cumsum, 0.9) + 1
            
            metrics = {
                'effective_rank': float(effective_rank),
                'rank_utilization': float(utilization),
                'n_components_90': int(n_components_90),
                'singular_values': singular_values,
            }
            
        except RuntimeError as e:
            print(f"[FeatureAnalyzer] SVD failed: {e}")
            metrics = {
                'effective_rank': 0.0,
                'rank_utilization': 0.0,
                'n_components_90': 0,
                'singular_values': None,
            }
        
        return metrics
    
    def analyze_plane_contribution(self, extras: Dict) -> Dict[str, float]:
        """
        分析各平面的贡献度
        
        Args:
            extras: 包含plane_features的字典
            
        Returns:
            contributions: 各平面贡献度
        """
        contributions = {}
        
        if extras.get('plane_features') is None:
            return contributions
        
        plane_features = extras['plane_features']
        if not plane_features:
            return contributions
        
        # 计算每个平面的平均激活强度
        plane_names = ['XY', 'XZ', 'YZ', 'XT', 'YT', 'ZT']
        
        for name in plane_names:
            activations = []
            for pf in plane_features:
                if name in pf:
                    feat = pf[name]
                    activation = feat.abs().mean().item()
                    activations.append(activation)
            
            if activations:
                contributions[f'plane_{name}'] = np.mean(activations)
        
        # 计算空间 vs 时间贡献比
        spatial_contrib = sum([contributions.get(f'plane_{p}', 0) for p in ['XY', 'XZ', 'YZ']])
        temporal_contrib = sum([contributions.get(f'plane_{p}', 0) for p in ['XT', 'YT', 'ZT']])
        
        if temporal_contrib > 1e-6:
            contributions['spatial_temporal_ratio'] = spatial_contrib / temporal_contrib
        else:
            contributions['spatial_temporal_ratio'] = float('inf')
        
        # 归一化贡献度
        total = sum([contributions.get(f'plane_{p}', 0) for p in plane_names])
        if total > 1e-6:
            for name in plane_names:
                key = f'plane_{name}'
                if key in contributions:
                    contributions[f'{key}_normalized'] = contributions[key] / total
        
        return contributions
    
    def analyze_control_separability(
        self, 
        features: torch.Tensor, 
        extras: Dict
    ) -> Dict[str, float]:
        """
        分析控制信号可分性
        
        Args:
            features: [N*M, D]
            extras: 包含N, M的字典
            
        Returns:
            separability_metrics: 可分性指标
        """
        N = extras['N']
        M = extras['M']
        
        # Reshape: [M, N, D]
        features_reshaped = features.reshape(M, N, -1)
        
        # 固定空间位置，看不同控制的特征差异
        # 选择几个代表性的空间位置
        n_spatial_samples = min(50, N)
        spatial_indices = torch.linspace(0, N-1, n_spatial_samples, dtype=torch.long)
        
        features_sampled = features_reshaped[:, spatial_indices, :]  # [M, n_spatial, D]
        
        # 计算控制间距离
        inter_control_distances = []
        for i in range(M):
            for j in range(i+1, M):
                dist = (features_sampled[i] - features_sampled[j]).norm(dim=-1).mean()
                inter_control_distances.append(dist.item())
        
        # 计算控制内距离 (同一控制，不同空间位置)
        intra_control_distances = []
        for m in range(M):
            feats = features_sampled[m]  # [n_spatial, D]
            if feats.shape[0] > 1:
                dists = torch.cdist(feats, feats)
                mask = ~torch.eye(dists.shape[0], dtype=torch.bool, device=dists.device)
                intra_control_distances.append(dists[mask].mean().item())
        
        inter_dist_mean = np.mean(inter_control_distances) if inter_control_distances else 0
        intra_dist_mean = np.mean(intra_control_distances) if intra_control_distances else 1e-6
        
        # 可分性指标: 类间距离 / 类内距离
        separability = inter_dist_mean / (intra_dist_mean + 1e-6)
        
        metrics = {
            'separability': float(separability),
            'inter_control_distance': float(inter_dist_mean),
            'intra_control_distance': float(intra_dist_mean),
        }
        
        return metrics
    
    def analyze_and_log(
        self, 
        tb_writer, 
        iteration: int, 
        scene=None
    ):
        """
        执行完整分析并记录到TensorBoard
        
        Args:
            tb_writer: TensorBoard writer
            iteration: 当前迭代步数
            scene: 场景对象
        """
        print(f"\n[FeatureAnalyzer] Analyzing at iteration {iteration}...")
        
        try:
            # 1. 采样
            positions = self.sample_positions()
            actions = self.sample_actions(scene)
            print(f"  Sampled {positions.shape[0]} positions, {actions.shape[0]} actions")
            
            # 2. 提取特征
            features, extras = self.extract_features(positions, actions)
            print(f"  Extracted features: {features.shape}")
            
            # 3. 计算各项指标
            stats = self.compute_statistics(features)
            similarity = self.compute_similarity(features)
            rank = self.compute_effective_rank(features)
            plane_contrib = self.analyze_plane_contribution(extras)
            control_sep = self.analyze_control_separability(features, extras)
            
            # 4. 记录到TensorBoard
            # 基础统计
            tb_writer.add_scalar('Feature/Mean', stats['mean'], iteration)
            tb_writer.add_scalar('Feature/Std', stats['std'], iteration)
            tb_writer.add_scalar('Feature/Max', stats['max'], iteration)
            tb_writer.add_scalar('Feature/Min', stats['min'], iteration)
            tb_writer.add_scalar('Feature/Norm', stats['norm'], iteration)
            tb_writer.add_scalar('Feature/Sparsity', stats['sparsity'], iteration)
            tb_writer.add_scalar('Feature/DeadNeurons', stats['dead_neurons'], iteration)
            tb_writer.add_scalar('Feature/Saturation', stats['saturation'], iteration)
            
            # 相似度
            tb_writer.add_scalar('Feature/MeanSimilarity', similarity['mean_similarity'], iteration)
            tb_writer.add_scalar('Feature/StdSimilarity', similarity['std_similarity'], iteration)
            
            # 有效秩
            tb_writer.add_scalar('Feature/EffectiveRank', rank['effective_rank'], iteration)
            tb_writer.add_scalar('Feature/RankUtilization', rank['rank_utilization'], iteration)
            tb_writer.add_scalar('Feature/NComponents90', rank['n_components_90'], iteration)
            
            # 平面贡献
            for key, value in plane_contrib.items():
                if 'normalized' not in key:
                    tb_writer.add_scalar(f'Plane/{key}', value, iteration)
            
            # 控制可分性
            tb_writer.add_scalar('Control/Separability', control_sep['separability'], iteration)
            tb_writer.add_scalar('Control/InterDistance', control_sep['inter_control_distance'], iteration)
            
            # 特征分布直方图
            tb_writer.add_histogram('Feature/Distribution', features.cpu().numpy(), iteration)
            
            # 奇异值谱
            if rank['singular_values'] is not None:
                from utils.visualizers import plot_scree
                scree_img = plot_scree(rank['singular_values'])
                tb_writer.add_image('Feature/ScreePlot', scree_img, iteration, dataformats='HWC')
            
            # 相似度矩阵
            if 'similarity_matrix' in similarity:
                from utils.visualizers import plot_similarity_matrix
                sim_img = plot_similarity_matrix(similarity['similarity_matrix'])
                tb_writer.add_image('Feature/SimilarityMatrix', sim_img, iteration, dataformats='HWC')
            
            # 平面贡献雷达图
            if plane_contrib:
                from utils.visualizers import plot_plane_contributions
                radar_img = plot_plane_contributions(plane_contrib)
                tb_writer.add_image('Plane/ContributionRadar', radar_img, iteration, dataformats='HWC')
            
            # 打印警告
            self._check_warnings(stats, similarity, rank, plane_contrib, control_sep)
            
            print(f"[FeatureAnalyzer] Analysis completed!")
            
        except Exception as e:
            print(f"[FeatureAnalyzer] Error during analysis: {e}")
            import traceback
            traceback.print_exc()
    
    def _check_warnings(self, stats, similarity, rank, plane_contrib, control_sep):
        """检查并打印警告"""
        warnings = []
        
        # 特征坍塌
        if similarity['mean_similarity'] > 0.9:
            warnings.append("⚠️  Feature collapse detected! (high similarity)")
        
        # 有效秩过低
        if rank['rank_utilization'] < 0.3:
            warnings.append("⚠️  Low feature rank! (dimension underutilized)")
        
        # 死神经元过多
        if stats['dead_neurons'] > 0.4:
            warnings.append(f"⚠️  High dead neuron ratio: {stats['dead_neurons']:.2%}")
        
        # 特征饱和
        if stats['saturation'] > 0.1:
            warnings.append(f"⚠️  Feature saturation detected: {stats['saturation']:.2%}")
        
        # 平面利用不均
        for key, value in plane_contrib.items():
            if key.startswith('plane_') and 'normalized' not in key:
                if value < 0.05:
                    warnings.append(f"⚠️  Plane {key} underutilized: {value:.4f}")
        
        # 控制可分性低
        if control_sep['separability'] < 1.5:
            warnings.append(f"⚠️  Low control separability: {control_sep['separability']:.2f}")
        
        if warnings:
            print("\n" + "="*70)
            print("  Feature Analysis Warnings:")
            for w in warnings:
                print(f"  {w}")
            print("="*70 + "\n")
