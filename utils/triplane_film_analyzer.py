"""
TriPlane+FiLM Feature Analyzer

分析 TriPlane+FiLM 架构的特征质量，包括：
1. TriPlane空间表征能力
2. Action编码质量和可分性
3. FiLM调制的有效性
4. 跨模态融合效果

与HexPlaneAnalyzer的主要区别：
- 关注3个空间平面而非6个时空平面
- 新增action embedding分析
- 新增FiLM modulation分析
- 新增spatial-action correlation分析
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


class TriPlaneFiLMAnalyzer:
    """
    TriPlane+FiLM架构的综合特征分析器
    
    主要功能:
    1. 分析TriPlane的空间表征能力
    2. 评估ControlProcessor的action编码质量
    3. 监控FiLM调制的有效性
    4. 可视化action在特征空间的分布
    5. 分析spatial-action跨模态融合效果
    """
    
    def __init__(
        self,
        deformation_net,
        config,
        num_sample_positions: int = 2000,
        num_sample_actions: int = 100,
        action_dim: Optional[int] = None,
        enable_tsne: bool = True,
        save_embeddings: bool = True,
        device: str = 'cuda'
    ):
        """
        初始化分析器
        
        Args:
            deformation_net: DeformationTriPlane实例
            config: 配置参数
            num_sample_positions: 空间采样点数
            num_sample_actions: 动作采样数
            action_dim: 动作维度（自动从config推断）
            enable_tsne: 启用t-SNE可视化
            save_embeddings: 保存特征向量供后续分析
            device: 计算设备
        """
        self.deformation_net = deformation_net
        self.config = config
        self.device = device
        
        # 提取关键组件
        # 注意: deformation_net 可能是 deform_network_triplane 或 DeformationTriPlane
        if hasattr(deformation_net, 'deformation_net'):
            # deform_network_triplane 包装器
            inner_net = deformation_net.deformation_net
        else:
            # 直接是 DeformationTriPlane
            inner_net = deformation_net
        
        self.triplane = inner_net.triplane
        self.action_processor = inner_net.action_processor
        self.film_decoder = inner_net.film_decoder
        
        # 采样配置
        self.num_pos = num_sample_positions
        self.num_actions = num_sample_actions
        self.action_dim = action_dim or getattr(config, 'control_input_dim', 15)
        
        # 可视化配置
        self.enable_tsne = enable_tsne
        self.save_embeddings = save_embeddings
        
        # 特征缓存
        self.cached_features = {}
        
        print(f"[TriPlaneFiLMAnalyzer] Initialized")
        print(f"  - Spatial samples: {self.num_pos}")
        print(f"  - Action samples: {self.num_actions}")
        print(f"  - Action dim: {self.action_dim}")
        print(f"  - t-SNE enabled: {self.enable_tsne}")
    
    # ========================================================================
    # 1. 空间特征分析
    # ========================================================================
    
    def analyze_spatial_features(self) -> Dict:
        """
        分析TriPlane的空间表征能力
        
        返回:
            包含空间特征各项指标的字典
        """
        print("[SpatialAnalysis] Analyzing TriPlane spatial features...")
        
        # 采样空间位置
        positions = self._sample_positions(self.num_pos)
        
        with torch.no_grad():
            spatial_feats = self.triplane(positions)  # [N, feat_dim]
        
        # 1. 特征多样性 (方差)
        feature_variance = spatial_feats.var(dim=0)
        mean_variance = feature_variance.mean().item()
        min_variance = feature_variance.min().item()
        
        # 2. 特征秩 (有效维度)
        U, S, V = torch.svd(spatial_feats.cpu())
        explained_variance_ratio = (S ** 2) / (S ** 2).sum()
        effective_rank = (explained_variance_ratio > 0.01).sum().item()
        
        # 3. 特征相似度 (检查是否坍塌)
        spatial_norm = F.normalize(spatial_feats, dim=1)
        similarity_matrix = torch.mm(spatial_norm, spatial_norm.t())
        off_diagonal = similarity_matrix - torch.eye(self.num_pos, device=self.device)
        mean_similarity = off_diagonal.abs().mean().item()
        
        # 4. 空间覆盖率 (通过KNN评估)
        distances = torch.cdist(positions, positions)
        k = 10
        knn_distances = torch.topk(distances, k=k+1, largest=False)[0][:, 1:]
        mean_knn_distance = knn_distances.mean().item()
        
        return {
            'mean_variance': mean_variance,
            'min_variance': min_variance,
            'effective_rank': effective_rank,
            'total_dims': spatial_feats.shape[1],
            'mean_similarity': mean_similarity,
            'diversity_score': 1.0 / (mean_similarity + 1e-6),
            'spatial_coverage': mean_knn_distance,
            'feature_variance': feature_variance.cpu().numpy(),
            'singular_values': S.numpy()
        }
    
    # ========================================================================
    # 2. Action编码分析
    # ========================================================================
    
    def analyze_action_encoding(self) -> Dict:
        """
        分析ControlProcessor的action编码质量
        
        重点评估:
        1. Action separability - 不同action的可分性
        2. Embedding rank - 有效编码维度
        3. Cluster quality - 聚类质量
        4. Feature distribution - 特征分布均匀性
        
        返回:
            包含action编码各项指标的字典
        """
        print("[ActionAnalysis] Analyzing action encoding quality...")
        
        # 1. 采样action空间
        actions = self._sample_actions(self.num_actions)
        
        # 2. 通过ControlProcessor编码
        with torch.no_grad():
            action_features = self.action_processor(actions)  # [N, control_dim]
        
        # 3. 计算action separability
        action_features_norm = F.normalize(action_features, dim=1)
        similarity_matrix = torch.mm(action_features_norm, 
                                      action_features_norm.t())
        
        # 对角线是自相似(=1)，非对角线应该小
        off_diagonal = similarity_matrix - torch.eye(self.num_actions, device=self.device)
        mean_similarity = off_diagonal.abs().mean().item()
        
        # Separability = 1 / mean_similarity (越大越好)
        action_separability = 1.0 / (mean_similarity + 1e-6)
        
        # 4. 计算有效维度 (通过SVD)
        U, S, V = torch.svd(action_features.cpu())
        explained_variance_ratio = (S ** 2) / (S ** 2).sum()
        effective_rank = (explained_variance_ratio > 0.01).sum().item()
        
        # 5. 聚类质量评估
        n_clusters = min(10, self.num_actions // 10)
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(action_features.cpu().numpy())
            silhouette = silhouette_score(action_features.cpu().numpy(), labels)
        else:
            silhouette = 0.0
        
        # 6. 平均配对距离
        distances = torch.cdist(action_features, action_features)
        triu_indices = torch.triu(torch.ones_like(distances), diagonal=1).bool()
        mean_distance = distances[triu_indices].mean().item()
        
        # 7. 特征分布均匀性 (通过标准差评估)
        feature_std = action_features.std(dim=0)
        std_uniformity = feature_std.std().item()  # 标准差的标准差，越小越均匀
        
        return {
            'action_separability': action_separability,
            'embedding_rank': effective_rank,
            'total_dims': action_features.shape[1],
            'cluster_quality': silhouette,
            'mean_pairwise_distance': mean_distance,
            'mean_similarity': mean_similarity,
            'std_uniformity': std_uniformity,
            'action_variance': action_features.var(dim=0).cpu().numpy(),
            'similarity_matrix': similarity_matrix.cpu().numpy(),
            'action_features': action_features.cpu().numpy(),
            'singular_values': S.numpy(),
            'explained_variance_ratio': explained_variance_ratio.numpy()
        }
    
    # ========================================================================
    # 3. FiLM调制分析
    # ========================================================================
    
    def analyze_film_modulation(self, num_samples: int = 1000) -> Dict:
        """
        分析FiLM层的调制效果
        
        关键问题:
        1. γ (scale) 的分布 - 应该有多样性，不应该都接近1
        2. β (shift) 的分布 - 应该有明显的偏移效果
        3. 不同action下的γ/β差异 - 验证action-dependent modulation
        4. 各层的调制强度 - 评估各层贡献
        
        返回:
            包含FiLM调制各项指标的字典
        """
        print("[FiLMAnalysis] Analyzing FiLM modulation patterns...")
        
        # 1. 采样空间位置和动作
        positions = self._sample_positions(num_samples)
        actions = self._sample_actions(num_samples)
        
        # 2. 获取空间特征和控制特征
        with torch.no_grad():
            spatial_feat = self.triplane(positions)
            action_feat = self.action_processor(actions)
        
        # 3. 逐层分析FiLM调制
        gamma_stats = []
        beta_stats = []
        modulation_strength = []
        
        h = spatial_feat
        for layer_idx, film_block in enumerate(self.film_decoder.film_blocks):
            # 获取该层的FiLM Layer
            film_layer = film_block.film
            
            # 前向传播（正确模拟FiLMBlock的流程）
            with torch.no_grad():
                # 保存skip输入（在变换之前）
                h_input = h
                
                # 主路径
                h_before = film_block.linear(h)
                h_before = film_block.norm(h_before)
                
                # 生成γ和β
                film_params = film_layer.film_generator(action_feat)
                gamma = film_params[:, :film_layer.feature_dim]
                beta = film_params[:, film_layer.feature_dim:]
            
            # 统计信息
            gamma_stats.append({
                'mean': gamma.mean().item(),
                'std': gamma.std().item(),
                'min': gamma.min().item(),
                'max': gamma.max().item(),
                'median': gamma.median().item(),
                'histogram': torch.histc(gamma.cpu(), bins=50, min=-2, max=4).numpy()
            })
            
            beta_stats.append({
                'mean': beta.mean().item(),
                'std': beta.std().item(),
                'min': beta.min().item(),
                'max': beta.max().item(),
                'median': beta.median().item(),
                'histogram': torch.histc(beta.cpu(), bins=50, min=-2, max=2).numpy()
            })
            
            # 计算调制强度 = |γ - 1| + |β|
            strength = (gamma - 1.0).abs().mean() + beta.abs().mean()
            modulation_strength.append(strength.item())
            
            # 完成该层的前向（正确的残差连接）
            with torch.no_grad():
                h_modulated = gamma * h_before + beta
                h = film_block.activation(h_modulated)
                identity = film_block.skip(h_input)  # 使用输入而非输出
                h = h + identity
        
        # 4. 计算各层贡献度（需要传入原始输入重新计算带梯度的特征）
        layer_contribution = self._compute_layer_contribution(
            positions[:100], actions[:100]  # 使用子集加速
        )
        
        return {
            'gamma_stats': gamma_stats,
            'beta_stats': beta_stats,
            'modulation_strength': modulation_strength,
            'layer_contribution': layer_contribution,
            'num_layers': len(self.film_decoder.film_blocks)
        }
    
    def _compute_layer_contribution(self, positions, actions) -> List[float]:
        """通过梯度敏感度计算各层贡献
        
        Args:
            positions: 原始空间位置输入 [N, 3]
            actions: 原始动作输入 [N, action_dim]
        """
        # 确保在启用梯度的上下文中计算
        with torch.enable_grad():
            # 重新计算特征（带梯度）
            positions = positions.clone().requires_grad_(True)
            actions = actions.clone().requires_grad_(True)
            
            spatial_feat = self.triplane(positions)
            action_feat = self.action_processor(actions)
            
            contributions = []
            h = spatial_feat
            num_layers = len(self.film_decoder.film_blocks)
            
            for idx, film_block in enumerate(self.film_decoder.film_blocks):
                h_out = film_block(h, action_feat)
                
                # 计算输出对输入的梯度
                output_sum = h_out.sum()
                # 只在最后一层保留计算图
                is_last = (idx == num_layers - 1)
                grad_h = torch.autograd.grad(output_sum, h, retain_graph=not is_last, 
                                             create_graph=False)[0]
                
                # 贡献度 = 梯度的L2范数
                contribution = grad_h.norm().item()
                contributions.append(contribution)
                
                # 下一层的输入需要新的梯度追踪
                if not is_last:
                    h = h_out.detach().requires_grad_(True)
        
        return contributions
    
    # ========================================================================
    # 4. 跨模态融合分析
    # ========================================================================
    
    def analyze_spatial_action_correlation(self) -> Dict:
        """
        分析空间特征与action特征的相关性
        
        核心思想:
        - 固定位置，变化action → 应该产生不同的最终特征
        - 固定action，变化位置 → 应该产生不同的最终特征
        - 交叉效应 → spatial和action不应该完全独立
        
        返回:
            包含跨模态相关性各项指标的字典
        """
        print("[CrossModalAnalysis] Computing spatial-action correlation...")
        
        # 1. 采样网格: 多个位置 × 多个动作
        n_pos = 50
        n_actions = 50
        
        positions = self._sample_positions(n_pos)
        actions = self._sample_actions(n_actions)
        
        # 2. 提取特征
        with torch.no_grad():
            spatial_feats = self.triplane(positions)           # [50, 96]
            action_feats = self.action_processor(actions)     # [50, 32]
        
        # 3. 计算Pearson相关系数
        spatial_norm = (spatial_feats - spatial_feats.mean(0)) / (spatial_feats.std(0) + 1e-6)
        action_norm = (action_feats - action_feats.mean(0)) / (action_feats.std(0) + 1e-6)
        
        # 相关性矩阵 [spatial_dim, control_dim]
        correlation_matrix = torch.mm(spatial_norm.t(), action_norm) / n_pos
        mean_abs_corr = correlation_matrix.abs().mean().item()
        
        # 4. 计算融合有效性
        # 好的融合: spatial和action特征应该有中等相关性 (不太强也不太弱)
        if mean_abs_corr < 0.1:
            fusion_effectiveness = mean_abs_corr / 0.1  # 太弱
        elif mean_abs_corr > 0.7:
            fusion_effectiveness = (1.0 - mean_abs_corr) / 0.3  # 太强
        else:
            fusion_effectiveness = 1.0  # 理想
        
        # 5. 测试FiLM融合后的效果
        # 创建所有组合 (n_pos × n_actions)
        pos_grid = positions.unsqueeze(1).repeat(1, n_actions, 1).reshape(-1, 3)
        act_grid = actions.unsqueeze(0).repeat(n_pos, 1, 1).reshape(-1, self.action_dim)
        
        with torch.no_grad():
            spatial_grid = self.triplane(pos_grid)
            action_grid = self.action_processor(act_grid)
            fused_feats = self.film_decoder(spatial_grid, action_grid)
        
        # 分析融合后的特征
        fused_feats = fused_feats.reshape(n_pos, n_actions, -1)
        
        # 计算位置主效应 (固定action，变化position)
        pos_variance = fused_feats.var(dim=0).mean().item()
        
        # 计算action主效应 (固定position，变化action)
        action_variance = fused_feats.var(dim=1).mean().item()
        
        # 计算总方差和交互效应
        total_variance = fused_feats.reshape(-1, fused_feats.shape[-1]).var(dim=0).mean().item()
        interaction_effect = max(0, total_variance - pos_variance - action_variance)
        
        return {
            'correlation_matrix': correlation_matrix.cpu().numpy(),
            'mean_correlation': mean_abs_corr,
            'fusion_effectiveness': fusion_effectiveness,
            'position_effect': pos_variance,
            'action_effect': action_variance,
            'interaction_effect': interaction_effect,
            'total_variance': total_variance,
            'fused_features': fused_feats.cpu().numpy()
        }
    
    # ========================================================================
    # 5. 可视化方法
    # ========================================================================
    
    def visualize_action_space(self, writer, iteration: int):
        """可视化action在特征空间的分布"""
        print("[Visualization] Generating action space visualizations...")
        
        # 1. 分析action编码
        action_analysis = self.analyze_action_encoding()
        
        # 2. t-SNE可视化
        if self.enable_tsne and self.num_actions >= 30:
            action_feats = action_analysis['action_features']  # [N, 32]
            
            # t-SNE降维到2D
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, self.num_actions // 3))
            action_2d = tsne.fit_transform(action_feats)
            
            # 绘图
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(action_2d[:, 0], action_2d[:, 1], 
                                c=np.arange(len(action_2d)),
                                cmap='viridis', alpha=0.6, s=50)
            ax.set_title(f'Action Embedding t-SNE (Iter {iteration})')
            ax.set_xlabel('t-SNE Dimension 1')
            ax.set_ylabel('t-SNE Dimension 2')
            plt.colorbar(scatter, label='Action Index')
            
            writer.add_figure('action/tsne_embedding', fig, iteration)
            plt.close()
        
        # 3. 相似度矩阵热图
        n_show = min(50, self.num_actions)
        fig, ax = plt.subplots(figsize=(10, 9))
        sns.heatmap(action_analysis['similarity_matrix'][:n_show, :n_show],
                    cmap='coolwarm', center=0, ax=ax,
                    xticklabels=10, yticklabels=10, cbar_kws={'label': 'Cosine Similarity'})
        ax.set_title(f'Action Similarity Matrix (Iter {iteration})')
        ax.set_xlabel('Action Index')
        ax.set_ylabel('Action Index')
        writer.add_figure('action/similarity_matrix', fig, iteration)
        plt.close()
        
        # 4. 维度重要性条形图
        fig, ax = plt.subplots(figsize=(12, 6))
        variance = action_analysis['action_variance']
        ax.bar(range(len(variance)), variance, color='steelblue')
        ax.set_xlabel('Feature Dimension')
        ax.set_ylabel('Variance')
        ax.set_title(f'Action Feature Variance per Dimension (Iter {iteration})')
        ax.grid(axis='y', alpha=0.3)
        writer.add_figure('action/feature_variance', fig, iteration)
        plt.close()
        
        # 5. 奇异值谱
        fig, ax = plt.subplots(figsize=(10, 6))
        singular_values = action_analysis['singular_values']
        ax.plot(singular_values, 'o-', linewidth=2, markersize=6)
        ax.set_xlabel('Component Index')
        ax.set_ylabel('Singular Value')
        ax.set_title(f'Action Embedding Singular Value Spectrum (Iter {iteration})')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        writer.add_figure('action/singular_values', fig, iteration)
        plt.close()
    
    def visualize_film_modulation(self, writer, iteration: int):
        """可视化FiLM调制效果"""
        print("[Visualization] Generating FiLM modulation visualizations...")
        
        film_analysis = self.analyze_film_modulation()
        num_layers = film_analysis['num_layers']
        
        # 1. γ和β的分布直方图
        fig, axes = plt.subplots(num_layers, 2, figsize=(14, 4*num_layers))
        if num_layers == 1:
            axes = axes.reshape(1, -1)
        
        for layer_idx in range(num_layers):
            gamma_hist = film_analysis['gamma_stats'][layer_idx]['histogram']
            beta_hist = film_analysis['beta_stats'][layer_idx]['histogram']
            gamma_mean = film_analysis['gamma_stats'][layer_idx]['mean']
            beta_mean = film_analysis['beta_stats'][layer_idx]['mean']
            
            # γ直方图
            axes[layer_idx, 0].bar(np.linspace(-2, 4, 50), gamma_hist, width=0.12, color='coral')
            axes[layer_idx, 0].axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='γ=1 (identity)')
            axes[layer_idx, 0].axvline(x=gamma_mean, color='blue', linestyle='-', linewidth=2, label=f'mean={gamma_mean:.3f}')
            axes[layer_idx, 0].set_title(f'Layer {layer_idx+1} γ (Scale) Distribution')
            axes[layer_idx, 0].set_xlabel('γ value')
            axes[layer_idx, 0].set_ylabel('Frequency')
            axes[layer_idx, 0].legend()
            axes[layer_idx, 0].grid(axis='y', alpha=0.3)
            
            # β直方图
            axes[layer_idx, 1].bar(np.linspace(-2, 2, 50), beta_hist, width=0.08, color='skyblue')
            axes[layer_idx, 1].axvline(x=0.0, color='red', linestyle='--', linewidth=2, label='β=0 (no shift)')
            axes[layer_idx, 1].axvline(x=beta_mean, color='blue', linestyle='-', linewidth=2, label=f'mean={beta_mean:.3f}')
            axes[layer_idx, 1].set_title(f'Layer {layer_idx+1} β (Shift) Distribution')
            axes[layer_idx, 1].set_xlabel('β value')
            axes[layer_idx, 1].set_ylabel('Frequency')
            axes[layer_idx, 1].legend()
            axes[layer_idx, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        writer.add_figure('film/gamma_beta_distribution', fig, iteration)
        plt.close()
        
        # 2. 调制强度和层贡献度对比图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        layers = [f'Layer {i+1}' for i in range(num_layers)]
        x_pos = np.arange(num_layers)
        
        # 调制强度
        strengths = film_analysis['modulation_strength']
        bars1 = ax1.bar(x_pos, strengths, color='orange', alpha=0.7)
        ax1.set_xlabel('FiLM Layer')
        ax1.set_ylabel('Modulation Strength')
        ax1.set_title(f'FiLM Modulation Strength per Layer (Iter {iteration})')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(layers)
        ax1.grid(axis='y', alpha=0.3)
        
        # 在柱子上标注数值
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{strengths[i]:.3f}',
                    ha='center', va='bottom', fontsize=10)
        
        # 层贡献度
        contributions = film_analysis['layer_contribution']
        bars2 = ax2.bar(x_pos, contributions, color='green', alpha=0.7)
        ax2.set_xlabel('FiLM Layer')
        ax2.set_ylabel('Gradient Contribution')
        ax2.set_title(f'FiLM Layer Contribution (Iter {iteration})')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(layers)
        ax2.grid(axis='y', alpha=0.3)
        
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{contributions[i]:.2f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        writer.add_figure('film/layer_metrics', fig, iteration)
        plt.close()
    
    def visualize_spatial_action_correlation(self, writer, iteration: int):
        """可视化空间-动作相关性"""
        print("[Visualization] Generating spatial-action correlation visualizations...")
        
        correlation_analysis = self.analyze_spatial_action_correlation()
        
        # 1. 相关性矩阵热图
        fig, ax = plt.subplots(figsize=(12, 10))
        corr_matrix = correlation_analysis['correlation_matrix']
        im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
        ax.set_xlabel('Control Feature Dimension', fontsize=12)
        ax.set_ylabel('Spatial Feature Dimension', fontsize=12)
        ax.set_title(f'Spatial-Action Feature Correlation Matrix (Iter {iteration})', fontsize=14)
        plt.colorbar(im, ax=ax, label='Pearson Correlation')
        writer.add_figure('fusion/correlation_matrix', fig, iteration)
        plt.close()
        
        # 2. 主效应和交互效应对比
        fig, ax = plt.subplots(figsize=(10, 6))
        effects = ['Position\nEffect', 'Action\nEffect', 'Interaction\nEffect', 'Total\nVariance']
        values = [
            correlation_analysis['position_effect'],
            correlation_analysis['action_effect'],
            correlation_analysis['interaction_effect'],
            correlation_analysis['total_variance']
        ]
        colors = ['steelblue', 'coral', 'green', 'purple']
        
        bars = ax.bar(effects, values, color=colors, alpha=0.7)
        ax.set_ylabel('Variance', fontsize=12)
        ax.set_title(f'Fusion Effect Decomposition (Iter {iteration})', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{values[i]:.4f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        writer.add_figure('fusion/effect_decomposition', fig, iteration)
        plt.close()
    
    # ========================================================================
    # 6. 主分析接口
    # ========================================================================
    
    def analyze_and_log(self, writer, iteration: int, scene=None):
        """
        完整的分析和日志记录流程
        
        Args:
            writer: TensorBoard SummaryWriter
            iteration: 当前训练迭代数
            scene: 场景对象（可选）
        """
        print(f"\n{'='*70}")
        print(f"[TriPlaneFiLMAnalyzer] Analysis at iteration {iteration}")
        print(f"{'='*70}")
        
        try:
            # 1. 空间特征分析
            print("\n[1/5] Analyzing spatial features...")
            spatial_metrics = self.analyze_spatial_features()
            
            writer.add_scalar('spatial/mean_variance', spatial_metrics['mean_variance'], iteration)
            writer.add_scalar('spatial/effective_rank', spatial_metrics['effective_rank'], iteration)
            writer.add_scalar('spatial/diversity_score', spatial_metrics['diversity_score'], iteration)
            writer.add_scalar('spatial/mean_similarity', spatial_metrics['mean_similarity'], iteration)
            
            # 2. Action编码分析
            print("[2/5] Analyzing action encoding...")
            action_metrics = self.analyze_action_encoding()
            
            writer.add_scalar('action/separability', action_metrics['action_separability'], iteration)
            writer.add_scalar('action/effective_rank', action_metrics['embedding_rank'], iteration)
            writer.add_scalar('action/cluster_quality', action_metrics['cluster_quality'], iteration)
            writer.add_scalar('action/mean_similarity', action_metrics['mean_similarity'], iteration)
            writer.add_scalar('action/std_uniformity', action_metrics['std_uniformity'], iteration)
            
            self.visualize_action_space(writer, iteration)
            
            # 3. FiLM调制分析
            print("[3/5] Analyzing FiLM modulation...")
            film_metrics = self.analyze_film_modulation()
            
            for i, strength in enumerate(film_metrics['modulation_strength']):
                writer.add_scalar(f'film/layer_{i+1}_strength', strength, iteration)
            
            for i, contrib in enumerate(film_metrics['layer_contribution']):
                writer.add_scalar(f'film/layer_{i+1}_contribution', contrib, iteration)
            
            # 记录γ和β统计
            for i in range(film_metrics['num_layers']):
                writer.add_scalar(f'film/layer_{i+1}_gamma_mean', 
                                 film_metrics['gamma_stats'][i]['mean'], iteration)
                writer.add_scalar(f'film/layer_{i+1}_beta_std',
                                 film_metrics['beta_stats'][i]['std'], iteration)
            
            self.visualize_film_modulation(writer, iteration)
            
            # 4. 跨模态相关性分析
            print("[4/5] Analyzing cross-modal correlation...")
            correlation_metrics = self.analyze_spatial_action_correlation()
            
            writer.add_scalar('fusion/effectiveness', correlation_metrics['fusion_effectiveness'], iteration)
            writer.add_scalar('fusion/mean_correlation', correlation_metrics['mean_correlation'], iteration)
            writer.add_scalar('fusion/position_effect', correlation_metrics['position_effect'], iteration)
            writer.add_scalar('fusion/action_effect', correlation_metrics['action_effect'], iteration)
            writer.add_scalar('fusion/interaction_effect', correlation_metrics['interaction_effect'], iteration)
            
            self.visualize_spatial_action_correlation(writer, iteration)
            
            # 5. 综合报告
            print("[5/5] Generating summary report...")
            self._print_summary_report(spatial_metrics, action_metrics, 
                                       film_metrics, correlation_metrics, iteration)
            
            print(f"{'='*70}\n")
            
        except Exception as e:
            print(f"[ERROR] Analysis failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _print_summary_report(self, spatial, action, film, fusion, iteration):
        """打印综合分析报告"""
        print("\n" + "="*70)
        print(f"ANALYSIS SUMMARY - Iteration {iteration}")
        print("="*70)
        
        print("\n[Spatial Features]")
        print(f"  Mean Variance:      {spatial['mean_variance']:.6f}")
        print(f"  Effective Rank:     {spatial['effective_rank']}/{spatial['total_dims']}")
        print(f"  Diversity Score:    {spatial['diversity_score']:.4f}")
        print(f"  Mean Similarity:    {spatial['mean_similarity']:.6f}")
        
        print("\n[Action Encoding]")
        print(f"  Separability:       {action['action_separability']:.4f}")
        print(f"  Effective Rank:     {action['embedding_rank']}/{action['total_dims']}")
        print(f"  Cluster Quality:    {action['cluster_quality']:.4f}")
        print(f"  Std Uniformity:     {action['std_uniformity']:.6f}")
        
        print("\n[FiLM Modulation]")
        for i, (strength, contrib) in enumerate(zip(film['modulation_strength'], 
                                                     film['layer_contribution'])):
            gamma_mean = film['gamma_stats'][i]['mean']
            beta_std = film['beta_stats'][i]['std']
            print(f"  Layer {i+1}: Strength={strength:.4f}, Contrib={contrib:.2f}, "
                  f"γ_mean={gamma_mean:.3f}, β_std={beta_std:.3f}")
        
        print("\n[Cross-Modal Fusion]")
        print(f"  Fusion Effectiveness: {fusion['fusion_effectiveness']:.4f}")
        print(f"  Mean Correlation:     {fusion['mean_correlation']:.4f}")
        print(f"  Position Effect:      {fusion['position_effect']:.6f}")
        print(f"  Action Effect:        {fusion['action_effect']:.6f}")
        print(f"  Interaction Effect:   {fusion['interaction_effect']:.6f}")
        
        # 警告信息
        print("\n[Health Check]")
        warnings = []
        
        if action['action_separability'] < 1.5:
            warnings.append("⚠️  Low action separability - actions may be too similar")
        
        if fusion['fusion_effectiveness'] < 0.5:
            warnings.append("⚠️  Poor fusion effectiveness - check FiLM configuration")
        
        if max(film['modulation_strength']) < 0.1:
            warnings.append("⚠️  Weak FiLM modulation - consider increasing learning rate")
        
        if spatial['diversity_score'] < 2.0:
            warnings.append("⚠️  Low spatial feature diversity - possible feature collapse")
        
        if action['embedding_rank'] < action['total_dims'] * 0.5:
            warnings.append("⚠️  Low action embedding rank - underutilized dimensions")
        
        if len(warnings) == 0:
            print("  ✅ All metrics look healthy!")
        else:
            for warning in warnings:
                print(f"  {warning}")
        
        print("="*70 + "\n")
    
    # ========================================================================
    # 7. 辅助方法
    # ========================================================================
    
    def _sample_positions(self, num_samples: int) -> torch.Tensor:
        """采样空间位置"""
        bounds = getattr(self.config, 'bounds', 1.6)
        # 使用均匀分布采样，确保在bounds范围内
        positions = (torch.rand(num_samples, 3, device=self.device) * 2 - 1) * bounds * 0.9
        return positions
    
    def _sample_actions(self, num_samples: int) -> torch.Tensor:
        """采样动作向量"""
        # 使用均匀分布采样，假设action归一化在[-π, π]范围
        # 如果实际范围不同，会通过ControlProcessor的编码器适应
        actions = (torch.rand(num_samples, self.action_dim, device=self.device) * 2 - 1) * 3.0
        return actions
