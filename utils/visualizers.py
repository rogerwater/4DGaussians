"""
可视化工具
用于生成TensorBoard图像
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns


def plot_similarity_matrix(sim_matrix: np.ndarray, max_size: int = 100) -> np.ndarray:
    """
    绘制相似度矩阵热图
    
    Args:
        sim_matrix: [N, N] 相似度矩阵
        max_size: 最大显示大小 (太大会很慢)
        
    Returns:
        image: [H, W, 3] RGB图像
    """
    # 下采样 (如果太大)
    if sim_matrix.shape[0] > max_size:
        step = sim_matrix.shape[0] // max_size
        sim_matrix = sim_matrix[::step, ::step]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 使用seaborn绘制热图
    sns.heatmap(
        sim_matrix,
        cmap='RdYlBu_r',
        center=0.5,
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Cosine Similarity'},
        ax=ax
    )
    
    ax.set_title('Feature Similarity Matrix')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Sample Index')
    
    # 转换为numpy数组
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return image


def plot_scree(singular_values: np.ndarray) -> np.ndarray:
    """
    绘制奇异值谱图 (Scree Plot)
    
    Args:
        singular_values: [D] 奇异值 (已排序)
        
    Returns:
        image: [H, W, 3] RGB图像
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 左图: 奇异值曲线
    ax1.plot(singular_values, 'o-', linewidth=2, markersize=4)
    ax1.set_xlabel('Component Index')
    ax1.set_ylabel('Singular Value (Normalized)')
    ax1.set_title('Scree Plot')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 右图: 累积解释方差
    cumsum = np.cumsum(singular_values)
    ax2.plot(cumsum, 'o-', linewidth=2, markersize=4, color='orange')
    ax2.axhline(y=0.9, color='r', linestyle='--', label='90% variance')
    ax2.axhline(y=0.95, color='g', linestyle='--', label='95% variance')
    ax2.set_xlabel('Component Index')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Cumulative Variance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 转换为numpy数组
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return image


def plot_plane_contributions(contributions: dict) -> np.ndarray:
    """
    绘制平面贡献度雷达图
    
    Args:
        contributions: 平面贡献度字典
        
    Returns:
        image: [H, W, 3] RGB图像
    """
    plane_names = ['XY', 'XZ', 'YZ', 'XT', 'YT', 'ZT']
    
    # 提取各平面的归一化贡献度
    values = []
    labels = []
    for name in plane_names:
        key = f'plane_{name}_normalized'
        if key in contributions:
            values.append(contributions[key])
            labels.append(name)
        elif f'plane_{name}' in contributions:
            values.append(contributions[f'plane_{name}'])
            labels.append(name)
    
    if not values:
        # 如果没有数据，返回空白图
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.text(0.5, 0.5, 'No plane contribution data', 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return image
    
    # 归一化 (如果还没归一化)
    values = np.array(values)
    if values.sum() > 1.1 or values.sum() < 0.9:
        values = values / (values.sum() + 1e-10)
    
    # 雷达图需要首尾相连
    values = np.concatenate([values, [values[0]]])
    
    # 角度
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # 绘制
    ax.plot(angles, values, 'o-', linewidth=2, color='blue')
    ax.fill(angles, values, alpha=0.25, color='blue')
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    
    # 设置范围
    ax.set_ylim(0, max(values) * 1.1)
    
    # 标题
    ax.set_title('HexPlane Contribution', fontsize=14, pad=20)
    
    # 网格
    ax.grid(True)
    
    # 转换为numpy数组
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return image


def plot_tsne(features: np.ndarray, labels: np.ndarray, n_samples: int = 1000) -> np.ndarray:
    """
    绘制t-SNE降维可视化
    
    Args:
        features: [N, D] 特征
        labels: [N] 标签 (不同控制)
        n_samples: 采样数量 (t-SNE很慢)
        
    Returns:
        image: [H, W, 3] RGB图像
    """
    from sklearn.manifold import TSNE
    
    # 采样
    if features.shape[0] > n_samples:
        indices = np.random.choice(features.shape[0], n_samples, replace=False)
        features = features[indices]
        labels = labels[indices]
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    
    # 绘制
    fig, ax = plt.subplots(figsize=(10, 8))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[colors[i]],
            label=f'Control {label}',
            alpha=0.6,
            s=20
        )
    
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title('Feature t-SNE Visualization (colored by control)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 转换为numpy数组
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return image


def plot_spatial_heatmap(
    features: np.ndarray, 
    positions: np.ndarray, 
    z_slice: float = 0.0
) -> np.ndarray:
    """
    绘制空间特征热图 (Z切片)
    
    Args:
        features: [N, D] 特征
        positions: [N, 3] 位置
        z_slice: Z切片位置
        
    Returns:
        image: [H, W, 3] RGB图像
    """
    # 选择接近z_slice的点
    z_tolerance = 0.1
    mask = np.abs(positions[:, 2] - z_slice) < z_tolerance
    
    if mask.sum() < 10:
        # 如果没有足够的点，返回空白图
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f'Not enough points at z={z_slice}', 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return image
    
    selected_positions = positions[mask]
    selected_features = features[mask]
    
    # 特征强度 (范数)
    feature_norm = np.linalg.norm(selected_features, axis=1)
    
    # 绘制散点图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(
        selected_positions[:, 0],
        selected_positions[:, 1],
        c=feature_norm,
        cmap='viridis',
        s=50,
        alpha=0.6
    )
    
    plt.colorbar(scatter, ax=ax, label='Feature Norm')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Feature Activation at Z={z_slice:.2f}')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 转换为numpy数组
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return image
