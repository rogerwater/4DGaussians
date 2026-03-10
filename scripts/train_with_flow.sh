#!/bin/bash

# =============================================================================
# 4DGaussians Training Script with Flow Supervision and TriPlane Support
# 支持光流监督和TriPlane架构的训练脚本
# =============================================================================

# ======================== GPU配置 ========================
export CUDA_VISIBLE_DEVICES=2

# ======================== 路径配置 ========================
SOURCE_PATH="/home/ubuntu/project/data/toyarm_tiny"
BASE_OUTPUT_DIR="/home/ubuntu/yyf/outputs"

# 生成时间戳（格式：YYYYMMDD_HHMMSS）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 使用时间戳的模型路径
MODEL_PATH="${BASE_OUTPUT_DIR}/flow_${TIMESTAMP}"

# ======================== 基础训练参数 ========================
ITERATIONS=15000
COARSE_ITERATIONS=3000
BATCH_SIZE=4

# ======================== 架构选择 ========================
# 是否使用TriPlane架构 (默认使用HexPlane)
USE_TRIPLANE=false  # true: 使用TriPlane架构, false: 使用HexPlane架构

if [ "$USE_TRIPLANE" = true ]; then
    TRIPLANE_FLAG="--use_triplane"
    echo "架构: TriPlane"
else
    TRIPLANE_FLAG=""
    echo "架构: HexPlane (默认)"
fi

# ======================== 光流监督配置 ========================
# 光流监督开关
USE_FLOW_LOSS=false  # true: 启用光流监督, false: 禁用光流监督

if [ "$USE_FLOW_LOSS" = true ]; then
    FLOW_LOSS_FLAG="--use_flow_loss"
    LAMBDA_FLOW=0.005                # 光流loss权重
    FLOW_START_ITER=3000             # 从第3000次迭代开始使用光流
    FLOW_VIS_INTERVAL=1000           # 光流可视化间隔（每1000次迭代保存一次）
    FLOW_VIS_DIR="${MODEL_PATH}/flow_vis"  # 光流可视化目录
    echo "光流监督: 启用 (λ=${LAMBDA_FLOW}, 起始迭代=${FLOW_START_ITER})"
else
    FLOW_LOSS_FLAG=""
    LAMBDA_FLOW=""
    FLOW_START_ITER=""
    FLOW_VIS_INTERVAL=""
    FLOW_VIS_DIR=""
    echo "光流监督: 禁用"
fi

# ======================== Control Encoder配置 ========================
# Control signal相关参数
CONTROL_INPUT_DIM=6              # 控制输入维度（默认6: [vx,vy,vz,wx,wy,wz]）
CONTROL_HIDDEN_DIM=64            # 控制编码器隐藏层维度
CONTROL_USE_PE=false             # 是否对控制信号使用位置编码
CONTROL_NUM_FREQUENCIES=4        # 位置编码频率数量（当control_use_pe=true时使用）

# ======================== FiLM Fusion配置 ========================
# FiLM (Feature-wise Linear Modulation) 相关参数
FILM_HIDDEN_DIM=64               # FiLM条件网络隐藏层维度
FILM_USE_RESIDUAL=true           # FiLM块中是否使用残差连接

if [ "$FILM_USE_RESIDUAL" = true ]; then
    FILM_RESIDUAL_FLAG="--film_use_residual"
else
    FILM_RESIDUAL_FLAG=""
fi

# ======================== 其他高级参数 ========================
# Deformation Network配置
NET_WIDTH=64                     # 变形MLP宽度
DEFOR_DEPTH=1                    # 变形MLP深度

# 网格相关参数
PLANE_TV_WEIGHT=0.0001           # 空间网格TV loss权重
TIME_SMOOTHNESS_WEIGHT=0.01      # 时间网格TV loss权重
L1_TIME_PLANES=0.0001            # 时间平面L1正则化权重

# 密集化参数
DENSIFY_GRAD_THRESHOLD=0.0002    # 密集化梯度阈值
DENSIFICATION_INTERVAL=100       # 密集化间隔
OPACITY_RESET_INTERVAL=3000      # 不透明度重置间隔

# ======================== 创建输出目录 ========================
mkdir -p ${MODEL_PATH}

if [ "$USE_FLOW_LOSS" = true ]; then
    mkdir -p ${FLOW_VIS_DIR}
fi

# ======================== 打印配置信息 ========================
echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                    4DGaussians 训练配置                                     ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 数据路径:        ${SOURCE_PATH}"
echo "💾 模型保存路径:    ${MODEL_PATH}"
echo "🔧 架构选择:        $([ "$USE_TRIPLANE" = true ] && echo "TriPlane" || echo "HexPlane (默认)")"
echo "🌊 光流监督:        $([ "$USE_FLOW_LOSS" = true ] && echo "启用 (λ=${LAMBDA_FLOW})" || echo "禁用")"
if [ "$USE_FLOW_LOSS" = true ]; then
    echo "   └─ 可视化目录:   ${FLOW_VIS_DIR}"
    echo "   └─ 起始迭代:     ${FLOW_START_ITER}"
    echo "   └─ 可视化间隔:   ${FLOW_VIS_INTERVAL}"
fi
echo "🔄 迭代次数:        粗阶段=${COARSE_ITERATIONS}, 精细阶段=${ITERATIONS}"
echo "📦 批量大小:        ${BATCH_SIZE}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ======================== 构建训练命令 ========================
CMD="python train.py \
    -s ${SOURCE_PATH} \
    -m ${MODEL_PATH} \
    --iterations ${ITERATIONS} \
    --coarse_iterations ${COARSE_ITERATIONS} \
    --batch_size ${BATCH_SIZE} \
    --eval"

# 添加架构选择
if [ "$USE_TRIPLANE" = true ]; then
    CMD="${CMD} ${TRIPLANE_FLAG}"
fi

# 添加光流监督相关参数
if [ "$USE_FLOW_LOSS" = true ]; then
    CMD="${CMD} ${FLOW_LOSS_FLAG} \
    --lambda_flow ${LAMBDA_FLOW} \
    --flow_loss_start_iter ${FLOW_START_ITER} \
    --flow_vis_interval ${FLOW_VIS_INTERVAL} \
    --flow_vis_dir ${FLOW_VIS_DIR}"
fi

# 添加Control Encoder参数
CMD="${CMD} \
    --control_input_dim ${CONTROL_INPUT_DIM} \
    --control_hidden_dim ${CONTROL_HIDDEN_DIM} \
    --control_num_frequencies ${CONTROL_NUM_FREQUENCIES}"

if [ "$CONTROL_USE_PE" = true ]; then
    CMD="${CMD} --control_use_pe"
fi

# 添加FiLM参数
CMD="${CMD} \
    --film_hidden_dim ${FILM_HIDDEN_DIM}"

if [ "$FILM_USE_RESIDUAL" = true ]; then
    CMD="${CMD} ${FILM_RESIDUAL_FLAG}"
fi

# 添加网络和网格参数
CMD="${CMD} \
    --net_width ${NET_WIDTH} \
    --defor_depth ${DEFOR_DEPTH} \
    --plane_tv_weight ${PLANE_TV_WEIGHT} \
    --time_smoothness_weight ${TIME_SMOOTHNESS_WEIGHT} \
    --l1_time_planes ${L1_TIME_PLANES} \
    --densify_grad_threshold_fine_init ${DENSIFY_GRAD_THRESHOLD} \
    --densification_interval ${DENSIFICATION_INTERVAL} \
    --opacity_reset_interval ${OPACITY_RESET_INTERVAL}"

# ======================== 执行训练 ========================
echo "🚀 开始训练..."
echo ""

eval ${CMD}

# ======================== 训练完成 ========================
echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                         训练完成！                                          ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "✅ 模型已保存至: ${MODEL_PATH}"
if [ "$USE_FLOW_LOSS" = true ]; then
    echo "✅ 光流可视化已保存至: ${FLOW_VIS_DIR}"
fi
echo ""
