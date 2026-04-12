#!/bin/bash
# 快速测试render_based光流方法 - 支持CEM-GD优化器
# Quick test script for render-based flow method - supports CEM-GD optimizer

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查必要文件
echo -e "\n${YELLOW}[1/5] Checking dependencies...${NC}"

if [ ! -d "gmflow" ]; then
    echo -e "${RED}✗ gmflow directory not found${NC}"
    exit 1
fi

if [ ! -f "gmflow/checkpoints/gmflow_sintel-0c07dcb3.pth" ]; then
    echo -e "${RED}✗ GMFlow checkpoint not found${NC}"
    echo -e "  Expected: gmflow/checkpoints/gmflow_sintel-0c07dcb3.pth"
    exit 1
fi

echo -e "${GREEN}✓ Dependencies OK${NC}"

# ============================================================================
# 配置参数 - 用户可以在此处修改参数
# Configuration - Modify parameters here
# ============================================================================
echo -e "\n${YELLOW}[2/5] Configuration${NC}"

# 必需参数 (Required)
MODEL_PATH="${1:-/home/ubuntu/yyf/4DGaussians/outputs/dm_control_push8/point_cloud/iteration_20000}"
INITIAL_IMAGE="${2:-/home/ubuntu/project/data/dm_control_push/cam06/frame_00001.jpg}"
TARGET_IMAGE="${3:-/home/ubuntu/project/data/dm_control_push/cam06/frame_00050.jpg}"
TRANSFORMS_JSON="${4:-/home/ubuntu/project/data/dm_control_push/transforms.json}"

# MPC 参数 (MPC Parameters)
CONTROL_DIM="${CONTROL_DIM:-15}"
NUM_STEPS="${NUM_STEPS:-25}"
HORIZON="${HORIZON:-5}"
DEVICE="${DEVICE:-cuda:2}"

# 优化器参数 (Optimizer Parameters)
OPTIMIZER="${OPTIMIZER:-cem}"        # 可选: cem 或 cem-gd (Options: cem or cem-gd)

# CEM 参数 (用于 --optimizer cem) (CEM parameters for --optimizer cem)
NUM_SAMPLES="${NUM_SAMPLES:-32}"      # CEM 采样数 (CEM samples)
OPT_ITERS="${OPT_ITERS:-10}"            # CEM 迭代次数 (CEM iterations)

# CEM-GD 参数 (用于 --optimizer cem-gd) (CEM-GD parameters for --optimizer cem-gd)
NUM_SAMPLES_INIT="${NUM_SAMPLES_INIT:-32}"       # 初始规划采样数 (Initial planning samples)
NUM_SAMPLES_REPLAN="${NUM_SAMPLES_REPLAN:-16}"   # 重规划采样数 (Replanning samples)
NUM_GRAD_SEQS="${NUM_GRAD_SEQS:-3}"               # 梯度优化序列数 (Gradient refinement sequences)
GRAD_LR="${GRAD_LR:-0.01}"                        # 梯度学习率 (Gradient learning rate)
GRAD_STEPS="${GRAD_STEPS:-15}"                    # 梯度下降步数 (Gradient descent steps)
GRADIENT_DEVICE="${GRADIENT_DEVICE:-cuda:1}"            # 梯度优化设备 (Gradient device, e.g., 'cuda:2')

# 点采样参数 (Point Sampling Parameters)
SAMPLING_METHOD="${SAMPLING_METHOD:-motion_mask}"  # 可选: sobel_hybrid, shi_tomasi, combined, texture, grid, motion_mask
RESAMPLE_MOTION_MASK="${RESAMPLE_MOTION_MASK:-true}"  # 每步重采样运动掩码 (Re-sample motion mask per step)

# 其他参数 (Other Parameters)
IMAGE_HEIGHT="${IMAGE_HEIGHT:-480}"
IMAGE_WIDTH="${IMAGE_WIDTH:-480}"
ACTION_LIMIT="${ACTION_LIMIT:-1.0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/cemgd-push8-traj0-full}"

# ============================================================================

# 打印配置 (Print configuration)
echo -e "${GREEN}Optimizer: ${OPTIMIZER}${NC}"
if [ "$OPTIMIZER" = "cem-gd" ]; then
    echo -e "  CEM-GD Parameters:"
    echo -e "    Initial samples:  ${NUM_SAMPLES_INIT}"
    echo -e "    Replan samples:   ${NUM_SAMPLES_REPLAN}"
    echo -e "    Gradient seqs:    ${NUM_GRAD_SEQS}"
    echo -e "    Gradient LR:      ${GRAD_LR}"
    echo -e "    Gradient steps:   ${GRAD_STEPS}"
else
    echo -e "  CEM Parameters:"
    echo -e "    Samples:          ${NUM_SAMPLES}"
    echo -e "    Opt iterations:   ${OPT_ITERS}"
fi
echo -e "MPC Steps:          ${NUM_STEPS}"
echo -e "Horizon:            ${HORIZON}"
echo -e "Sampling method:    ${SAMPLING_METHOD}"
echo -e "Output directory:   ${OUTPUT_DIR}"

# 检查模型和图像
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}✗ Model path not found: $MODEL_PATH${NC}"
    echo -e "${YELLOW}Usage: $0 <model_path> <initial_image> <target_image> <transforms_json>${NC}"
    exit 1
fi

if [ ! -f "$INITIAL_IMAGE" ]; then
    echo -e "${RED}✗ Initial image not found: $INITIAL_IMAGE${NC}"
    exit 1
fi

if [ ! -f "$TARGET_IMAGE" ]; then
    echo -e "${RED}✗ Target image not found: $TARGET_IMAGE${NC}"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo -e "\n${YELLOW}[2/5] Verifying configuration...${NC}"
echo "  Model path exists: $([ -d "$MODEL_PATH" ] && echo "✓" || echo "✗ NOT FOUND")"
echo "  Initial image exists: $([ -f "$INITIAL_IMAGE" ] && echo "✓" || echo "✗ NOT FOUND")"
echo "  Target image exists: $([ -f "$TARGET_IMAGE" ] && echo "✓" || echo "✗ NOT FOUND")"
echo "  Device to use: $DEVICE"
echo "  Optimizer: $OPTIMIZER"

echo -e "\n${YELLOW}[3/5] Running MPC with ${OPTIMIZER} optimizer...${NC}"
echo "  Log directory: $OUTPUT_DIR"

# 构建命令参数
COMMON_ARGS=(
    --model_path "$MODEL_PATH"
    --initial_image "$INITIAL_IMAGE"
    --target_image "$TARGET_IMAGE"
    --control_dim "$CONTROL_DIM"
    --num_steps "$NUM_STEPS"
    --horizon "$HORIZON"
    --device "$DEVICE"
    --output_dir "$OUTPUT_DIR"
    --image_height "$IMAGE_HEIGHT"
    --image_width "$IMAGE_WIDTH"
    --transforms_json "$TRANSFORMS_JSON"
    --sampling_method "$SAMPLING_METHOD"
    --action_limit "$ACTION_LIMIT"
    --optimizer "$OPTIMIZER"
)

# 添加 CEM 或 CEM-GD 特定参数
if [ "$OPTIMIZER" = "cem-gd" ]; then
    COMMON_ARGS+=(
        --num_samples_init "$NUM_SAMPLES_INIT"
        --num_samples_replan "$NUM_SAMPLES_REPLAN"
        --num_grad_seqs "$NUM_GRAD_SEQS"
        --grad_lr "$GRAD_LR"
        --grad_steps "$GRAD_STEPS"
        --opt_iters "$OPT_ITERS"
    )
    if [ -n "$GRADIENT_DEVICE" ]; then
        COMMON_ARGS+=(--gradient_device "$GRADIENT_DEVICE")
    fi
else
    COMMON_ARGS+=(
        --num_samples "$NUM_SAMPLES"
        --opt_iters "$OPT_ITERS"
    )
fi

# 添加重采样标志
if [ "$RESAMPLE_MOTION_MASK" = "true" ]; then
    COMMON_ARGS+=(--resample_motion_mask_per_step)
fi

# 运行MPC
python3 test/integration/test_cotracker_mpc.py "${COMMON_ARGS[@]}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n${GREEN}[4/5] ✓ Execution completed successfully${NC}"
    

    echo -e "\n${YELLOW}[5/5] Output files:${NC}"
    echo "  Directory: $OUTPUT_DIR"
    ls -lh "$OUTPUT_DIR"/*.png 2>/dev/null | head -10
    
    echo -e "\n${GREEN}======================================${NC}"
    echo -e "${GREEN}Quick test completed!${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo -e "\nTo view results:"
    echo -e "  cd $OUTPUT_DIR"
    echo -e "  eog *.png  # View images"
else
    echo -e "\n${RED}[4/5] ✗ Execution failed with code $EXIT_CODE${NC}"
    echo -e "${YELLOW}Check the error messages above${NC}"
    exit $EXIT_CODE
fi
