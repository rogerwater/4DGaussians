#!/bin/bash
# 快速测试render_based光流方法

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Render-Based Flow MPC Quick Start${NC}"
echo -e "${GREEN}======================================${NC}"

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

# 配置参数（用户需要修改这些）
echo -e "\n${YELLOW}[2/5] Configuration${NC}"

MODEL_PATH="${1:-/home/ubuntu/yyf/4DGaussians/outputs/dm_control_push_test_flow2/point_cloud/iteration_12000}"  # 第一个参数或默认值
INITIAL_IMAGE="${2:-/home/ubuntu/yyf/4DGaussians/assets/start-end/cam5_sample1_frame_00001.jpg}"
TARGET_IMAGE="${3:-/home/ubuntu/yyf/4DGaussians/assets/start-end/cam5_sample1_frame_00018.jpg}"
CONTROL_DIM="${4:-15}"
DEVICE="${5:-cuda:2}"
TRANSFORMS_JSON="${6:-/home/ubuntu/yyf/4DGaussians/assets/example_transforms.json}"

# 移除旧的USE_INITIAL_CONTROL参数（现在从transforms_json自动读取joint_pos）

# 检查模型和图像
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}✗ Model path not found: $MODEL_PATH${NC}"
    echo -e "${YELLOW}Usage: $0 <model_path> <initial_image> <target_image> [control_dim] [device]${NC}"
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
LOG_DIR="outputs/render_based_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo -e "\n${YELLOW}[3/5] Running MPC with render_based flow...${NC}"
echo "  Log directory: $LOG_DIR"

# 运行MPC
# - 初始joint_pos从transforms_json的frames中自动读取
# - 第一步渲染初始状态，后续步骤使用CEM规划
COMMON_ARGS=(
    --model_path "$MODEL_PATH"
    --initial_image "$INITIAL_IMAGE"
    --target_image "$TARGET_IMAGE"
    --control_dim "$CONTROL_DIM"
    --num_steps 20
    --horizon 5
    --num_samples 32
    --opt_iters 10
    --device "$DEVICE"
    --log_dir "$LOG_DIR"
    --vgg_weight 0.5
    --vgg_layer relu3_3
    --sampling_strategy adaptive
    --image_height 480
    --image_width 480
    --transforms_json "$TRANSFORMS_JSON"
    --direction_weight 0.5
    --direction_loss_type cosine
    --action_regularization_weight 0.3
    --max_action_delta 0.25
    --use_motion_mask
    --motion_threshold_percentile 60.0
    --save_video
)

echo "  Mode: CEM from initial state (joint_pos loaded from transforms.json)"
python3 demo_flow_guided_mpc.py "${COMMON_ARGS[@]}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n${GREEN}[4/5] ✓ Execution completed successfully${NC}"
    
    echo -e "\n${YELLOW}[5/5] Output files:${NC}"
    echo "  Directory: $LOG_DIR"
    ls -lh "$LOG_DIR"/*.png 2>/dev/null | head -10
    
    echo -e "\n${GREEN}======================================${NC}"
    echo -e "${GREEN}Quick test completed!${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo -e "\nTo view results:"
    echo -e "  cd $LOG_DIR"
    echo -e "  eog *.png  # View images"
    echo -e "\nFor full run, edit the script or use:"
    echo -e "  python3 demo_flow_guided_mpc.py --model_path ... --transforms_json ... --num_steps 20 --horizon 10"
else
    echo -e "\n${RED}[4/5] ✗ Execution failed with code $EXIT_CODE${NC}"
    echo -e "${YELLOW}Check the error messages above${NC}"
    exit $EXIT_CODE
fi
