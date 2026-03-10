#!/bin/bash
# 完整实验配置 - 用于正式实验

MODEL_PATH="./assets/iteration_5000"
ITERATION=5000
TARGET_IMAGE="./assets/start-end/frame_00045.jpg"

# 完整实验参数
HORIZON=10
NUM_SAMPLES=64
NUM_STEPS=20

FLOW_WEIGHT=0.7
IMAGE_WEIGHT=0.3
USE_SPARSE_RENDER=false

LOG_DIR="./logs/flow_mpc_full_experiment"
SAVE_VIDEO=true
VERBOSE=true
