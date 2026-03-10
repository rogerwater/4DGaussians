#!/usr/bin/env python
"""
将outputs目录中的step_*_best_plan.gif或step_*_executed.png拼接成一个完整的轨迹视频
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from pathlib import Path

def extract_step_number(filename):
    """从文件名中提取步骤数字"""
    match = re.search(r'step_(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1

def load_gif_first_frame(gif_path):
    """加载GIF的第一帧"""
    img = Image.open(gif_path)
    return np.array(img.convert('RGB')) / 255.0

def load_image(img_path):
    """加载PNG图像"""
    img = Image.open(img_path)
    return np.array(img.convert('RGB')) / 255.0

def create_video_from_steps(input_dir, output_path, fps=5, file_pattern='step_*_best_plan.gif'):
    """
    从step文件创建视频
    
    Args:
        input_dir: 输入目录
        output_path: 输出视频路径 (.gif 或 .mp4)
        fps: 帧率
        file_pattern: 文件匹配模式
    """
    print(f"\n{'='*70}")
    print(f"创建轨迹视频")
    print(f"{'='*70}")
    print(f"输入目录: {input_dir}")
    print(f"输出文件: {output_path}")
    print(f"帧率: {fps} FPS")
    print(f"文件模式: {file_pattern}")
    
    # 查找所有匹配的文件
    input_path = Path(input_dir)
    all_files = list(input_path.glob(file_pattern))
    
    if len(all_files) == 0:
        print(f"\n❌ 错误: 未找到匹配 '{file_pattern}' 的文件!")
        return
    
    print(f"\n找到 {len(all_files)} 个文件")
    
    # 按步骤数字排序
    sorted_files = sorted(all_files, key=lambda x: extract_step_number(x.name))
    
    # 加载初始观察（如果存在）
    initial_obs_path = input_path / 'initial_observation.png'
    frames = []
    
    if initial_obs_path.exists():
        print(f"\n加载初始观察: {initial_obs_path.name}")
        initial_frame = load_image(initial_obs_path)
        frames.append(initial_frame)
    
    # 加载所有步骤的图像
    print(f"\n加载步骤图像...")
    for i, file_path in enumerate(sorted_files):
        step_num = extract_step_number(file_path.name)
        
        if file_path.suffix == '.gif':
            frame = load_gif_first_frame(file_path)
        else:
            frame = load_image(file_path)
        
        frames.append(frame)
        if (i + 1) % 5 == 0 or (i + 1) == len(sorted_files):
            print(f"  已加载 {i+1}/{len(sorted_files)} 帧")
    
    print(f"\n总帧数: {len(frames)}")
    
    # 保存为GIF
    if output_path.endswith('.gif'):
        print(f"\n保存为GIF...")
        from mpc.utils import write_moviepy_gif
        write_moviepy_gif(output_path, frames, fps=fps, verbose=True)
        print(f"✓ 已保存: {output_path}")
    
    # 保存为MP4
    if output_path.endswith('.mp4'):
        print(f"\n保存为MP4...")
        try:
            import moviepy.editor as mpy
            # 转换为0-255范围
            frames_255 = [(frame * 255).astype(np.uint8) for frame in frames]
            clip = mpy.ImageSequenceClip(frames_255, fps=fps)
            clip.write_videofile(output_path, codec='libx264', verbose=False, logger=None)
            print(f"✓ 已保存: {output_path}")
        except Exception as e:
            print(f"❌ 保存MP4失败: {e}")
    
    # 创建预览图（前6帧）
    preview_path = output_path.replace('.gif', '_preview.png').replace('.mp4', '_preview.png')
    print(f"\n创建预览图...")
    n_preview = min(6, len(frames))
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for i in range(n_preview):
        axes[i].imshow(frames[i])
        axes[i].axis('off')
        if i == 0 and initial_obs_path.exists():
            axes[i].set_title('Initial', fontsize=10)
        else:
            step_idx = i if not initial_obs_path.exists() else i - 1
            axes[i].set_title(f'Step {step_idx}', fontsize=10)
    
    for i in range(n_preview, 6):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(preview_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存预览: {preview_path}")
    
    print(f"\n{'='*70}")
    print("✓ 完成!")
    print(f"{'='*70}")

def main():
    parser = argparse.ArgumentParser(description='Create trajectory video from step images')
    parser.add_argument('--input_dir', type=str, default='outputs/real_camera_demo',
                        help='Input directory containing step_*.gif or step_*_executed.png')
    parser.add_argument('--output', type=str, default='outputs/real_camera_demo/trajectory_video.gif',
                        help='Output video path (.gif or .mp4)')
    parser.add_argument('--fps', type=int, default=5,
                        help='Frames per second')
    parser.add_argument('--pattern', type=str, default='step_*_best_plan.gif',
                        help='File pattern to match (e.g., step_*_best_plan.gif or step_*_executed.png)')
    args = parser.parse_args()
    
    create_video_from_steps(
        input_dir=args.input_dir,
        output_path=args.output,
        fps=args.fps,
        file_pattern=args.pattern
    )

if __name__ == "__main__":
    main()
