#!/usr/bin/env python3
"""
将outputs/real_camera_demo中的step图片拼接成视频
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob

def create_video_from_steps(output_dir, fps=5):
    """
    从rendered_step_*.png图片创建视频
    
    Args:
        output_dir: 包含渲染图片的目录
        fps: 帧率
    """
    # 查找所有rendered_step图片
    image_files = []
    
    # 查找所有rendered_step_XXXX.png文件
    for step_file in glob.glob(os.path.join(output_dir, "rendered_step_*.png")):
        # 从文件名提取步数: rendered_step_0000.png -> 0
        basename = os.path.basename(step_file)
        try:
            # 提取步数: rendered_step_0001.png -> 1
            step_str = basename.replace('rendered_step_', '').replace('.png', '')
            step_num = int(step_str)
            image_files.append((step_num, step_file))
        except (IndexError, ValueError):
            print(f"警告: 无法解析文件名 {basename}")
            continue
    
    # 按步数排序
    image_files.sort(key=lambda x: x[0])
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    if len(image_files) == 0:
        print("错误：未找到任何图像文件")
        return
    
    # 读取所有图像
    frames = []
    for step_num, filepath in image_files:
        try:
            # 使用PIL加载图像（更可靠）
            from PIL import Image as PILImage
            img = PILImage.open(filepath)
            frame = np.array(img.convert('RGB'))
            
            # 确保是uint8格式
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            frames.append(frame)
            print(f"  加载 Step {step_num}: {os.path.basename(filepath)} - shape {frame.shape}")
        except Exception as e:
            print(f"  警告：加载 {filepath} 失败: {e}")
    
    if len(frames) == 0:
        print("错误：无法加载任何图像")
        return
    
    print(f"\n总共加载 {len(frames)} 帧")
    
    # 统一图像尺寸（使用第一帧的尺寸）
    if len(frames) > 0:
        target_shape = frames[0].shape[:2]
    else:
        target_shape = (256, 256)  # 默认尺寸
    
    print(f"统一图像尺寸为: {target_shape}")
    
    # Resize所有图像
    resized_frames = []
    for i, frame in enumerate(frames):
        if frame.shape[:2] != target_shape:
            from PIL import Image as PILImage
            # 确保frame是uint8
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            # 处理RGBA -> RGB
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]
            
            img = PILImage.fromarray(frame)
            img_resized = img.resize((target_shape[1], target_shape[0]), PILImage.LANCZOS)
            frame = np.array(img_resized)
            print(f"  调整第{i}帧尺寸: {frames[i].shape[:2]} -> {target_shape}")
        else:
            # 即使尺寸相同，也要确保是RGB (not RGBA)
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]
        resized_frames.append(frame)
    
    frames = resized_frames
    
    # 保存为GIF
    try:
        import moviepy.editor as mpy
        gif_path = os.path.join(output_dir, "trajectory_from_steps.gif")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(gif_path) if os.path.dirname(gif_path) else output_dir, exist_ok=True)
        
        # 使用moviepy直接保存（避免mpc.utils的参数问题）
        frames_uint8 = []
        for frame in frames:
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            frames_uint8.append(frame)
        
        clip = mpy.ImageSequenceClip(frames_uint8, fps=fps)
        clip.write_gif(gif_path, fps=fps, logger=None)
        print(f"\n✓ 已保存GIF: {gif_path}")
    except Exception as e:
        import traceback
        print(f"保存GIF失败: {e}")
        print(traceback.format_exc())
    
    # 保存为MP4
    try:
        import moviepy.editor as mpy
        
        # 确保frames是uint8
        frames_uint8 = []
        for frame in frames:
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            frames_uint8.append(frame)
        
        clip = mpy.ImageSequenceClip(frames_uint8, fps=fps)
        mp4_path = os.path.join(output_dir, "trajectory_from_steps.mp4")
        clip.write_videofile(mp4_path, codec='libx264', verbose=False, logger=None)
        print(f"✓ 已保存MP4: {mp4_path}")
    except Exception as e:
        print(f"保存MP4失败: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='从rendered_step图片创建视频')
    parser.add_argument('--output_dir', type=str, 
                        default='./outputs/flow_guided_mpc',
                        help='包含rendered_step_*.png图片的目录')
    parser.add_argument('--fps', type=int, default=5,
                        help='视频帧率')
    
    args = parser.parse_args()
    
    create_video_from_steps(args.output_dir, args.fps)
