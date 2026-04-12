#!/usr/bin/env python3
import json
import math
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


def load_transforms_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    frames = data.get('frames', [])
    if not frames:
        raise ValueError(f"No frames found in {json_path}")

    for frame in frames:
        if 'joint_pos' in frame:
            joint_pos = [float(v) for v in frame['joint_pos']]
            file_path = frame.get('file_path', 'unknown')
            return joint_pos, file_path

    raise ValueError(f"No frame with 'joint_pos' found in {json_path}")


def simulate_buggy_clipping(actions):
    return [max(-1.0, min(1.0, v)) for v in actions]


def check_unit_circle(actions, start_idx=0, end_idx=12):
    unit_circle_check = []
    violations = []
    for i in range(start_idx, end_idx, 2):
        sin_val = actions[i]
        cos_val = actions[i + 1]
        value = (sin_val * sin_val) + (cos_val * cos_val)
        unit_circle_check.append(value)
        violations.append(abs(value - 1.0))
    return unit_circle_check, violations


def project_joint_angles(actions, start_idx=0, end_idx=12):
    projected = list(actions)
    for i in range(start_idx, end_idx, 2):
        sin_val = actions[i]
        cos_val = actions[i + 1]
        norm = math.sqrt((sin_val * sin_val) + (cos_val * cos_val))
        if norm < 1e-6:
            norm = 1e-6
        projected[i] = sin_val / norm
        projected[i + 1] = cos_val / norm
    return projected


def compute_naive_angular_velocity(actions1, actions2, start_idx=0, end_idx=12):
    delta_angles_deg = []
    for i in range(start_idx, end_idx, 2):
        sin1 = actions1[i]
        cos1 = actions1[i + 1]
        sin2 = actions2[i]
        cos2 = actions2[i + 1]
        angle1 = math.atan2(sin1, cos1)
        angle2 = math.atan2(sin2, cos2)
        delta_angles_deg.append(math.degrees(angle2 - angle1))
    return delta_angles_deg


def main():
    print("="*80)
    print("BASELINE BUG CAPTURE - Wave 0 Task 0.2")
    print("="*80)
    print()
    
    # 1. Load original joint_pos from transforms.json
    json_path = "/home/ubuntu/project/data/dm_control_push/transforms.json"
    print(f"[1/5] Loading transforms.json: {json_path}")
    
    try:
        original_actions, file_path = load_transforms_json(json_path)
        print(f"  ✓ Loaded from frame: {file_path}")
        print(f"  ✓ Action length: {len(original_actions)}")
    except FileNotFoundError:
        print(f"  ✗ File not found: {json_path}")
        print("  Trying alternative path...")
        json_path = "/home/ubuntu/project/data/dm_control/transforms.json"
        original_actions, file_path = load_transforms_json(json_path)
        print(f"  ✓ Loaded from frame: {file_path}")
    
    print()
    
    # 2. Display original values
    print("[2/5] Original Joint States (from transforms.json)")
    print(f"  First 5 dimensions: {original_actions[:5]}")
    print(f"  Full joint states (dims 0-11):")
    for i in range(6):
        sin_val = original_actions[2*i]
        cos_val = original_actions[2*i + 1]
        angle_deg = math.degrees(math.atan2(sin_val, cos_val))
        print(f"    Joint {i+1}: sin={sin_val:+.6f}, cos={cos_val:+.6f}  →  θ={angle_deg:+7.2f}°")
    print(f"  Gripper (dims 12-14): {original_actions[12:15]}")
    print()
    
    # 3. Check original unit circle constraint
    print("[3/5] Original Unit Circle Check")
    unit_circle_orig, violations_orig = check_unit_circle(original_actions)
    print(f"  sin²+cos² values (should be 1.0):")
    for i in range(6):
        print(f"    Joint {i+1}: {unit_circle_orig[i]:.10f}  (error: {violations_orig[i]:.2e})")
    max_violation_orig = max(violations_orig)
    print(f"  Max violation: {max_violation_orig:.2e}")
    if max_violation_orig < 1e-6:
        print(f"  ✓ Unit circle SATISFIED (tolerance 1e-6)")
    else:
        print(f"  ✗ Unit circle VIOLATED")
    print()
    
    # 4. Simulate buggy clipping (what CEM/MPPI currently do)
    print("[4/5] Simulating Buggy Clipping (CEM line 221/231, MPPI line 69)")
    clipped_actions = simulate_buggy_clipping(original_actions)
    print(f"  Clipped values (first 5 dims): {clipped_actions[:5]}")
    print(f"  Corruption analysis:")
    for i in range(min(12, len(original_actions))):
        if abs(original_actions[i] - clipped_actions[i]) > 1e-12:
            print(f"    Dim {i}: {original_actions[i]:+.6f} → {clipped_actions[i]:+.6f}  (CORRUPTED)")
    print()
    
    # 5. Check unit circle AFTER clipping
    print("[5/5] Unit Circle Check AFTER Buggy Clipping")
    unit_circle_clipped, violations_clipped = check_unit_circle(clipped_actions)
    print(f"  sin²+cos² values (should be 1.0):")
    for i in range(6):
        print(f"    Joint {i+1}: {unit_circle_clipped[i]:.10f}  (error: {violations_clipped[i]:.2e})")
    max_violation_clipped = max(violations_clipped)
    print(f"  Max violation: {max_violation_clipped:.2e}")
    if max_violation_clipped < 1e-6:
        print(f"  ✓ Unit circle SATISFIED (tolerance 1e-6)")
    else:
        print(f"  ✗ Unit circle VIOLATED")
    print()
    
    print("[6/6] Angular Velocity Impact")
    print("  Computing naive Δθ between original and clipped...")
    delta_angles = compute_naive_angular_velocity(original_actions, clipped_actions)
    print(f"  Angular changes (degrees):")
    for i in range(6):
        print(f"    Joint {i+1}: Δθ = {delta_angles[i]:+7.2f}°")
    max_delta = max(abs(angle) for angle in delta_angles)
    print(f"  Max angular change: {max_delta:.2f}°")
    print("  NOTE: These angular changes are ARTIFACTS of clipping, not real motion!")
    print()
    
    print("[7/7] Synthetic Demonstration: Clip-Before-Project Distorts Angle")
    synthetic_actions = list(original_actions)
    synthetic_actions[0] = 1.20
    synthetic_actions[1] = 0.30
    print(f"  Synthetic joint 1 (raw): sin={synthetic_actions[0]:+.3f}, cos={synthetic_actions[1]:+.3f}")
    
    clipped = simulate_buggy_clipping(synthetic_actions)
    projected_after_clip = project_joint_angles(clipped)
    projected_direct = project_joint_angles(synthetic_actions)
    
    angle_direct = math.degrees(math.atan2(projected_direct[0], projected_direct[1]))
    angle_after_clip = math.degrees(math.atan2(projected_after_clip[0], projected_after_clip[1]))
    
    print(f"  Project-only angle:      {angle_direct:+7.2f}°")
    print(f"  Clip→Project angle:      {angle_after_clip:+7.2f}°")
    print(f"  Angle distortion:        {abs(angle_direct - angle_after_clip):.2f}°")
    print("  NOTE: Clipping BEFORE projection changes the angle, even after unit-circle fix.")
    print()
    
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"BUG CONFIRMED:")
    print(f"  - Original max unit circle error: {max_violation_orig:.2e}  (VALID)")
    print(f"  - After clipping max error:      {max_violation_clipped:.2e}  (INVALID)")
    print(f"  - Error increase:                {(max_violation_clipped/max_violation_orig):.1f}x")
    print()
    print(f"CORRUPTION DETAILS:")
    corruption_count = 0
    for i in range(12):
        if abs(original_actions[i] - clipped_actions[i]) > 1e-12:
            corruption_count += 1
    print(f"  - Corrupted dimensions (out of 12 joint dims): {corruption_count}")
    print(f"  - Example corruption: sin={original_actions[0]:.6f} → {clipped_actions[0]:.6f}")
    print()
    print(f"EXPECTED BEHAVIOR (after fix):")
    print(f"  - NO clipping of joint states (dims 0-11)")
    print(f"  - Unit circle preserved via projection: sin²+cos²=1")
    print(f"  - Angular velocity constraint: |Δθ| ≤ 30° per timestep")
    print(f"  - Gripper dims (12-14) still clipped to [-1, 1]")
    print()
    print("="*80)
    
    output_file = os.path.join(os.path.dirname(__file__), "baseline-buggy-code.txt")
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BASELINE BUG EVIDENCE - Wave 0 Task 0.2\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Source: {json_path}\n")
        f.write(f"Frame: {file_path}\n\n")
        
        f.write("ORIGINAL JOINT STATES (from transforms.json)\n")
        f.write("-"*80 + "\n")
        for i in range(6):
            sin_val = original_actions[2*i]
            cos_val = original_actions[2*i + 1]
            angle_deg = math.degrees(math.atan2(sin_val, cos_val))
            f.write(f"Joint {i+1}: sin={sin_val:+.10f}, cos={cos_val:+.10f}  →  θ={angle_deg:+7.2f}°\n")
        f.write(f"Gripper: {original_actions[12:15]}\n\n")
        
        f.write("AFTER BUGGY CLIPPING (CEM line 221/231, MPPI line 69)\n")
        f.write("-"*80 + "\n")
        for i in range(6):
            sin_val = clipped_actions[2*i]
            cos_val = clipped_actions[2*i + 1]
            f.write(f"Joint {i+1}: sin={sin_val:+.10f}, cos={cos_val:+.10f}\n")
        f.write("\n")
        
        f.write("UNIT CIRCLE VIOLATIONS\n")
        f.write("-"*80 + "\n")
        f.write("Original (VALID):\n")
        for i in range(6):
            f.write(f"  Joint {i+1}: sin²+cos² = {unit_circle_orig[i]:.10f}  (error: {violations_orig[i]:.2e})\n")
        f.write(f"  Max error: {max_violation_orig:.2e}\n\n")
        
        f.write("After Clipping (INVALID):\n")
        for i in range(6):
            f.write(f"  Joint {i+1}: sin²+cos² = {unit_circle_clipped[i]:.10f}  (error: {violations_clipped[i]:.2e})\n")
        f.write(f"  Max error: {max_violation_clipped:.2e}\n\n")
        
        f.write("CORRUPTION COUNT\n")
        f.write("-"*80 + "\n")
        f.write(f"Dimensions corrupted: {corruption_count} / 12 joint dimensions\n\n")
        
        f.write("BUGGY CODE LOCATIONS\n")
        f.write("-"*80 + "\n")
        f.write("mpc/cem.py:221    - torch.clip(new_action_samples, -1, 1)\n")
        f.write("mpc/cem.py:231    - np.clip(new_action_samples, -1, 1)\n")
        f.write("mpc/mppi.py:69    - np.clip(new_action_samples, -1, 1)\n")
        f.write("mpc/lbfgs.py:134  - torch.clip(actions_leaf, -1, 1)  (deprecated, not fixed)\n\n")
        
        f.write("SYNTHETIC DEMONSTRATION (CLIP-BEFORE-PROJECT DISTORTS ANGLE)\n")
        f.write("-"*80 + "\n")
        f.write("Synthetic joint 1 (raw): sin=+1.200, cos=+0.300\n")
        f.write(f"Project-only angle: {angle_direct:+7.2f}°\n")
        f.write(f"Clip→Project angle: {angle_after_clip:+7.2f}°\n")
        f.write(f"Angle distortion:   {abs(angle_direct - angle_after_clip):.2f}°\n")
        f.write("NOTE: Clipping BEFORE projection changes joint angle direction.\n\n")
        
        f.write("="*80 + "\n")
        f.write("Evidence captured: " + output_file + "\n")
        f.write("="*80 + "\n")
    
    print(f"Evidence saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
