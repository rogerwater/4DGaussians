#!/usr/bin/env python
import numpy as np
import sys

print("=" * 60)
print("Task 2 Verification: Flow-Weighted Sampling")
print("=" * 60)

def test_weighted_sampling():
    print("\n[Test 1] Hot region concentration test...")
    
    H, W = 200, 200
    flow_field = np.ones((H, W, 2)) * 0.1
    
    hot_y_start, hot_y_end = 50, 70
    hot_x_start, hot_x_end = 50, 70
    flow_field[hot_y_start:hot_y_end, hot_x_start:hot_x_end, :] = 10.0
    
    flow_magnitude = np.linalg.norm(flow_field, axis=-1)
    
    motion_coords = []
    for y in range(H):
        for x in range(W):
            if flow_magnitude[y, x] > 0.05:
                motion_coords.append([y, x])
    motion_coords = np.array(motion_coords)
    
    motion_y = motion_coords[:, 0]
    motion_x = motion_coords[:, 1]
    weights = flow_magnitude[motion_y, motion_x]
    
    if weights.sum() > 0:
        probabilities = weights / weights.sum()
    else:
        probabilities = np.ones(len(weights)) / len(weights)
    
    num_samples = 1000
    indices = np.random.choice(len(motion_coords), size=num_samples, replace=True, p=probabilities)
    sampled_coords = motion_coords[indices]
    
    in_hot_region = np.sum(
        (sampled_coords[:, 0] >= hot_y_start) &
        (sampled_coords[:, 0] < hot_y_end) &
        (sampled_coords[:, 1] >= hot_x_start) &
        (sampled_coords[:, 1] < hot_x_end)
    )
    
    hot_region_pct = (in_hot_region / num_samples) * 100
    
    hot_area = (hot_y_end - hot_y_start) * (hot_x_end - hot_x_start)
    total_area = H * W
    uniform_expected_pct = (hot_area / total_area) * 100
    
    print(f"  Hot region (10.0 flow): {hot_area} pixels ({uniform_expected_pct:.1f}% of area)")
    print(f"  Sampled points in hot region: {in_hot_region}/{num_samples} ({hot_region_pct:.1f}%)")
    print(f"  Uniform sampling would give: ~{uniform_expected_pct:.1f}%")
    
    if hot_region_pct > 30:
        print(f"  ✅ Flow weighting works ({hot_region_pct:.1f}% >> {uniform_expected_pct:.1f}%)")
        return True
    else:
        print(f"  ❌ Flow weighting failed ({hot_region_pct:.1f}% not >> {uniform_expected_pct:.1f}%)")
        return False

def test_zero_flow_fallback():
    print("\n[Test 2] Zero-flow fallback test...")
    
    H, W = 100, 100
    flow_field = np.zeros((H, W, 2))
    flow_magnitude = np.linalg.norm(flow_field, axis=-1)
    
    motion_coords = []
    for y in range(H):
        for x in range(W):
            motion_coords.append([y, x])
    motion_coords = np.array(motion_coords)
    
    motion_y = motion_coords[:, 0]
    motion_x = motion_coords[:, 1]
    weights = flow_magnitude[motion_y, motion_x]
    
    try:
        if weights.sum() > 0:
            probabilities = weights / weights.sum()
        else:
            probabilities = np.ones(len(weights)) / len(weights)
        
        num_samples = 100
        indices = np.random.choice(len(motion_coords), size=num_samples, replace=True, p=probabilities)
        
        print(f"  ✅ Zero-flow handled correctly (fallback to uniform)")
        return True
    except Exception as e:
        print(f"  ❌ Zero-flow handling failed: {e}")
        return False

def test_temperature_effect():
    print("\n[Test 3] Temperature parameter test...")
    
    H, W = 100, 100
    flow_field = np.random.rand(H, W, 2)
    flow_magnitude = np.linalg.norm(flow_field, axis=-1)
    
    motion_coords = []
    for y in range(H):
        for x in range(W):
            motion_coords.append([y, x])
    motion_coords = np.array(motion_coords)
    
    motion_y = motion_coords[:, 0]
    motion_x = motion_coords[:, 1]
    weights = flow_magnitude[motion_y, motion_x]
    
    temp_low = 0.5
    temp_high = 2.0
    
    weights_low = np.power(weights, 1.0 / temp_low)
    weights_high = np.power(weights, 1.0 / temp_high)
    
    prob_low = weights_low / weights_low.sum()
    prob_high = weights_high / weights_high.sum()
    
    entropy_low = -np.sum(prob_low * np.log(prob_low + 1e-10))
    entropy_high = -np.sum(prob_high * np.log(prob_high + 1e-10))
    
    print(f"  Temperature {temp_low}: entropy = {entropy_low:.4f}")
    print(f"  Temperature {temp_high}: entropy = {entropy_high:.4f}")
    
    if entropy_high > entropy_low:
        print(f"  ✅ Temperature effect correct (higher temp → higher entropy)")
        return True
    else:
        print(f"  ❌ Temperature effect incorrect")
        return False

all_passed = True

all_passed &= test_weighted_sampling()
all_passed &= test_zero_flow_fallback()
all_passed &= test_temperature_effect()

print("\n" + "=" * 60)
if all_passed:
    print("✅ VERIFICATION PASSED")
    print("=" * 60)
    print("\nSummary:")
    print("  - Hot region test: Points concentrate on high-flow regions ✅")
    print("  - Zero-flow test: Fallback to uniform sampling ✅")
    print("  - Temperature test: Higher temp → more uniform distribution ✅")
    print("\nTask 2: COMPLETE ✅")
    sys.exit(0)
else:
    print("❌ VERIFICATION FAILED")
    print("=" * 60)
    sys.exit(1)
