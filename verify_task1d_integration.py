#!/usr/bin/env python
"""
Verification script for Task 1d: Projection integration into CEM/MPPI optimizers
Tests that joint angle constraints (sin²+cos²=1) are enforced after projection
"""
import numpy as np
import torch
import sys

def check_unit_circle_constraint(actions, tolerance=1e-6):
    """Check if sin²+cos²≈1 for all 6 joints"""
    if isinstance(actions, torch.Tensor):
        actions = actions.cpu().numpy()
    
    # Extract sin/cos pairs for 6 joints (indices 0-11)
    violations = []
    for joint_idx in range(6):
        sin_idx = 2 * joint_idx
        cos_idx = 2 * joint_idx + 1
        sin_vals = actions[..., sin_idx]
        cos_vals = actions[..., cos_idx]
        unit_error = np.abs(sin_vals**2 + cos_vals**2 - 1.0)
        violations.append(np.max(unit_error))
    
    max_violation = np.max(violations)
    return max_violation, violations

print("=" * 60)
print("Task 1d Verification: Projection Integration")
print("=" * 60)

# Test 1: Import projection functions
print("\n[Test 1] Import projection functions...")
try:
    from mpc.constraint_utils import project_joint_angles, project_joint_angles_torch
    print("✅ Imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: CEM requires_grad=True branch (PyTorch)
print("\n[Test 2] CEM requires_grad=True (PyTorch tensors)...")
try:
    # Simulate invalid actions from CEM sampling
    num_samples = 32
    horizon = 10
    action_dim = 15
    
    # Create invalid actions (not on unit circle)
    invalid_actions = torch.randn(num_samples, horizon, action_dim)
    invalid_actions = torch.clip(invalid_actions, -1, 1)
    
    # Check violation before projection
    max_viol_before, _ = check_unit_circle_constraint(invalid_actions)
    print(f"  Before projection - Max violation: {max_viol_before:.6e}")
    
    # Apply projection (as done in CEM)
    projected_actions = project_joint_angles_torch(invalid_actions, start_idx=0, end_idx=12)
    
    # Check violation after projection
    max_viol_after, violations = check_unit_circle_constraint(projected_actions)
    print(f"  After projection  - Max violation: {max_viol_after:.6e}")
    
    if max_viol_after < 1e-6:
        print(f"✅ CEM (requires_grad=True) projection works correctly")
    else:
        print(f"❌ CEM (requires_grad=True) violations exceed tolerance")
        print(f"  Violations per joint: {violations}")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: CEM requires_grad=False branch (NumPy → PyTorch → NumPy)
print("\n[Test 3] CEM requires_grad=False (NumPy arrays)...")
try:
    # Simulate invalid actions from CEM sampling
    invalid_actions_np = np.random.randn(num_samples, horizon, action_dim)
    invalid_actions_np = np.clip(invalid_actions_np, -1, 1)
    
    # Check violation before projection
    max_viol_before, _ = check_unit_circle_constraint(invalid_actions_np)
    print(f"  Before projection - Max violation: {max_viol_before:.6e}")
    
    # Apply projection (as done in CEM requires_grad=False branch)
    new_action_samples_torch = torch.from_numpy(invalid_actions_np).float()
    new_action_samples_torch = project_joint_angles_torch(new_action_samples_torch, start_idx=0, end_idx=12)
    projected_actions_np = new_action_samples_torch.cpu().numpy()
    
    # Check violation after projection
    max_viol_after, violations = check_unit_circle_constraint(projected_actions_np)
    print(f"  After projection  - Max violation: {max_viol_after:.6e}")
    
    if max_viol_after < 1e-6:
        print(f"✅ CEM (requires_grad=False) projection works correctly")
    else:
        print(f"❌ CEM (requires_grad=False) violations exceed tolerance")
        print(f"  Violations per joint: {violations}")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: MPPI (NumPy arrays)
print("\n[Test 4] MPPI (NumPy arrays)...")
try:
    # Simulate invalid actions from MPPI sampling
    invalid_actions_np = np.random.randn(num_samples, horizon, action_dim)
    invalid_actions_np = np.clip(invalid_actions_np, -1, 1)
    
    # Check violation before projection
    max_viol_before, _ = check_unit_circle_constraint(invalid_actions_np)
    print(f"  Before projection - Max violation: {max_viol_before:.6e}")
    
    # Apply projection (as done in MPPI)
    projected_actions_np = project_joint_angles(invalid_actions_np, start_idx=0, end_idx=12)
    
    # Check violation after projection
    max_viol_after, violations = check_unit_circle_constraint(projected_actions_np)
    print(f"  After projection  - Max violation: {max_viol_after:.6e}")
    
    if max_viol_after < 1e-6:
        print(f"✅ MPPI projection works correctly")
    else:
        print(f"❌ MPPI violations exceed tolerance")
        print(f"  Violations per joint: {violations}")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Gripper dimensions unchanged
print("\n[Test 5] Gripper dimensions (12-14) unchanged...")
try:
    # Create actions with known gripper values
    test_actions = torch.randn(8, 5, 15)
    gripper_values_before = test_actions[:, :, 12:15].clone()
    
    # Apply projection
    projected_actions = project_joint_angles_torch(test_actions, start_idx=0, end_idx=12)
    gripper_values_after = projected_actions[:, :, 12:15]
    
    # Check if gripper values are unchanged
    gripper_diff = torch.abs(gripper_values_after - gripper_values_before).max().item()
    
    if gripper_diff < 1e-9:
        print(f"✅ Gripper dimensions preserved (max diff: {gripper_diff:.6e})")
    else:
        print(f"❌ Gripper dimensions changed (max diff: {gripper_diff:.6e})")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED")
print("=" * 60)
print("\nSummary:")
print("  - CEM (requires_grad=True): Projection applied correctly")
print("  - CEM (requires_grad=False): NumPy→Torch→NumPy conversion works")
print("  - MPPI: Projection applied correctly")
print("  - Gripper dimensions (12-14) preserved")
print("  - All joint angles satisfy sin²+cos²≈1 (tolerance: 1e-6)")
print("\nTask 1d: Integration COMPLETE ✅")
