#!/usr/bin/env python
import numpy as np
import torch
import sys
import re

print("=" * 60)
print("Task 1 Verification: Joint Angle Constraint Enforcement")
print("=" * 60)

def check_unit_circle(actions, tolerance=1e-6):
    if isinstance(actions, torch.Tensor):
        actions = actions.cpu().numpy()
    
    violations = []
    for joint_idx in range(6):
        sin_idx = 2 * joint_idx
        cos_idx = 2 * joint_idx + 1
        sin_vals = actions[..., sin_idx]
        cos_vals = actions[..., cos_idx]
        unit_error = np.abs(sin_vals**2 + cos_vals**2 - 1.0)
        violations.append(np.max(unit_error))
    
    return np.max(violations), violations

all_passed = True

print("\n[Test 1] Config class import and parameters...")
try:
    from arguments.planning_dmcontrol import PlanningDMControlParams
    
    config = PlanningDMControlParams(parser=None, sentinel=True)
    
    assert hasattr(config, 'constraint_tolerance'), "Missing constraint_tolerance"
    assert hasattr(config, 'unit_circle_penalty_weight'), "Missing unit_circle_penalty_weight"
    assert hasattr(config, 'enable_projection'), "Missing enable_projection"
    assert hasattr(config, 'enable_penalty'), "Missing enable_penalty"
    assert hasattr(config, 'flow_magnitude_exponent'), "Missing flow_magnitude_exponent"
    
    assert config.constraint_tolerance == 1e-6, f"Wrong constraint_tolerance: {config.constraint_tolerance}"
    assert config.unit_circle_penalty_weight == 10.0, f"Wrong penalty weight: {config.unit_circle_penalty_weight}"
    assert config.enable_projection == True, f"Wrong enable_projection: {config.enable_projection}"
    assert config.enable_penalty == True, f"Wrong enable_penalty: {config.enable_penalty}"
    assert config.flow_magnitude_exponent == 1.0, f"Wrong flow_magnitude_exponent: {config.flow_magnitude_exponent}"
    
    print("  ✅ Config class and parameters correct")
except Exception as e:
    print(f"  ❌ Config test failed: {e}")
    all_passed = False

print("\n[Test 2] Projection (NumPy) - invalid action test...")
try:
    from mpc.constraint_utils import project_joint_angles
    
    invalid_actions = np.random.randn(32, 10, 15)
    invalid_actions = np.clip(invalid_actions, -1, 1)
    
    max_viol_before, _ = check_unit_circle(invalid_actions)
    
    projected_actions = project_joint_angles(invalid_actions, start_idx=0, end_idx=12)
    
    max_viol_after, violations = check_unit_circle(projected_actions)
    
    print(f"  Before: max violation = {max_viol_before:.6e}")
    print(f"  After:  max violation = {max_viol_after:.6e}")
    
    assert max_viol_after < 1e-6, f"Violations too large: {max_viol_after}"
    print(f"  ✅ NumPy projection works correctly")
        
except Exception as e:
    print(f"  ❌ Projection test failed: {e}")
    import traceback
    traceback.print_exc()
    all_passed = False

print("\n[Test 3] Projection (PyTorch) - batch+horizon shape...")
try:
    from mpc.constraint_utils import project_joint_angles_torch
    
    invalid_actions = torch.randn(32, 10, 15)
    invalid_actions = torch.clip(invalid_actions, -1, 1)
    
    projected_actions = project_joint_angles_torch(invalid_actions, start_idx=0, end_idx=12)
    
    max_viol, violations = check_unit_circle(projected_actions)
    
    print(f"  Shape: {projected_actions.shape}")
    print(f"  Max violation: {max_viol:.6e}")
    
    assert max_viol < 1e-6, f"Violations too large: {max_viol}"
    assert projected_actions.shape == (32, 10, 15), f"Wrong shape: {projected_actions.shape}"
    print(f"  ✅ PyTorch projection works correctly")
        
except Exception as e:
    print(f"  ❌ PyTorch projection test failed: {e}")
    import traceback
    traceback.print_exc()
    all_passed = False

print("\n[Test 4] Projection edge case - zero vector...")
try:
    from mpc.constraint_utils import project_joint_angles_torch
    
    zero_actions = torch.zeros(8, 5, 15)
    
    projected_actions = project_joint_angles_torch(zero_actions, start_idx=0, end_idx=12)
    
    has_nan = torch.isnan(projected_actions).any().item()
    has_inf = torch.isinf(projected_actions).any().item()
    
    assert not has_nan, "Produced NaN values"
    assert not has_inf, "Produced Inf values"
    print(f"  ✅ Zero vector handled (no NaN/Inf)")
        
except Exception as e:
    print(f"  ❌ Zero vector test failed: {e}")
    all_passed = False

print("\n[Test 5] Penalty parameter added to ActionRegularizationObjective...")
try:
    from mpc.flow_objectives import ActionRegularizationObjective
    import inspect
    
    sig = inspect.signature(ActionRegularizationObjective.__init__)
    params = sig.parameters
    
    assert 'unit_circle_penalty_weight' in params, "Parameter not found in __init__"
    
    default_val = params['unit_circle_penalty_weight'].default
    assert default_val == 10.0, f"Wrong default: {default_val}"
    
    obj = ActionRegularizationObjective(unit_circle_penalty_weight=10.0)
    assert hasattr(obj, 'unit_circle_penalty_weight'), "Attribute not stored"
    assert obj.unit_circle_penalty_weight == 10.0, f"Wrong value: {obj.unit_circle_penalty_weight}"
    
    print(f"  ✅ Penalty parameter correctly added (default=10.0)")
        
except Exception as e:
    print(f"  ❌ Penalty parameter test failed: {e}")
    import traceback
    traceback.print_exc()
    all_passed = False

print("\n[Test 6] Penalty code added to compute_reward...")
try:
    with open('mpc/flow_objectives.py', 'r') as f:
        flow_obj_code = f.read()
    
    has_sin_cos_extract = 'sin_vals' in flow_obj_code and 'cos_vals' in flow_obj_code
    has_unit_error = 'unit_error' in flow_obj_code or 'unit_penalty' in flow_obj_code
    has_weight_check = 'unit_circle_penalty_weight' in flow_obj_code
    
    assert has_sin_cos_extract, "No sin/cos extraction code found"
    assert has_unit_error, "No unit error computation found"
    assert has_weight_check, "No weight check found"
    
    print(f"  ✅ Penalty computation code present in compute_reward")
        
except Exception as e:
    print(f"  ❌ Penalty code test failed: {e}")
    all_passed = False

print("\n[Test 7] CEM integration - projection calls...")
try:
    with open('mpc/cem.py', 'r') as f:
        cem_code = f.read()
    
    has_import = 'from mpc.constraint_utils import project_joint_angles_torch' in cem_code
    assert has_import, "Missing import statement"
    
    projection_count = len(re.findall(r'project_joint_angles_torch\(', cem_code))
    
    print(f"  Import present: {has_import}")
    print(f"  Projection calls: {projection_count}")
    
    assert projection_count == 2, f"Expected 2 calls, found {projection_count}"
    print(f"  ✅ CEM has projection integration (2 branches)")
        
except Exception as e:
    print(f"  ❌ CEM integration test failed: {e}")
    all_passed = False

print("\n[Test 8] MPPI integration - projection calls...")
try:
    with open('mpc/mppi.py', 'r') as f:
        mppi_code = f.read()
    
    has_import = 'from mpc.constraint_utils import project_joint_angles' in mppi_code
    assert has_import, "Missing import statement"
    
    projection_count = len(re.findall(r'project_joint_angles\(', mppi_code))
    
    print(f"  Import present: {has_import}")
    print(f"  Projection calls: {projection_count}")
    
    assert projection_count >= 1, f"Expected >= 1 calls, found {projection_count}"
    print(f"  ✅ MPPI has projection integration")
        
except Exception as e:
    print(f"  ❌ MPPI integration test failed: {e}")
    all_passed = False

print("\n[Test 9] Gripper dimensions preserved...")
try:
    from mpc.constraint_utils import project_joint_angles_torch
    
    test_actions = torch.randn(8, 5, 15)
    gripper_before = test_actions[:, :, 12:15].clone()
    
    projected = project_joint_angles_torch(test_actions, start_idx=0, end_idx=12)
    gripper_after = projected[:, :, 12:15]
    
    diff = torch.abs(gripper_after - gripper_before).max().item()
    
    print(f"  Max gripper diff: {diff:.6e}")
    assert diff < 1e-9, f"Gripper changed: {diff}"
    print(f"  ✅ Gripper dimensions (12-14) preserved")
        
except Exception as e:
    print(f"  ❌ Gripper test failed: {e}")
    all_passed = False

print("\n" + "=" * 60)
if all_passed:
    print("✅ VERIFICATION PASSED")
    print("=" * 60)
    print("\nSummary:")
    print("  - Config class: 5 parameters with correct defaults ✅")
    print("  - Projection (NumPy): Enforces sin²+cos²=1 ✅")
    print("  - Projection (PyTorch): Handles batch+horizon shapes ✅")
    print("  - Projection edge case: No NaN/Inf on zero vectors ✅")
    print("  - Penalty parameter: Added to ActionRegularizationObjective ✅")
    print("  - Penalty code: Present in compute_reward method ✅")
    print("  - CEM integration: 2 projection calls (both branches) ✅")
    print("  - MPPI integration: 1+ projection call ✅")
    print("  - Gripper preservation: Dimensions 12-14 unchanged ✅")
    print("\nTasks 1a-1d: COMPLETE ✅")
    sys.exit(0)
else:
    print("❌ VERIFICATION FAILED")
    print("=" * 60)
    sys.exit(1)
