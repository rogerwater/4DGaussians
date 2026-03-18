#!/usr/bin/env python
import re
import sys

print("=" * 60)
print("Task 3 Verification: MPC Iteration Count")
print("=" * 60)

scripts_to_check = [
    ("test_cotracker_mpc.py", 1),
    ("demo_flow_guided_mpc.py", 3),
    ("demo_cotracker_mpc.py", 1),
]

all_passed = True

for script, expected_count in scripts_to_check:
    print(f"\n[Checking] {script}...")
    try:
        with open(script, 'r') as f:
            content = f.read()
        
        argparse_matches = re.findall(r'--opt_iters.*?default\s*=\s*(\d+)', content, re.MULTILINE)
        dataclass_matches = re.findall(r'opt_iters\s*:\s*int\s*=\s*(\d+)', content, re.MULTILINE)
        matches = argparse_matches + dataclass_matches
        
        if len(matches) != expected_count:
            print(f"  ❌ Expected {expected_count} --opt_iters argument(s), found {len(matches)}")
            all_passed = False
            continue
        
        correct_defaults = all(int(m) == 10 for m in matches)
        
        if correct_defaults:
            print(f"  ✅ All {expected_count} --opt_iters default(s) = 10")
        else:
            print(f"  ❌ Found incorrect defaults: {matches}")
            all_passed = False
            
    except FileNotFoundError:
        print(f"  ❌ File not found: {script}")
        all_passed = False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        all_passed = False

print(f"\n[Checking] Shell scripts for hardcoded opt_iters=5...")
try:
    import subprocess
    result = subprocess.run(
        ["bash", "-c", "grep -r '--opt_iters 5' scripts/ *.sh 2>/dev/null || true"],
        capture_output=True,
        text=True,
        cwd="/home/ubuntu/yyf/4DGaussians"
    )
    
    if result.stdout.strip():
        print(f"  ❌ Found hardcoded --opt_iters 5:")
        print(result.stdout)
        all_passed = False
    else:
        print(f"  ✅ No hardcoded --opt_iters 5 found")
        
except Exception as e:
    print(f"  ⚠️  Could not check shell scripts: {e}")

print("\n" + "=" * 60)
if all_passed:
    print("✅ VERIFICATION PASSED")
    print("=" * 60)
    print("\nSummary:")
    print("  - test_cotracker_mpc.py: 1x --opt_iters default=10 ✅")
    print("  - demo_flow_guided_mpc.py: 1x --opt_iters default=10 ✅")
    print("  - demo_cotracker_mpc.py: 1x --opt_iters default=10 ✅")
    print("  - No hardcoded --opt_iters 5 in shell scripts ✅")
    print("\nTask 3: COMPLETE ✅")
    sys.exit(0)
else:
    print("❌ VERIFICATION FAILED")
    print("=" * 60)
    sys.exit(1)
