# CUDA Device Mapping Explanation

## User's Question (用户问题)

> "在sh里面定义要使用cuda3，但是实际上脚本还是在找cuda0"
> 
> Translation: "I defined cuda3 in the shell script, but the script is still looking for cuda0"

## Answer (答案)

**This is CORRECT and EXPECTED behavior!** 这是**正确的预期行为**！

## How CUDA_VISIBLE_DEVICES Works

When you set `CUDA_VISIBLE_DEVICES=3`, PyTorch **remaps** the GPUs:

```
Physical GPUs:     0    1    2    3    4    5    ...
                   ↓    ↓    ↓    ↓    ↓    ↓
After CUDA_VISIBLE_DEVICES=3:
PyTorch sees:      -    -    -    0    -    -    ...
                            (only GPU #3 is visible as cuda:0)
```

### Example Flow

1. **Shell script** (`run_cotracker_test.sh`):
   ```bash
   DEVICE="cuda:3"  # User wants physical GPU #3
   python3 test/integration/test_cotracker_mpc.py --device "$DEVICE"
   ```

2. **Python script** (`test_cotracker_mpc.py:60-80`):
   ```python
   def parse_device_early():
       device = sys.argv[i + 1]  # "cuda:3"
       device_id = "3"            # Extract ID
       os.environ['CUDA_VISIBLE_DEVICES'] = "3"  # Set before torch import!
       return 'cuda:0', device_id  # Return cuda:0 (remapped)
   ```

3. **PyTorch**:
   ```python
   import torch  # After CUDA_VISIBLE_DEVICES is set
   
   # PyTorch only sees GPU #3, so:
   torch.cuda.device_count()     # → 1 (only one GPU visible)
   torch.cuda.current_device()   # → 0 (first visible GPU)
   
   # When you use cuda:0 in code:
   tensor = torch.zeros(10).to('cuda:0')  # Actually uses physical GPU #3!
   ```

## Verification

Run this test to confirm:

```bash
cd /home/ubuntu/yyf/4DGaussians
python3 << 'EOF'
import os
import sys

# Simulate your shell script passing --device cuda:3
sys.argv = ['test.py', '--device', 'cuda:3']

# Extract device ID and set CUDA_VISIBLE_DEVICES BEFORE importing torch
device = sys.argv[2]  # "cuda:3"
device_id = device.split(':')[1]  # "3"
os.environ['CUDA_VISIBLE_DEVICES'] = device_id  # Set to "3"

# NOW import torch
import torch

print(f"Shell script requested: {device}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"PyTorch sees {torch.cuda.device_count()} GPU(s)")
print(f"PyTorch current device: cuda:{torch.cuda.current_device()}")
print(f"Physical GPU name: {torch.cuda.get_device_name(0)}")
print(f"\n✓ When code uses 'cuda:0', it runs on physical GPU #{device_id}")
EOF
```

**Expected output:**
```
Shell script requested: cuda:3
CUDA_VISIBLE_DEVICES: 3
PyTorch sees 1 GPU(s)
PyTorch current device: cuda:0
Physical GPU name: NVIDIA GeForce RTX 4090

✓ When code uses 'cuda:0', it runs on physical GPU #3
```

## Debug Logs Added

I added debug prints to help you verify this:

### 1. Shell script output (`run_cotracker_test.sh`):
```
[2/5] Verifying configuration...
  Device to use: cuda:3
  Optimizer: cem-gd
```

### 2. Python script output (`test_cotracker_mpc.py`):
```
[Device Mapping] Requested device: cuda:3
[Device Mapping] Set CUDA_VISIBLE_DEVICES=3
[Device Mapping] PyTorch will use: cuda:0 (mapped to physical GPU 3)

[PyTorch Verification]
  args.device = cuda:0
  torch.cuda.is_available() = True
  torch.cuda.current_device() = 0
  torch.cuda.get_device_name(0) = NVIDIA GeForce RTX 4090
  Physical GPU ID (from CUDA_VISIBLE_DEVICES) = 3
```

## Why This Design?

**Benefits:**
1. **Code portability**: Same code works on different GPUs by changing CUDA_VISIBLE_DEVICES
2. **Multi-GPU control**: Can restrict scripts to specific GPUs without code changes
3. **Resource isolation**: Prevents accidental access to other GPUs
4. **Standard practice**: This is how PyTorch/TensorFlow/CUDA handle GPU selection

**Common misunderstanding:**
- ❌ "cuda:0 in code means physical GPU #0"
- ✅ "cuda:0 in code means the FIRST VISIBLE GPU (remapped via CUDA_VISIBLE_DEVICES)"

## Troubleshooting

If the script is **actually** using the wrong GPU (not just displaying "cuda:0"), check:

1. **CUDA_VISIBLE_DEVICES timing**: Must be set BEFORE `import torch`
   - ✅ Our code does this at line 66 (before line 82 `import torch`)

2. **Verify physical GPU usage**:
   ```bash
   # Run your script, then check GPU usage
   nvidia-smi
   
   # Look for python process - it should be on GPU #3, not GPU #0
   ```

3. **Check for hardcoded device overrides**:
   ```bash
   # Search for any code that might override device
   grep -n "cuda:0" test/integration/test_cotracker_mpc.py | grep -v "# comment"
   ```

## Conclusion (结论)

The behavior you're seeing is **correct and expected** (你看到的行为是**正确且符合预期的**). When the code shows "cuda:0", it's actually using the physical GPU you specified in the shell script (physical GPU #3).

If you want to verify it's using the correct physical GPU, run `nvidia-smi` while the script is running and check which GPU shows activity.

---

**Files Modified:**
- `test/integration/test_cotracker_mpc.py` (lines 60-87, 352-365): Added debug prints
- `run_cotracker_test.sh` (lines 110-117): Added device verification output

**How to test:**
```bash
cd /home/ubuntu/yyf/4DGaussians
bash run_cotracker_test.sh
# Check the debug output to confirm device mapping
```
