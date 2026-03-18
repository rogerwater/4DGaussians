# Video Synthesis Integration for MPC Planning

**Date**: 2026-03-12
**Author**: Assistant
**Related Files**: 
- `scripts/synthesize_mpc_videos.py` (new)
- `test_cotracker_mpc.py` (modified)
- `scripts/README_synthesize_mpc_videos.md` (new)

---

## Overview

Added video synthesis functionality to automatically create videos from MPC planning results. Two videos are generated:

1. **planning_result.mp4** - Rendered frames (step_XXXX_rendered.png)
2. **planning_result_with_points.mp4** - Frames with tracking points (step_XXXX_with_points.png)

Both videos are created at **10 FPS** by default.

---

## Changes

### 1. New Script: `scripts/synthesize_mpc_videos.py`

**Purpose**: Standalone script to synthesize videos from MPC planning images

**Features**:
- Finds all `step_*_rendered.png` and `step_*_with_points.png` files
- Sorts them naturally (handles step_0000 to step_9999+)
- Creates MP4 videos using OpenCV (cv2.VideoWriter with mp4v codec)
- Configurable FPS (default: 10)
- Optional output filename suffix

**Usage**:
```bash
# Basic usage
python scripts/synthesize_mpc_videos.py --input_dir outputs/test_cam06_with_initial_u_correct

# Custom FPS
python scripts/synthesize_mpc_videos.py --input_dir outputs/test_cam06_with_initial_u_correct --fps 15

# With suffix
python scripts/synthesize_mpc_videos.py --input_dir outputs/my_test --output_suffix "v2"
```

**Key Functions**:
- `synthesize_video(image_paths, output_path, fps)` - Core video creation logic
- `natural_sort_key(s)` - Natural sorting for filenames with numbers

---

### 2. Modified: `test_cotracker_mpc.py`

**Added Functions** (lines 160-241):

#### `synthesize_planning_videos(output_dir, fps=10)`
Main function that finds step images and creates both videos.

**Logic**:
1. Find all `step_*_rendered.png` files
2. Find all `step_*_with_points.png` files
3. Create `planning_result.mp4` from rendered images
4. Create `planning_result_with_points.mp4` from points images
5. Return `True` if at least one video was created successfully

#### `create_video_from_images(image_paths, output_path, fps)`
Helper function that uses OpenCV to write video frames.

**Logic**:
1. Read first image to get dimensions (height, width)
2. Create `cv2.VideoWriter` with mp4v codec
3. Iterate through all images, resize if needed, write to video
4. Release video writer

**Integration Point** (lines 775-783):
Added after saving metrics, before final completion message:
```python
print("\n" + "="*70)
print("Synthesizing Videos")
print("="*70)

synthesize_success = synthesize_planning_videos(args.output_dir, fps=10)

print("\n" + "="*70)
print("Test Complete!")
print(f"All outputs saved to: {args.output_dir}")
if synthesize_success:
    print("✓ Planning videos synthesized successfully")
print("="*70)
```

---

### 3. New Documentation: `scripts/README_synthesize_mpc_videos.md`

Comprehensive documentation covering:
- Automatic integration into test_cotracker_mpc.py
- Manual usage examples
- Input requirements
- Output format
- Dependencies
- Troubleshooting
- Batch processing examples

---

## Testing

### Test 1: Standalone Script
```bash
conda run -n Gaussians4D python scripts/synthesize_mpc_videos.py \
    --input_dir outputs/test_cam06_with_initial_u_correct \
    --fps 10
```

**Result**: ✅ Success
- Created `planning_result.mp4` (123K, 11 frames)
- Created `planning_result_with_points.mp4` (403K, 10 frames)

### Test 2: Integrated Function
```bash
conda run -n Gaussians4D python -c "
from test_cotracker_mpc import synthesize_planning_videos
result = synthesize_planning_videos('outputs/test_cam06_with_initial_u_correct', fps=10)
"
```

**Result**: ✅ Success
- Both videos created successfully
- Function returned `True`

---

## Video Format Details

**Codec**: MP4v (OpenCV default, widely compatible)
**Container**: MP4
**FPS**: 10 (configurable)
**Resolution**: Matches input images (480x480 for dm_control_push dataset)
**Color Space**: BGR (OpenCV default)

**Note**: If playback issues occur with MP4v codec, videos can be re-encoded to H.264:
```bash
ffmpeg -i planning_result.mp4 -vcodec libx264 planning_result_h264.mp4
```

---

## Integration Flow

When `test_cotracker_mpc.py` runs:

1. **Planning Loop** - Generates step images (step_0000 to step_XXXX)
2. **Save Results** - Saves metrics.json, loss_history.csv, action_sequence.npy
3. **Synthesize Videos** ← NEW
   - Finds all step_*_rendered.png files
   - Finds all step_*_with_points.png files
   - Creates planning_result.mp4
   - Creates planning_result_with_points.mp4
4. **Completion** - Prints summary with video synthesis status

---

## File Locations

**New Files**:
- `/home/ubuntu/yyf/4DGaussians/scripts/synthesize_mpc_videos.py` (executable)
- `/home/ubuntu/yyf/4DGaussians/scripts/README_synthesize_mpc_videos.md`

**Modified Files**:
- `/home/ubuntu/yyf/4DGaussians/test_cotracker_mpc.py`
  - Added `synthesize_planning_videos()` function (lines 160-213)
  - Added `create_video_from_images()` helper (lines 215-241)
  - Added video synthesis call in main() (lines 775-783)

**Test Output**:
- `/home/ubuntu/yyf/4DGaussians/outputs/test_cam06_with_initial_u_correct/planning_result.mp4`
- `/home/ubuntu/yyf/4DGaussians/outputs/test_cam06_with_initial_u_correct/planning_result_with_points.mp4`

---

## Future Enhancements

Possible improvements for future iterations:

1. **H.264 Codec**: Use libx264 for better compression and compatibility
   - Requires ffmpeg-python or subprocess call to ffmpeg
   
2. **Side-by-Side Comparison**: Create a third video with rendered + points side-by-side
   
3. **Trajectory Overlay**: Add trajectory lines showing point movement over time
   
4. **Metadata Overlay**: Add frame number, step number, action values as text overlay
   
5. **GIF Export**: Option to also export as animated GIF for easy sharing
   
6. **Configurable FPS**: Allow user to set FPS via command-line argument in test_cotracker_mpc.py

---

## Dependencies

**Required**:
- opencv-python (cv2) - Video encoding/decoding
- numpy - Array operations

**Already Available**: Both packages are installed in the Gaussians4D conda environment.

---

## Notes

1. **Natural Sorting**: The script uses natural sorting to correctly order files (step_0, step_1, ..., step_10, step_11) instead of lexicographic sorting (step_0, step_1, step_10, step_11, step_2, ...)

2. **Automatic Cleanup**: No temporary files are created. Videos are written directly from PNG images.

3. **Error Handling**: 
   - Gracefully handles missing images (skips and warns)
   - Handles dimension mismatches (resizes to first image dimensions)
   - Returns False if no images found

4. **Performance**: Video synthesis is fast (~1 second for 10 frames at 480x480 resolution)

---

## Verification

To verify the integration works in a full MPC test run:

```bash
python test_cotracker_mpc.py \
    --camera_name cam06 \
    --initial_frame_name frame_00001 \
    --num_steps 10 \
    --horizon 10 \
    --device cuda:1 \
    --output_dir outputs/test_video_synthesis_integration

# Expected output at end:
# ======================================================================
# Synthesizing Videos
# ======================================================================
#   ✓ Rendered video: outputs/test_video_synthesis_integration/planning_result.mp4 (11 frames @ 10 FPS)
#   ✓ Points video: outputs/test_video_synthesis_integration/planning_result_with_points.mp4 (10 frames @ 10 FPS)
# 
# ======================================================================
# Test Complete!
# All outputs saved to: outputs/test_video_synthesis_integration
# ✓ Planning videos synthesized successfully
# ======================================================================
```

---

## Summary

✅ **Standalone script created**: `scripts/synthesize_mpc_videos.py`  
✅ **Integration complete**: Automatic video synthesis in `test_cotracker_mpc.py`  
✅ **Documentation added**: `scripts/README_synthesize_mpc_videos.md`  
✅ **Tested successfully**: Both standalone and integrated modes work  
✅ **No breaking changes**: Existing functionality preserved  

The video synthesis feature is now fully integrated and will automatically run after every MPC planning test.
