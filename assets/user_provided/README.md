# User-Provided Images

This directory contains the initial and target frames provided by the user for MPC planning.

## Files

- `initial_frame.png` - Initial frame (Image 1 from user, showing robot arm and yellow cube)
- `target_frame.png` - Target frame (Image 2 from user, showing robot arm in different pose with yellow cube)

## Usage

These images are used by `test_cotracker_mpc.py` with the following configuration:

- **Camera**: cam06 from `/home/ubuntu/project/data/dm_control_push/transforms.json`
- **Initial Control u**: Extracted from cam06/frame_00001.jpg entry in transforms.json
- **Resolution**: 480x480 (matching dm_control dataset)

## Data Source

From transforms.json:
- Camera ID: 6
- Initial frame: cam06/frame_00001.jpg
- Initial control u: 15-dimensional vector (6 joints × 2 [sin/cos] + 3 gripper)
  ```
  [-0.9977839652403836, 0.06653689735159651, -0.4846871119066772, -0.8746876034056754, 
   0.8378519760656709, -0.5458974868991893, -0.6205189510461596, 0.7841914507265262, 
   0.9031293515909968, -0.42936857627780073, 0.8247429443121709, 0.5655078034892989, 
   0.6062964901087786, 0.5926931803994663, 0.5927027465737802]
  ```

## Note

The images provided by the user need to be manually saved to this directory as:
- `initial_frame.png` (from [Image 1])
- `target_frame.png` (from [Image 2])

The images should be saved at 480x480 resolution to match the camera parameters in transforms.json.
