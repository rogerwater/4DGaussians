import json
import math
import os


def load_joint_positions(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    frames = data.get("frames", [])
    joint_positions = []
    for frame in frames:
        if "joint_pos" in frame:
            joint_positions.append([float(v) for v in frame["joint_pos"]])
    return joint_positions


def angle_from_pair(sin_val, cos_val):
    return math.atan2(sin_val, cos_val)


def wrapped_delta(angle_new, angle_old):
    delta = angle_new - angle_old
    return math.atan2(math.sin(delta), math.cos(delta))


def main():
    json_path = "/home/ubuntu/project/data/dm_control_push/transforms.json"
    if not os.path.exists(json_path):
        json_path = "/home/ubuntu/project/data/dm_control/transforms.json"

    joint_positions = load_joint_positions(json_path)
    if not joint_positions:
        raise RuntimeError("No joint_pos frames found")

    first = joint_positions[0]
    first_value = first[0]

    unit_circle_max_error = 0.0
    for frame in joint_positions:
        for i in range(0, 12, 2):
            sin_val = frame[i]
            cos_val = frame[i + 1]
            error = abs((sin_val * sin_val + cos_val * cos_val) - 1.0)
            if error > unit_circle_max_error:
                unit_circle_max_error = error

    max_delta = 0.0
    sum_delta = 0.0
    count_delta = 0
    violations = 0
    threshold = math.radians(30.0)

    for idx in range(1, len(joint_positions)):
        prev = joint_positions[idx - 1]
        curr = joint_positions[idx]
        for j in range(0, 12, 2):
            prev_angle = angle_from_pair(prev[j], prev[j + 1])
            curr_angle = angle_from_pair(curr[j], curr[j + 1])
            delta = abs(wrapped_delta(curr_angle, prev_angle))
            sum_delta += delta
            count_delta += 1
            if delta > max_delta:
                max_delta = delta
            if delta > threshold:
                violations += 1

    mean_delta = sum_delta / count_delta if count_delta else 0.0

    report = []
    report.append("END-TO-END VALIDATION (DATA-ONLY)")
    report.append("Source: " + json_path)
    report.append("Frames with joint_pos: " + str(len(joint_positions)))
    report.append("First joint_pos[0]: " + str(first_value))
    report.append("Unit circle max error: " + str(unit_circle_max_error))
    report.append("Angular velocity stats (radians):")
    report.append("  Max Δθ: " + str(max_delta))
    report.append("  Mean Δθ: " + str(mean_delta))
    report.append("  Violations (>30°): " + str(violations))
    report.append("")
    report.append("CEM optimizer execution: NOT RUN (numpy/torch unavailable in environment)")
    report.append("Clipping removal verified via code inspection in mpc/cem.py and mpc/mppi.py")

    output_path = os.path.join(os.path.dirname(__file__), "end_to_end_validation.txt")
    with open(output_path, "w") as f:
        f.write("\n".join(report))

    print("\n".join(report))


if __name__ == "__main__":
    main()
