import math
import os


def make_actions(angles_deg):
    angles_rad = []
    for row in angles_deg:
        angles_rad.append([math.radians(v) for v in row])
    sin_vals = [[math.sin(v) for v in row] for row in angles_rad]
    cos_vals = [[math.cos(v) for v in row] for row in angles_rad]
    actions = []
    for t in range(len(angles_deg)):
        pairs = []
        for j in range(len(angles_deg[t])):
            pairs.extend([sin_vals[t][j], cos_vals[t][j]])
        actions.append(pairs)
    return actions


def evaluate_actions(actions, threshold_deg):
    threshold = math.radians(threshold_deg)
    max_delta = 0.0
    for idx in range(1, len(actions)):
        prev = actions[idx - 1]
        curr = actions[idx]
        for j in range(0, 12, 2):
            prev_angle = math.atan2(prev[j], prev[j + 1])
            curr_angle = math.atan2(curr[j], curr[j + 1])
            delta = math.atan2(math.sin(curr_angle - prev_angle), math.cos(curr_angle - prev_angle))
            max_delta = max(max_delta, abs(delta))
    valid = max_delta <= threshold
    penalty = max(max_delta - threshold, 0.0)
    return valid, penalty


def main():
    results = []

    actions = make_actions([
        [0, 0, 0, 0, 0, 0],
        [60, 60, 60, 60, 60, 60],
    ])
    name = "large_delta_60deg"
    results.append((name, actions))

    actions = make_actions([
        [0, 0, 0, 0, 0, 0],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    ])
    actions[0][0] = 0.9999
    actions[0][1] = 0.0141
    actions[1][0] = 0.9999
    actions[1][1] = 0.0141
    name = "near_boundary_sincos"
    results.append((name, actions))

    actions = make_actions([
        [170, -170, 170, -170, 170, -170],
        [-170, 170, -170, 170, -170, 170],
        [170, -170, 170, -170, 170, -170],
    ])
    name = "multi_wrap_180"
    results.append((name, actions))

    actions = make_actions([
        [0, 0, 0, 0, 0, 0],
        [10, 10, 10, 10, 10, 10],
    ])
    name = "optical_flow_independence"
    results.append((name, actions))

    report_lines = []
    report_lines.append("FAILURE CASE TESTING (DATA-ONLY)")
    threshold = 30.0

    for name, actions in results:
        valid, penalty = evaluate_actions(actions, threshold)
        report_lines.append(f"{name}: valid={bool(valid)}, penalty={float(penalty)}")

    output_path = os.path.join(os.path.dirname(__file__), "failure_case_testing.txt")
    with open(output_path, "w") as f:
        f.write("\n".join(report_lines))

    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
