import numpy as np
import torch


def project_joint_angles(actions: np.ndarray, start_idx=0, end_idx=12) -> np.ndarray:
    if (end_idx - start_idx) % 2 != 0:
        raise ValueError("Joint angle sin/cos slice must contain an even number of dimensions.")

    pairs = actions[..., start_idx:end_idx].reshape(
        *actions.shape[:-1], (end_idx - start_idx) // 2, 2
    )
    norms = np.linalg.norm(pairs, axis=-1, keepdims=True)
    pairs_normalized = pairs / np.maximum(norms, 1e-6)

    actions_copy = actions.copy()
    actions_copy[..., start_idx:end_idx] = pairs_normalized.reshape(
        *actions.shape[:-1], end_idx - start_idx
    )
    return actions_copy


def project_joint_angles_torch(actions: torch.Tensor, start_idx=0, end_idx=12) -> torch.Tensor:
    if (end_idx - start_idx) % 2 != 0:
        raise ValueError("Joint angle sin/cos slice must contain an even number of dimensions.")

    pairs = actions[..., start_idx:end_idx].reshape(
        *actions.shape[:-1], (end_idx - start_idx) // 2, 2
    )
    norms = torch.linalg.norm(pairs, dim=-1, keepdim=True)
    pairs_normalized = pairs / torch.clamp(norms, min=1e-6)

    actions_copy = actions.clone()
    actions_copy[..., start_idx:end_idx] = pairs_normalized.reshape(
        *actions.shape[:-1], end_idx - start_idx
    )
    return actions_copy


def compute_angular_velocity(actions: np.ndarray, start_idx=0, end_idx=12) -> np.ndarray:
    if (end_idx - start_idx) % 2 != 0:
        raise ValueError("Joint angle sin/cos slice must contain an even number of dimensions.")

    pairs = actions[..., start_idx:end_idx].reshape(
        *actions.shape[:-1], (end_idx - start_idx) // 2, 2
    )
    sin_vals = pairs[..., 0]
    cos_vals = pairs[..., 1]
    angles = np.arctan2(sin_vals, cos_vals)
    delta_angles = angles[..., 1:, :] - angles[..., :-1, :]
    delta_wrapped = np.arctan2(np.sin(delta_angles), np.cos(delta_angles))
    return delta_wrapped


def compute_angular_velocity_torch(actions: torch.Tensor, start_idx=0, end_idx=12) -> torch.Tensor:
    if (end_idx - start_idx) % 2 != 0:
        raise ValueError("Joint angle sin/cos slice must contain an even number of dimensions.")

    pairs = actions[..., start_idx:end_idx].reshape(
        *actions.shape[:-1], (end_idx - start_idx) // 2, 2
    )
    sin_vals = pairs[..., 0]
    cos_vals = pairs[..., 1]
    angles = torch.atan2(sin_vals, cos_vals)
    delta_angles = angles[..., 1:, :] - angles[..., :-1, :]
    delta_wrapped = torch.atan2(torch.sin(delta_angles), torch.cos(delta_angles))
    return delta_wrapped


def check_angular_velocity_constraint(
    actions: np.ndarray,
    action_t_prev: np.ndarray = None,
    max_angular_velocity: float = 0.524,
    start_idx: int = 0,
    end_idx: int = 12,
):
    if (end_idx - start_idx) % 2 != 0:
        raise ValueError("Joint angle sin/cos slice must contain an even number of dimensions.")

    if action_t_prev is None:
        full_sequence = actions
    else:
        prev = np.broadcast_to(action_t_prev, actions.shape)
        full_sequence = np.concatenate([prev[:, :1, :], actions], axis=1)

    if full_sequence.shape[1] < 2:
        num_samples = actions.shape[0]
        return np.ones(num_samples, dtype=bool), np.zeros(num_samples, dtype=np.float32)

    delta_wrapped = compute_angular_velocity(full_sequence, start_idx=start_idx, end_idx=end_idx)
    max_delta = np.max(np.abs(delta_wrapped), axis=(1, 2))
    valid_mask = max_delta <= max_angular_velocity
    penalty = np.maximum(max_delta - max_angular_velocity, 0.0)
    return valid_mask, penalty


def check_angular_velocity_constraint_torch(
    actions: torch.Tensor,
    action_t_prev: torch.Tensor = None,
    max_angular_velocity: float = 0.524,
    start_idx: int = 0,
    end_idx: int = 12,
):
    if (end_idx - start_idx) % 2 != 0:
        raise ValueError("Joint angle sin/cos slice must contain an even number of dimensions.")

    if action_t_prev is None:
        full_sequence = actions
    else:
        prev = action_t_prev.expand_as(actions)
        full_sequence = torch.cat([prev[:, :1, :], actions], dim=1)

    if full_sequence.shape[1] < 2:
        num_samples = actions.shape[0]
        return torch.ones(num_samples, dtype=torch.bool, device=actions.device), torch.zeros(
            num_samples, dtype=actions.dtype, device=actions.device
        )

    delta_wrapped = compute_angular_velocity_torch(full_sequence, start_idx=start_idx, end_idx=end_idx)
    max_delta = torch.amax(torch.abs(delta_wrapped), dim=(1, 2))
    valid_mask = max_delta <= max_angular_velocity
    penalty = torch.clamp(max_delta - max_angular_velocity, min=0.0)
    return valid_mask, penalty
