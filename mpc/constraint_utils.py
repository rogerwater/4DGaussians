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
