"""
Inference-only renderer for MPC rollout.

This module provides a unified rendering interface with:
1. Backend switch (`legacy`, `gsplat`, `fast_gauss`)
2. Execution mode switch (`inprocess`, `process`)
3. Batch rendering + de-duplication + small LRU cache
"""

import multiprocessing as mp
from collections import OrderedDict
import math
from typing import Optional

import numpy as np
import torch

from gaussian_renderer import render
from utils.sh_utils import eval_sh


def _normalize_render_mode(render_mode: str) -> str:
    mode = (render_mode or "RGB").upper().replace(" ", "")
    if mode in ("RGB", "RGB+D"):
        return mode
    raise ValueError(f"Unsupported render_mode={render_mode}. Use 'RGB' or 'RGB+D'.")


def _control_signature(control_vec: torch.Tensor, decimals: int = 4):
    vec = control_vec.detach().float().cpu().numpy()
    rounded = np.round(vec, decimals=decimals)
    return tuple(rounded.tolist())


def _worker_main(
    model_kwargs,
    backend: str,
    render_batch_size: int,
    render_cache_size: int,
    default_render_mode: str,
    dedup_key_mode: str,
    request_q,
    response_q,
):
    # Import in worker to avoid circular import at module load time.
    from mpc.gaussian_dynamics_model import GaussianDynamicsModel

    kwargs = dict(model_kwargs)
    kwargs["renderer_backend"] = backend
    kwargs["render_execution_mode"] = "inprocess"
    kwargs["render_batch_size"] = render_batch_size
    kwargs["render_cache_size"] = render_cache_size
    kwargs["render_mode"] = default_render_mode
    kwargs["render_dedup_key_mode"] = dedup_key_mode

    model = GaussianDynamicsModel(**kwargs)

    while True:
        msg = request_q.get()
        if msg is None:
            break
        if msg.get("type") == "close":
            break
        if msg.get("type") != "render":
            continue

        request_id = msg["request_id"]
        try:
            controls_np = msg["controls"]
            timestamps_np = msg["timestamps"]
            render_mode = _normalize_render_mode(msg.get("render_mode", default_render_mode))

            controls = torch.from_numpy(controls_np).to(model.device).float()
            timestamps = torch.from_numpy(timestamps_np).to(model.device).float()

            rendered = model.render_batch_with_controls(
                controls,
                time_batch=timestamps,
                grad_enabled=False,
                render_mode=render_mode,
            )
            output_np = rendered.detach().cpu().numpy()
            response_q.put(
                {
                    "request_id": request_id,
                    "ok": True,
                    "output": output_np,
                }
            )
        except Exception as exc:  # noqa: BLE001
            response_q.put(
                {
                    "request_id": request_id,
                    "ok": False,
                    "error": str(exc),
                }
            )

    model.close()


class InferenceRenderer:
    def __init__(
        self,
        model,
        backend: str = "legacy",
        execution_mode: str = "inprocess",
        render_batch_size: int = 8,
        render_cache_size: int = 16,
        default_render_mode: str = "RGB",
        dedup_key_mode: str = "timestamp_control",
        worker_model_kwargs: Optional[dict] = None,
    ):
        self.model = model
        self.backend = backend
        self.execution_mode = execution_mode
        self.render_batch_size = max(1, int(render_batch_size))
        self.render_cache_size = max(0, int(render_cache_size))
        self.default_render_mode = _normalize_render_mode(default_render_mode)
        self.dedup_key_mode = dedup_key_mode
        self.worker_model_kwargs = dict(worker_model_kwargs or {})

        self._cache = OrderedDict()
        self._backend_warned = set()
        self._request_id = 0

        self._ctx = None
        self._request_q = None
        self._response_q = None
        self._worker = None
        self._stream = None
        if torch.cuda.is_available() and "cuda" in str(self.model.device):
            self._stream = torch.cuda.Stream(device=torch.device(self.model.device))

    def clear_cache(self):
        self._cache.clear()

    def close(self):
        if self._worker is None:
            return
        try:
            self._request_q.put({"type": "close"})
        except Exception:  # noqa: BLE001
            pass
        self._worker.join(timeout=2.0)
        if self._worker.is_alive():
            self._worker.terminate()
            self._worker.join(timeout=1.0)
        self._worker = None
        self._request_q = None
        self._response_q = None
        self._ctx = None

    def render_batch(self, controls, time_batch=None, grad_enabled=False, render_mode=None):
        render_mode = _normalize_render_mode(render_mode or self.default_render_mode)

        if isinstance(controls, np.ndarray):
            controls = torch.from_numpy(controls).float()
        controls = controls.to(self.model.device)
        if controls.dim() == 1:
            controls = controls.unsqueeze(0)

        num_items = controls.shape[0]

        if time_batch is None:
            time_batch = torch.zeros(num_items, device=controls.device, dtype=torch.float32)
        elif isinstance(time_batch, np.ndarray):
            time_batch = torch.from_numpy(time_batch).float().to(controls.device)
        else:
            time_batch = time_batch.to(controls.device).float()
        if time_batch.dim() == 0:
            time_batch = time_batch.repeat(num_items)

        if grad_enabled or self.execution_mode != "process":
            return self._render_batch_inprocess(controls, time_batch, grad_enabled, render_mode)
        return self._render_batch_process(controls, time_batch, render_mode)

    def _cache_key(self, control_vec: torch.Tensor, timestamp: float, render_mode: str):
        ts = round(float(timestamp), 6)
        if self.dedup_key_mode == "timestamp":
            return (ts, render_mode)
        return (ts, render_mode, _control_signature(control_vec))

    def _cache_get(self, key):
        if key not in self._cache:
            return None
        value = self._cache.pop(key)
        self._cache[key] = value
        return value

    def _cache_put(self, key, value: torch.Tensor):
        if self.render_cache_size <= 0:
            return
        value_cpu = value.detach().cpu()
        if key in self._cache:
            self._cache.pop(key)
        self._cache[key] = value_cpu
        while len(self._cache) > self.render_cache_size:
            self._cache.popitem(last=False)

    def _render_batch_inprocess(self, controls, time_batch, grad_enabled: bool, render_mode: str):
        num_items = controls.shape[0]
        outputs = [None] * num_items

        # Keep gradient path deterministic: no dedupe/cache in grad mode.
        if grad_enabled:
            for i in range(num_items):
                outputs[i] = self._render_single(controls[i], float(time_batch[i]), True, render_mode)
            return torch.stack(outputs, dim=0)

        key_to_indices = OrderedDict()
        for i in range(num_items):
            key = self._cache_key(controls[i], float(time_batch[i]), render_mode)
            key_to_indices.setdefault(key, []).append(i)

        for key, indices in key_to_indices.items():
            cached = self._cache_get(key)
            if cached is not None:
                rendered = cached
            else:
                ref_idx = indices[0]
                rendered = self._render_single(
                    controls[ref_idx], float(time_batch[ref_idx]), False, render_mode
                ).detach().cpu()
                self._cache_put(key, rendered)

            for i in indices:
                outputs[i] = rendered

        return torch.stack(outputs, dim=0)

    def _ensure_worker(self):
        if self._worker is not None and self._worker.is_alive():
            return

        self._ctx = mp.get_context("spawn")
        self._request_q = self._ctx.Queue(maxsize=2)
        self._response_q = self._ctx.Queue(maxsize=2)
        self._worker = self._ctx.Process(
            target=_worker_main,
            args=(
                self.worker_model_kwargs,
                self.backend,
                self.render_batch_size,
                self.render_cache_size,
                self.default_render_mode,
                self.dedup_key_mode,
                self._request_q,
                self._response_q,
            ),
            daemon=True,
        )
        self._worker.start()

    def _render_batch_process(self, controls, time_batch, render_mode: str):
        self._ensure_worker()

        self._request_id += 1
        request_id = self._request_id
        self._request_q.put(
            {
                "type": "render",
                "request_id": request_id,
                "controls": controls.detach().cpu().numpy().astype(np.float32),
                "timestamps": time_batch.detach().cpu().numpy().astype(np.float32),
                "render_mode": render_mode,
            }
        )

        while True:
            response = self._response_q.get()
            if response.get("request_id") != request_id:
                continue
            if not response.get("ok", False):
                raise RuntimeError(response.get("error", "render worker failed"))
            return torch.from_numpy(response["output"])

    def _warn_backend_fallback(self, backend: str, reason: str):
        key = (backend, reason)
        if key in self._backend_warned:
            return
        self._backend_warned.add(key)
        print(f"[InferenceRenderer] Backend '{backend}' unavailable ({reason}), fallback to legacy.")

    def _render_single(self, control_vec, time_value: float, grad_enabled: bool, render_mode: str):
        backend = self.backend
        if grad_enabled:
            backend = "legacy"

        if backend == "legacy":
            return self._render_single_legacy(control_vec, time_value, grad_enabled, render_mode)
        if backend == "gsplat":
            try:
                return self._render_single_gsplat(control_vec, time_value, render_mode)
            except Exception as exc:  # noqa: BLE001
                self._warn_backend_fallback("gsplat", str(exc))
                return self._render_single_legacy(control_vec, time_value, grad_enabled, render_mode)
        if backend == "fast_gauss":
            try:
                return self._render_single_fast_gauss(control_vec, time_value, render_mode)
            except Exception as exc:  # noqa: BLE001
                self._warn_backend_fallback("fast_gauss", str(exc))
                return self._render_single_legacy(control_vec, time_value, grad_enabled, render_mode)

        self._warn_backend_fallback(backend, "unknown backend")
        return self._render_single_legacy(control_vec, time_value, grad_enabled, render_mode)

    def _render_single_legacy(self, control_vec, time_value: float, grad_enabled: bool, render_mode: str):
        cam = self.model.camera
        cam.time = time_value
        control_vec = control_vec.to(self.model.device)
        if control_vec.dim() == 1:
            control_vec = control_vec.unsqueeze(0)

        if grad_enabled:
            pkg = render(
                cam,
                self.model.gaussians,
                self.model.pipe_params,
                self.model.background,
                override_control_vec=control_vec,
                stage="fine",
            )
        else:
            if self._stream is not None:
                with torch.cuda.stream(self._stream):
                    with torch.inference_mode():
                        pkg = render(
                            cam,
                            self.model.gaussians,
                            self.model.pipe_params,
                            self.model.background,
                            override_control_vec=control_vec,
                            stage="fine",
                        )
                self._stream.synchronize()
            else:
                with torch.inference_mode():
                    pkg = render(
                        cam,
                        self.model.gaussians,
                        self.model.pipe_params,
                        self.model.background,
                        override_control_vec=control_vec,
                        stage="fine",
                    )

        rgb = pkg["render"]
        if render_mode == "RGB":
            return rgb
        depth = pkg["depth"].unsqueeze(0)
        return torch.cat([rgb, depth], dim=0)

    def _eval_sh_colors(self, means3d_final: torch.Tensor, shs_final: torch.Tensor) -> torch.Tensor:
        """Evaluate SH colors in python, aligned with legacy convert_SHs_python path."""
        if shs_final is None:
            raise RuntimeError("shs_final is None, cannot evaluate SH colors")
        if shs_final.dim() != 3:
            raise RuntimeError(f"Unexpected shs_final shape: {tuple(shs_final.shape)}")

        # [N, K, 3] -> [N, 3, K]
        shs_view = shs_final.transpose(1, 2).contiguous()
        coeff_count = shs_view.shape[-1]
        max_degree_from_coeff = int(math.sqrt(float(coeff_count)) - 1.0)
        max_degree_from_coeff = max(max_degree_from_coeff, 0)
        active_degree = int(self.model.gaussians.active_sh_degree)
        sh_degree = max(0, min(active_degree, max_degree_from_coeff))

        camera_center = self.model.camera.camera_center.to(means3d_final.device)
        dirs = means3d_final - camera_center.unsqueeze(0).expand(means3d_final.shape[0], -1)
        dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-8)

        sh2rgb = eval_sh(sh_degree, shs_view, dirs)
        return torch.clamp_min(sh2rgb + 0.5, 0.0)

    def _render_single_gsplat(self, control_vec, time_value: float, render_mode: str):
        means3D = self.model.gaussians.get_xyz
        scales = self.model.gaussians._scaling
        rotations = self.model.gaussians._rotation
        opacity = self.model.gaussians._opacity
        shs = self.model.gaussians.get_features

        control_vec = control_vec.to(means3D.device)
        if control_vec.dim() == 1:
            control_vec = control_vec.unsqueeze(0)
        control_batch = control_vec.repeat(means3D.shape[0], 1)

        means3D_final, scales_final, rotations_final, opacity_final, shs_final = self.model.gaussians._deformation(
            means3D, scales, rotations, opacity, shs, control_batch
        )
        scales_final = self.model.gaussians.scaling_activation(scales_final)
        quats_final = self.model.gaussians.rotation_activation(rotations_final)
        opacity_final = self.model.gaussians.opacity_activation(opacity_final).squeeze(-1)

        # Evaluate view-dependent color from SH (aligned with legacy path).
        colors = self._eval_sh_colors(means3D_final, shs_final)

        viewmat = self.model.camera.world_view_transform.transpose(0, 1).to(means3D.device)
        fx = self.model.focal_x
        fy = self.model.focal_y
        if fx is None or fy is None:
            fx = (self.model.image_width * 0.5) / np.tan(float(self.model.camera.FoVx) * 0.5)
            fy = (self.model.image_height * 0.5) / np.tan(float(self.model.camera.FoVy) * 0.5)
        K = torch.tensor(
            [[fx, 0.0, self.model.cx], [0.0, fy, self.model.cy], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=means3D.device,
        )

        with torch.inference_mode():
            # New gsplat API (v1.x): rasterization(...)
            try:
                from gsplat import rasterization

                rgb, alpha, info = rasterization(
                    means=means3D_final,
                    quats=quats_final,
                    scales=scales_final,
                    opacities=opacity_final,
                    colors=colors,
                    viewmats=viewmat.unsqueeze(0),
                    Ks=K.unsqueeze(0),
                    width=int(self.model.image_width),
                    height=int(self.model.image_height),
                    render_mode=render_mode,
                    packed=False,
                )

                # Output expected shape: [H, W, C], convert to [C, H, W]
                if rgb.dim() == 4:
                    rgb = rgb[0]
                if rgb.shape[-1] in (3, 4):
                    rgb = rgb.permute(2, 0, 1).contiguous()
            except Exception:
                # Old gsplat API (v0.x): project_gaussians(...) + rasterize_gaussians(...)
                # Depth mode is unavailable in old API, fallback to legacy renderer.
                if render_mode != "RGB":
                    raise RuntimeError("old gsplat API does not support render_mode != RGB")

                from gsplat import project_gaussians, rasterize_gaussians

                block_width = 16
                fx_val = float(fx)
                fy_val = float(fy)
                cx_val = float(self.model.cx)
                cy_val = float(self.model.cy)
                H = int(self.model.image_height)
                W = int(self.model.image_width)

                (
                    xys,
                    depths,
                    radii,
                    conics,
                    compensation,
                    num_tiles_hit,
                    cov3d,
                ) = project_gaussians(
                    means3D_final,
                    scales_final,
                    1.0,  # global scale
                    quats_final,
                    viewmat,
                    fx_val,
                    fy_val,
                    cx_val,
                    cy_val,
                    H,
                    W,
                    block_width,
                )

                opacity_old = opacity_final
                if opacity_old.dim() == 1:
                    opacity_old = opacity_old.unsqueeze(-1)

                bg = self.model.background.to(colors.device)
                rgb_hw = rasterize_gaussians(
                    xys,
                    depths,
                    radii,
                    conics,
                    num_tiles_hit,
                    colors,
                    opacity_old,
                    H,
                    W,
                    block_width,
                    background=bg,
                )
                rgb = rgb_hw.permute(2, 0, 1).contiguous()

        # Robust fallback for unstable gsplat frames.
        if (not torch.isfinite(rgb).all()) or float(rgb.abs().max()) < 1e-8:
            raise RuntimeError("gsplat produced invalid or all-zero frame")
        return rgb

    def _render_single_fast_gauss(self, control_vec, time_value: float, render_mode: str):
        raise NotImplementedError("fast_gauss backend is feature-flag only and not wired in this environment")
