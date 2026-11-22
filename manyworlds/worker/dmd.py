"""Few-step sampling for DMD-distilled video models (FastWan2.2-TI2V-5B: 3 steps at timesteps 1000, 757, 522).

A DMD student is trained to predict the clean video in one pass from a fixed set of noise levels; sampling it with
a multistep solver (diffusers' UniPC, the repo's default scheduler) at 3 steps gives mush (2026-11-22 benchmark: 0 of
5 clips passed). This scheduler plugs into the standard diffusers Wan pipeline: at each listed timestep the model's
flow prediction gives the clean latents (x0 = x - sigma * v), which are re-noised to the next timestep (fresh
noise, as DMD samples), and returned clean after the last. Sigma is timestep / num_train_timesteps.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _Config:
    num_train_timesteps: int = 1000


class DMDScheduler:
    order = 1
    init_noise_sigma = 1.0

    def __init__(self, timesteps: list[int], *, seed: int = 0, num_train_timesteps: int = 1000):
        self.config = _Config(num_train_timesteps)
        self._steps = [int(t) for t in timesteps]
        self._seed = int(seed)
        self.timesteps = None
        self._i = 0

    def set_timesteps(self, num_inference_steps=None, device=None, **_):
        import torch
        self.timesteps = torch.tensor(self._steps, dtype=torch.float32, device=device)
        self._i = 0
        self._gen = torch.Generator("cpu").manual_seed(self._seed)

    def sigma(self, t) -> float:
        return float(t) / self.config.num_train_timesteps

    def step(self, model_output, timestep, sample, return_dict=False, **_):
        import torch
        s = self.sigma(timestep)
        x0 = sample.float() - s * model_output.float()
        self._i += 1
        if self._i < len(self._steps):
            s_next = self.sigma(self._steps[self._i])
            noise = torch.randn(x0.shape, generator=self._gen, dtype=torch.float32).to(x0.device)
            out = (1 - s_next) * x0 + s_next * noise
        else:
            out = x0
        out = out.to(sample.dtype)
        return (out,) if not return_dict else type("Out", (), {"prev_sample": out})()
