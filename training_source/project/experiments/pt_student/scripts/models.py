"""PT-student models and auxiliary targets (PLAN.md section 2).

Students read the SAME 320-D observation as C5 (group ``actor``: 10 x 16 joint positions relative to default, then
10 x 16 previous commands, term-major) and output the raw 16-D action u (the environment applies tanh and the
+-1/24 rad delta mapping, exactly as for C5).  Auxiliary heads (training only, never inputs):
cube position (3, standardised), cube orientation (4: sin 4psi, cos 4psi, tilt x, tilt y), clean joint positions (16).
"""

from __future__ import annotations

import torch
from torch import nn

N_FRAMES, N_J = 10, 16
TOKEN_DIM = 2 * N_J
OBS_DIM = N_FRAMES * TOKEN_DIM
AUX_WEIGHTS = {"pos": 0.5, "ori": 0.3, "joint": 0.2}


def to_tokens(obs: torch.Tensor) -> torch.Tensor:
  """(B, 320) term-major -> (B, 10, 32): frame i = [joint_pos_i (16), prev_cmd_i (16)], oldest first."""
  B = obs.shape[0]
  return obs.reshape(B, 2, N_FRAMES, N_J).permute(0, 2, 1, 3).reshape(B, N_FRAMES, TOKEN_DIM)


class TokenNorm(nn.Module):
  """Per token feature (32) mean/std, fitted once on the first rollout, then frozen (eps 1e-2 as rsl_rl)."""

  def __init__(self):
    super().__init__()
    self.register_buffer("mean", torch.zeros(TOKEN_DIM))
    self.register_buffer("std", torch.ones(TOKEN_DIM))
    self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

  @torch.no_grad()
  def fit(self, obs: torch.Tensor) -> None:
    t = to_tokens(obs).reshape(-1, TOKEN_DIM).double()
    self.mean.copy_(t.mean(0).float())
    self.std.copy_(t.std(0).float())
    self.fitted.fill_(True)

  def forward(self, tok: torch.Tensor) -> torch.Tensor:
    return (tok - self.mean) / (self.std + 1e-2)


class _Heads(nn.Module):
  def __init__(self, d: int):
    super().__init__()
    self.action = nn.Linear(d, N_J)
    nn.init.zeros_(self.action.weight)
    nn.init.zeros_(self.action.bias)
    self.pos = nn.Linear(d, 3)
    self.ori = nn.Linear(d, 4)
    self.joint = nn.Linear(d, N_J)

  def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"action": self.action(z), "pos": self.pos(z), "ori": self.ori(z), "joint": self.joint(z)}


class TransformerStudent(nn.Module):
  """S-TF: 10 frame tokens + CLS, learned positional embedding, 3 x (4 heads, width 256, FF 1024), CLS -> heads."""

  kind = "transformer"

  def __init__(self, d: int = 256, layers: int = 3, heads: int = 4, ff: int = 1024):
    super().__init__()
    self.cfg = {"kind": self.kind, "d": d, "layers": layers, "heads": heads, "ff": ff}
    self.norm = TokenNorm()
    self.embed = nn.Linear(TOKEN_DIM, d)
    self.cls = nn.Parameter(torch.zeros(1, 1, d))
    self.pos_emb = nn.Parameter(torch.randn(1, N_FRAMES + 1, d) * 0.02)
    layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
    self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
    self.final_norm = nn.LayerNorm(d)
    self.heads = _Heads(d)

  def forward(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
    x = self.embed(self.norm(to_tokens(obs)))
    x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1) + self.pos_emb
    z = self.final_norm(self.encoder(x)[:, 0])
    return self.heads(z)


class MLPStudent(nn.Module):
  """S-MLP: C5's actor architecture (320 -> 512-512-256 ELU) with the same normalisation and heads."""

  kind = "mlp"

  def __init__(self, hidden=(512, 512, 256)):
    super().__init__()
    self.cfg = {"kind": self.kind, "hidden": list(hidden)}
    self.norm = TokenNorm()
    layers, d = [], OBS_DIM
    for h in hidden:
      layers += [nn.Linear(d, h), nn.ELU()]
      d = h
    self.mlp = nn.Sequential(*layers)
    self.heads = _Heads(d)

  def forward(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
    return self.heads(self.mlp(self.norm(to_tokens(obs)).reshape(obs.shape[0], OBS_DIM)))


def build(cfg: dict) -> nn.Module:
  if cfg["kind"] == "transformer":
    return TransformerStudent(cfg["d"], cfg["layers"], cfg["heads"], cfg["ff"])
  if cfg["kind"] == "mlp":
    return MLPStudent(tuple(cfg["hidden"]))
  raise KeyError(cfg["kind"])


def save(model: nn.Module, path, **meta) -> None:
  torch.save({"cfg": model.cfg, "state_dict": model.state_dict(), **meta}, path)


def load(path, device="cpu") -> nn.Module:
  ck = torch.load(path, map_location=device, weights_only=False)
  m = build(ck["cfg"]).to(device)
  m.load_state_dict(ck["state_dict"])
  return m.eval()


# ------------------------------------------------------------------------------------------- targets
def aux_targets(cube_state: torch.Tensor, critic: torch.Tensor) -> dict[str, torch.Tensor]:
  """cube_state: (B, 17) = pos 3 | quat (w,x,y,z) 4 | lin 3 | ang 3 | contact 4 (palm frame).
  critic: (B, 83), first 16 = clean joint positions relative to default."""
  from in_hand_rotation_mjlab.estimation.rotations import quat_to_matrix
  from in_hand_rotation_mjlab.scratch_study.features import P_MEAN, P_STD, canonical_rotation

  dev, dt = cube_state.device, cube_state.dtype
  pos = ((cube_state[:, 0:3] - torch.tensor(P_MEAN, device=dev, dtype=dt)) / torch.tensor(P_STD, device=dev, dtype=dt)).clamp(-5, 5)
  q = cube_state[:, 3:7]
  Rc = canonical_rotation(quat_to_matrix(q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)))
  psi = torch.atan2(Rc[:, 1, 0], Rc[:, 0, 0])
  ori = torch.stack([torch.sin(4 * psi), torch.cos(4 * psi), Rc[:, 0, 2], Rc[:, 1, 2]], dim=-1)
  return {"pos": pos, "ori": ori, "joint": critic[:, :N_J]}


def loss_terms(out: dict, tanh_teacher: torch.Tensor, tgt: dict) -> dict[str, torch.Tensor]:
  return {"action": ((torch.tanh(out["action"]) - tanh_teacher) ** 2).mean(),
          "pos": ((out["pos"] - tgt["pos"]) ** 2).mean(), "ori": ((out["ori"] - tgt["ori"]) ** 2).mean(),
          "joint": ((out["joint"] - tgt["joint"]) ** 2).mean()}


def yaw_from_ori(ori: torch.Tensor) -> torch.Tensor:
  """Orientation head -> psi in (-pi/4, pi/4] (yaw modulo 90 deg)."""
  return torch.atan2(ori[..., 0], ori[..., 1]) / 4.0


def n_params(m: nn.Module) -> int:
  return int(sum(p.numel() for p in m.parameters()))

