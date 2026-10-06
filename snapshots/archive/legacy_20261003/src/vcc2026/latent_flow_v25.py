"""V25 prior-anchored latent transport and independent count readouts.

The caller supplies legal training cells, a frozen role-specific NB decoder,
and prior-anchored destination-control PCs. This module reads no files or target
outcomes. Flow matching uses genuine interpolation times and velocity targets;
unpaired technical-cell coupling is not a biological trajectory.

The final experiment keeps the historical integer generator. NB is a masked
expected-count readout of transported latents, not a replacement stochastic
count generator. Both readouts average expected COUNTS before taking log ratios.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any, NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


DEFAULT_CONFIG = {
    "latent_dim": 64, "z_dim": 130, "h_dim": 64,
    "condition_hidden": 192, "velocity_hidden": 512, "velocity_layers": 2,
    "decoder_hidden": 256, "decoder_layers": 2,
    "theta_floor": 1e-4, "initial_theta": 10.0, "correction_bound": 6.0,
}


def config(values: Mapping[str, Any] | None = None) -> dict[str, Any]:
    result = {**DEFAULT_CONFIG, **(dict(values) if values is not None else {})}
    for name in ("latent_dim", "z_dim", "h_dim", "condition_hidden", "velocity_hidden",
                 "velocity_layers", "decoder_hidden", "decoder_layers"):
        if int(result[name]) < 1:
            raise ValueError(f"{name} must be positive")
    if result["theta_floor"] <= 0 or result["initial_theta"] <= result["theta_floor"]:
        raise ValueError("Initial NB inverse dispersion must exceed a positive floor")
    if result["correction_bound"] <= 0:
        raise ValueError("Correction bound must be positive")
    return result


def _moments(cells: Tensor) -> Tensor:
    if cells.ndim != 3 or cells.shape[1] < 1:
        raise ValueError("Source cells must have shape (batch, positive_cells, latent_dim)")
    return torch.cat((cells.mean(1), cells.std(1, correction=0)), dim=-1)


def _condition_like(condition: Tensor, state: Tensor) -> Tensor:
    if condition.ndim == 2 and state.ndim == 3 and condition.shape[0] == state.shape[0]:
        condition = condition[:, None, :]
    try:
        return torch.broadcast_to(condition, (*state.shape[:-1], condition.shape[-1]))
    except RuntimeError as error:
        raise ValueError("Condition and latent-state batch shapes disagree") from error


def _scalar_like(value: Tensor | float, leading: tuple[int, ...], reference: Tensor) -> Tensor:
    result = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if len(leading) == 2 and result.ndim == 1 and result.shape[0] == leading[0]:
        result = result[:, None]
    try:
        return torch.broadcast_to(result, leading)
    except RuntimeError as error:
        raise ValueError("Scalar/time/library batch shape disagrees with latent cells") from error


def _gene_mask(measurement: Tensor, leading: tuple[int, ...], ng: int, reference: Tensor) -> Tensor:
    mask = torch.as_tensor(measurement, device=reference.device)
    if mask.shape[-1] != ng:
        raise ValueError("Measurement mask gene axis disagrees")
    if len(leading) == 2 and mask.ndim == 2 and mask.shape[0] == leading[0]:
        mask = mask[:, None, :]
    try:
        mask = torch.broadcast_to(mask, (*leading, ng))
    except RuntimeError as error:
        raise ValueError("Measurement mask batch shape disagrees") from error
    if torch.any((mask != 0) & (mask != 1)):
        raise ValueError("Measurement mask must identify observed genes with binary entries")
    mask = mask.bool()
    if torch.any(mask.sum(-1) == 0):
        raise ValueError("Every cell needs at least one actually measured readout gene")
    return mask


class LatentResponse(nn.Module):
    """Exactly matched deterministic/flow conditional-vector-field backbone.

    The deterministic endpoint uses the same network at fixed t=0, without ODE
    integration. The flow uses t-dependent velocities. This matches allocated
    parameters; the deterministic arm does not learn variation in the time input.
    Both heads start at zero and therefore preserve prior-anchored x0 exactly.
    """

    def __init__(self, cfg: Mapping[str, Any] | None = None):
        super().__init__()
        self.cfg = c = config(cfg)
        hidden = c["condition_hidden"]
        cond_dim = c["z_dim"] + c["h_dim"] + 1 + 4 * c["latent_dim"]
        self.condition_projection = nn.Sequential(nn.Linear(cond_dim, hidden), nn.SiLU(),
                                                 nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        width = c["velocity_hidden"]
        modules: list[nn.Module] = [nn.Linear(c["latent_dim"] + hidden + 1, width), nn.SiLU()]
        for _ in range(c["velocity_layers"] - 1):
            modules.extend((nn.Linear(width, width), nn.SiLU()))
        modules.append(nn.Linear(width, c["latent_dim"]))
        self.field = nn.Sequential(*modules)
        nn.init.zeros_(self.field[-1].weight)
        nn.init.zeros_(self.field[-1].bias)

    def condition(self, z: Tensor, h: Tensor, source_control: Tensor,
                  source_perturbed: Tensor, source_valid: Tensor) -> Tensor:
        if z.ndim != 2 or h.ndim != 2 or source_valid.shape != (len(z),):
            raise ValueError("z/h must be matrices and source_valid a per-target vector")
        if z.shape[-1] != self.cfg["z_dim"] or h.shape[-1] != self.cfg["h_dim"]:
            raise ValueError("z/h feature dimensions disagree with frozen role features")
        features = torch.cat((z, h, source_valid[:, None], _moments(source_control),
                              _moments(source_perturbed)), dim=-1)
        return self.condition_projection(features)

    def velocity(self, xt: Tensor, t: Tensor | float, condition: Tensor) -> Tensor:
        if xt.ndim not in (2, 3) or xt.shape[-1] != self.cfg["latent_dim"]:
            raise ValueError("Latent cells must have shape (N,D) or (B,K,D)")
        time = _scalar_like(t, tuple(xt.shape[:-1]), xt)
        if torch.any((time < 0) | (time > 1)):
            raise ValueError("Flow interpolation time must be in [0,1]")
        context = _condition_like(condition, xt)
        return self.field(torch.cat((xt, time[..., None], context), dim=-1))

    def deterministic(self, x0: Tensor, condition: Tensor) -> Tensor:
        return x0 + self.velocity(x0, 0.0, condition)

    def endpoint(self, x0: Tensor, condition: Tensor, steps: int = 8,
                 method: str = "euler") -> Tensor:
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("ODE steps must be a positive integer")
        if method not in ("euler", "midpoint"):
            raise ValueError("Supported fixed ODE methods are euler and midpoint")
        state = x0
        dt = 1.0 / steps
        for index in range(steps):
            time = index * dt
            velocity = self.velocity(state, time, condition)
            if method == "midpoint":
                midpoint = state + .5 * dt * velocity
                velocity = self.velocity(midpoint, time + .5 * dt, condition)
            state = state + dt * velocity
        return state


def flow_matching_targets(x0: Tensor, x1: Tensor, t: Tensor | float) -> tuple[Tensor, Tensor]:
    if x0.shape != x1.shape or x0.ndim not in (2, 3):
        raise ValueError("Unpaired start/end sample arrays must have matching latent shapes")
    time = _scalar_like(t, tuple(x0.shape[:-1]), x0)
    if torch.any((time < 0) | (time > 1)):
        raise ValueError("Flow interpolation time must be in [0,1]")
    return (1 - time[..., None]) * x0 + time[..., None] * x1, x1 - x0


def _latent_mse(prediction: Tensor, target: Tensor, pair_mask: Tensor | None = None) -> Tensor:
    error = (prediction - target).square().mean(-1)
    if pair_mask is None:
        return error.mean()
    mask = _scalar_like(pair_mask, tuple(error.shape), error)
    if torch.any((mask != 0) & (mask != 1)):
        raise ValueError("Latent auxiliary eligibility must be a binary cell mask")
    if torch.any(mask != 0):
        return torch.where(mask != 0, error, torch.zeros_like(error)).sum() / mask.sum()
    # HCT mean-only batches must have an exactly zero auxiliary term with a
    # valid computation graph; they do not become synthetic flow labels.
    return error.sum() * 0


def _eligible_endpoints(x0: Tensor, x1: Tensor, pair_mask: Tensor | None) -> Tensor:
    if x0.shape != x1.shape:
        raise ValueError("Unpaired start/end arrays must have matching latent shapes")
    if pair_mask is None:
        return x1
    mask = _scalar_like(pair_mask, tuple(x0.shape[:-1]), x0)
    if torch.any((mask != 0) & (mask != 1)):
        raise ValueError("Latent auxiliary eligibility must be a binary cell mask")
    # A mean-only background may use NaN as its explicit absent-cell-label
    # sentinel. It contributes no interpolation/velocity/end-point label.
    return torch.where(mask[..., None] != 0, x1, x0)


def flow_matching_loss(model: LatentResponse, x0: Tensor, x1: Tensor, condition: Tensor,
                       t: Tensor | None = None, pair_mask: Tensor | None = None,
                       generator: torch.Generator | None = None) -> Tensor:
    if t is None:
        t = torch.rand(x0.shape[:-1], dtype=x0.dtype, device=x0.device, generator=generator)
    x1 = _eligible_endpoints(x0, x1, pair_mask)
    xt, target_velocity = flow_matching_targets(x0, x1, t)
    return _latent_mse(model.velocity(xt, t, condition), target_velocity, pair_mask)


def deterministic_endpoint_loss(model: LatentResponse, x0: Tensor, x1: Tensor,
                                condition: Tensor, pair_mask: Tensor | None = None) -> Tensor:
    x1 = _eligible_endpoints(x0, x1, pair_mask)
    return _latent_mse(model.deterministic(x0, condition), x1, pair_mask)


class DecoderOutput(NamedTuple):
    mu: Tensor
    theta: Tensor | None
    logits: Tensor


class NBDecoder(nn.Module):
    """Independent raw-count NB decoder with actual-axis library offsets.

    ``measurement`` is the actual likelihood axis during decoder training and
    the declared common readout axis during transport prediction. The supplied
    library MUST be the sum on that same axis. No library is estimated from PCs.
    NB inverse dispersion is gene-specific; no target/context one-hot is used.
    """

    def __init__(self, ng: int, cfg: Mapping[str, Any] | None = None):
        super().__init__()
        if ng < 1:
            raise ValueError("NB output gene count must be positive")
        self.ng = int(ng)
        self.cfg = c = config(cfg)
        width = c["decoder_hidden"]
        modules: list[nn.Module] = [nn.Linear(c["latent_dim"], width), nn.SiLU()]
        for _ in range(c["decoder_layers"] - 1):
            modules.extend((nn.Linear(width, width), nn.SiLU()))
        modules.append(nn.Linear(width, ng))
        self.network = nn.Sequential(*modules)
        initial = c["initial_theta"] - c["theta_floor"]
        inverse_softplus = initial + math.log(-math.expm1(-initial))
        self.dispersion_unconstrained = nn.Parameter(torch.full((ng,), inverse_softplus))

    def forward(self, pc: Tensor, library: Tensor, measurement: Tensor) -> DecoderOutput:
        if pc.ndim not in (2, 3) or pc.shape[-1] != self.cfg["latent_dim"]:
            raise ValueError("PC decoder input must have shape (N,D) or (B,K,D)")
        mask = _gene_mask(measurement, tuple(pc.shape[:-1]), self.ng, pc)
        totals = _scalar_like(library, tuple(pc.shape[:-1]), pc)
        if torch.any(~torch.isfinite(totals)) or torch.any(totals <= 0):
            raise ValueError("Library offsets must be finite positive actual-axis count sums")
        logits = self.network(pc)
        masked_logits = logits.masked_fill(~mask, -torch.inf)
        probabilities = F.softmax(masked_logits, dim=-1)
        mu = totals[..., None] * probabilities
        theta = F.softplus(self.dispersion_unconstrained) + self.cfg["theta_floor"]
        return DecoderOutput(mu, theta, logits)


def masked_nb_nll(counts: Tensor, mu: Tensor, theta: Tensor, measurement: Tensor,
                  reduction: str = "mean", validate_counts: bool = True) -> Tensor:
    """NB2 negative log likelihood on measured coordinates only.

    Reduction mean weights observed cell-gene coordinates equally. Integer
    labels may be stored in float32, but PC or log-normalized values are invalid.
    ``validate_counts=False`` is appropriate only after a cache-level validation.
    """
    if not isinstance(theta, Tensor):
        raise ValueError("NB likelihood requires NB inverse dispersion, not a PCA readout")
    if counts.shape != mu.shape or counts.ndim not in (2, 3):
        raise ValueError("Counts and NB means must have matching cell-gene shapes")
    mask = _gene_mask(measurement, tuple(counts.shape[:-1]), counts.shape[-1], counts)
    if validate_counts:
        observed = counts[mask]
        if torch.any(~torch.isfinite(observed)) or torch.any(observed < 0) or torch.any(observed != observed.round()):
            raise ValueError("NB supervision must be finite nonnegative integer raw counts")
    if torch.any(~torch.isfinite(mu)) or torch.any(mu < 0) or torch.any(~torch.isfinite(theta)) or torch.any(theta <= 0):
        raise ValueError("NB means/dispersion must be finite and nonnegative/positive")
    # Unmeasured values do not enter the likelihood, even if caller retains a
    # sentinel outside its likelihood mask.
    x = torch.where(mask, counts, torch.zeros_like(counts))
    mean = mu.clamp_min(1e-8)
    inverse_dispersion = theta.to(dtype=mu.dtype, device=mu.device)
    log_total = torch.log(inverse_dispersion + mean)
    log_probability = (torch.lgamma(x + inverse_dispersion) - torch.lgamma(inverse_dispersion)
                       - torch.lgamma(x + 1) + inverse_dispersion * (torch.log(inverse_dispersion) - log_total)
                       + x * (torch.log(mean) - log_total))
    nll = torch.where(mask, -log_probability, torch.zeros_like(log_probability))
    if reduction == "none":
        return nll
    if reduction == "sum":
        return nll.sum()
    if reduction == "mean":
        return nll.sum() / mask.sum()
    raise ValueError("NB reduction must be none, sum or mean")


class PCAInverseDecoder(nn.Module):
    """Frozen rank-64 inverse PCA, mapped to the same common count-readout axis.

    Truncated PCA is an approximate reconstruction. Negative reconstructed log
    expression is clipped at zero; values above log1p(50000) are clipped at that
    physical common-axis normalization bound. Expected nonnegative weights are
    renormalized to the supplied common-axis library. It has no NB likelihood.
    """

    def __init__(self, encoder: Mapping[str, Any], output_genes: Sequence[str],
                 input_genes: Sequence[str] | None = None):
        super().__init__()
        genes = [str(gene) for gene in encoder["genes"]]
        if input_genes is not None and list(map(str, input_genes)) != genes:
            raise ValueError("PCA readout input axis differs from its frozen encoder")
        output = list(map(str, output_genes))
        if len(output) != len(set(output)) or len(genes) != len(set(genes)):
            raise ValueError("PCA readout gene axes must be unique")
        positions = {gene: i for i, gene in enumerate(output)}
        if not all(gene in positions for gene in genes):
            raise ValueError("Every common input gene must be represented on the output axis")
        self.ng = len(output)
        self.latent_dim = int(encoder["basis"].shape[0])
        self.normalization = 50000.0
        self.input_genes = genes
        self.output_genes = output
        for name in ("basis", "gene_center", "gene_scale", "pc_scale"):
            self.register_buffer(name, torch.as_tensor(encoder[name], dtype=torch.float32).clone())
        self.register_buffer("input_to_output", torch.tensor([positions[gene] for gene in genes], dtype=torch.long))
        common = torch.zeros(self.ng, dtype=torch.bool)
        common[self.input_to_output] = True
        self.register_buffer("common_mask", common)

    def forward(self, pc: Tensor, library: Tensor, measurement: Tensor) -> DecoderOutput:
        if pc.ndim not in (2, 3) or pc.shape[-1] != self.latent_dim:
            raise ValueError("PCA readout latent dimension disagrees")
        mask = _gene_mask(measurement, tuple(pc.shape[:-1]), self.ng, pc)
        if torch.any(mask & ~self.common_mask):
            raise ValueError("Inverse PCA readout cannot declare genes outside its actual common axis measured")
        totals = _scalar_like(library, tuple(pc.shape[:-1]), pc)
        if torch.any(~torch.isfinite(totals)) or torch.any(totals <= 0):
            raise ValueError("PCA readout needs positive actual common-axis library sizes")
        reconstructed_log = ((pc * self.pc_scale) @ self.basis) * self.gene_scale + self.gene_center
        common_weights = torch.expm1(reconstructed_log.clamp(0, math.log1p(self.normalization)))
        weights = pc.new_zeros((*pc.shape[:-1], self.ng))
        weights[..., self.input_to_output] = common_weights
        weights = weights * mask
        sums = weights.sum(-1, keepdim=True)
        fallback = mask.to(pc.dtype) / mask.sum(-1, keepdim=True)
        probabilities = torch.where(sums > 0, weights / sums.clamp_min(1e-12), fallback)
        mu = totals[..., None] * probabilities
        logits = torch.log(weights.clamp_min(1e-12)).masked_fill(~mask, -torch.inf)
        return DecoderOutput(mu, None, logits)


def readout_delta(endpoint: Tensor, base_pc: Tensor, library: Tensor,
                  measurement: Tensor, decoder: nn.Module, pseudocount: float = .5) -> Tensor:
    """Signed correction from paired readouts of transported/base latent cells.

    Cell expected counts are averaged arithmetically before log2; this is never
    mean(log count) or a log-normalized label interpreted as a raw count.
    The same axis/library vector must be supplied to both endpoint and base.
    """
    if endpoint.shape != base_pc.shape or pseudocount <= 0:
        raise ValueError("Base/end latent shapes must match and pseudocount must be positive")
    endpoint_counts = decoder(endpoint, library, measurement).mu
    base_counts = decoder(base_pc, library, measurement).mu
    if endpoint_counts.ndim == 3:
        endpoint_mean = endpoint_counts.mean(1)
        base_mean = base_counts.mean(1)
    elif endpoint_counts.ndim == 2:
        endpoint_mean, base_mean = endpoint_counts, base_counts
    else:
        raise ValueError("Readout accepts one cell or a cell set per condition")
    return torch.log2((endpoint_mean + pseudocount) / (base_mean + pseudocount))


def merge_prior(raw_prior: Tensor, correction: Tensor, seen: Tensor, ownmask: Tensor,
                measurement: Tensor, bound: float = 6.0) -> Tensor:
    """Full V1 plus signed bounded correction; protected entries exact V1."""
    if raw_prior.shape != correction.shape or ownmask.shape != raw_prior.shape or raw_prior.ndim != 2:
        raise ValueError("Prior/correction/own mask must have matching (targets, genes) shape")
    if bound <= 0:
        raise ValueError("Signed correction bound must be positive")
    mask = _gene_mask(measurement, tuple(raw_prior.shape[:-1]), raw_prior.shape[-1], raw_prior)
    if seen.shape not in ((raw_prior.shape[-1],), raw_prior.shape):
        raise ValueError("seen must have shape (genes,) or (targets, genes)")
    editable = mask & seen.bool() & ownmask.bool()
    signed = bound * torch.tanh(correction / bound)
    return torch.where(editable, raw_prior + signed, raw_prior)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
