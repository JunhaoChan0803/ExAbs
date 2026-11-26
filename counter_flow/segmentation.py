"""AM-guided temporal boundaries and padded high-level training windows."""

from __future__ import annotations

import numpy as np
import torch

from .models import AM, calibrated


@torch.no_grad()
def score_edges(am, calibration, features, minimum_gap=6, maximum_gap=14):
    """Score observed edges from [T+1,C,H,W] features as log(q)-u.
    q is the calibrated CDF minus head disagreement u; index by start and duration.
    """
    if not 1 <= minimum_gap <= maximum_gap:
        raise ValueError("Segment gaps must be positive and ordered")
    if am.training:
        raise ValueError("Freeze AM in evaluation mode before selecting boundaries")
    device = next(am.parameters()).device
    features = torch.as_tensor(features, dtype=torch.float32, device=device)
    if features.ndim != 4 or len(features) <= minimum_gap:
        raise ValueError("Expected a trajectory of spatial feature maps")
    embeddings = torch.cat(
        [
            am.frame(features[start : start + 64])
            for start in range(0, len(features), 64)
        ]
    )
    sources, targets, durations = [], [], []
    for start in range(len(features)):
        for duration in range(minimum_gap, maximum_gap + 1):
            if start + duration < len(features):
                sources.append(start)
                targets.append(start + duration)
                durations.append(duration)
    edges = np.full((len(features), maximum_gap + 1), -np.inf)
    for start in range(0, len(sources), 512):
        source = torch.tensor(sources[start : start + 512], device=device)
        target = torch.tensor(targets[start : start + 512], device=device)
        duration = torch.tensor(durations[start : start + 512], device=device)
        logits = am.encoded(embeddings[source], embeddings[target])
        if maximum_gap > logits.shape[-1]:
            raise ValueError("A segment duration exceeds the AM prediction support")
        aggregate = calibrated(AM.cdf(logits.mean(0)), calibration)
        members = calibrated(AM.cdf(logits), calibration)
        probability = aggregate.gather(1, (duration - 1)[:, None])[:, 0]
        uncertainty = members.std(0, unbiased=False).gather(1, (duration - 1)[:, None])[
            :, 0
        ]
        corrected = probability - uncertainty
        utility = corrected.clamp_min(1e-6).log() - uncertainty
        edges[
            np.asarray(sources[start : start + 512]),
            np.asarray(durations[start : start + 512]),
        ] = utility.cpu().numpy()
    return edges


def optimal_boundaries(
    edges, start, window_steps=40, segments=4, minimum_gap=6, maximum_gap=14
):
    """Maximize edge utility with fixed window endpoints and segment count.

    Durations are visited in ascending order. Only a strict score improvement
    replaces a predecessor, which makes tie handling deterministic.
    """
    edges = np.asarray(edges)
    if (
        edges.ndim != 2
        or start < 0
        or start + window_steps >= len(edges)
        or window_steps <= 0
        or segments <= 0
        or not 1 <= minimum_gap <= maximum_gap < edges.shape[1]
    ):
        raise ValueError("Invalid edge table or segmentation window")
    score = np.full((segments + 1, window_steps + 1), -np.inf)
    previous = np.full((segments + 1, window_steps + 1), -1, dtype=int)
    score[0, 0] = 0
    for segment in range(1, segments + 1):
        for end in range(1, window_steps + 1):
            for duration in range(minimum_gap, maximum_gap + 1):
                begin = end - duration
                if begin < 0 or not np.isfinite(score[segment - 1, begin]):
                    continue
                value = score[segment - 1, begin] + edges[start + begin, duration]
                if value > score[segment, end]:
                    score[segment, end] = value
                    previous[segment, end] = begin
    if not np.isfinite(score[segments, window_steps]):
        raise ValueError("No admissible segmentation covers the requested window")
    end = window_steps
    boundaries = [end]
    for segment in range(segments, 0, -1):
        end = int(previous[segment, end])
        boundaries.append(end)
    return np.array(boundaries[::-1]) + start


def training_windows(features, actions, boundaries, maximum_gap=14):
    """Gather observed features, actions [B,S,D,2], and masks [B,S,D].
    Boundaries have shape [B,S+1]; observed targets are detached.
    """
    features = torch.as_tensor(features)
    actions = torch.as_tensor(actions, dtype=torch.float32, device=features.device)
    bounds = torch.as_tensor(boundaries, dtype=torch.long, device=features.device)
    if features.ndim != 4 or actions.shape != (len(features) - 1, 2):
        raise ValueError("Expected T+1 feature maps and T two-dimensional actions")
    if bounds.ndim != 2 or bounds.shape[1] < 2 or maximum_gap <= 0:
        raise ValueError("Expected a batch of ordered segment boundaries")
    gaps = bounds[:, 1:] - bounds[:, :-1]
    if bool(
        (bounds < 0).any()
        or (bounds >= len(features)).any()
        or (gaps < 1).any()
        or (gaps > maximum_gap).any()
    ):
        raise ValueError("Boundary indices or action-chunk lengths are invalid")
    batch, segments = gaps.shape
    chunks = actions.new_zeros((batch, segments, maximum_gap, 2))
    masks = torch.zeros(
        (batch, segments, maximum_gap), dtype=torch.bool, device=features.device
    )
    for row in range(batch):
        for segment in range(segments):
            start, end = bounds[row, segment : segment + 2].tolist()
            chunks[row, segment, : end - start] = actions[start:end]
            masks[row, segment, : end - start] = True
    return (
        features[bounds[:, :-1]].detach(),
        features[bounds[:, 1:]].detach(),
        chunks,
        masks,
    )
