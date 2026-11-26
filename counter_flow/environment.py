"""Downward force field and two passages, each exactly two disc diameters."""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

RADIUS = 0.02
STEP = 0.015
GOAL_RADIUS = 0.04
IMAGE_SIZE = 98
SUBSTEPS = 8


@dataclass(frozen=True)
class Layout:
    group: str
    difficulty: str
    band_y: float
    band_height: float
    mirror: bool = False
    gap_width: float = 0.08
    current_strength: float = 1.25

    def __post_init__(self):
        if abs(self.gap_width - 4 * RADIUS) > 1e-12:
            raise ValueError("Gap must equal two diameters")
        if self.current_strength <= 1:
            raise ValueError("Current must overpower upward control")

    @property
    def uid(self):
        return self.group + ("_mirror" if self.mirror else "_original")

    @property
    def bottom(self):
        return self.band_y - self.band_height / 2

    @property
    def top(self):
        return self.band_y + self.band_height / 2


def in_current(layout, positions):
    p = np.asarray(positions, np.float64)
    return (
        (p[..., 0] >= layout.gap_width)
        & (p[..., 0] <= 1 - layout.gap_width)
        & (p[..., 1] >= layout.bottom)
        & (p[..., 1] <= layout.top)
    )


def valid_position(layout, positions, clearance=0.0):
    p = np.asarray(positions)
    r = RADIUS + clearance
    return (p >= r - 1e-10).all(-1) & (p <= 1 - r + 1e-10).all(-1)


def transition(layout, position, action):
    a = np.asarray(action, np.float64)
    if a.shape != (2,) or not np.isfinite(a).all():
        raise ValueError("Invalid action")
    a = np.clip(a, -1, 1)
    a = a / max(1.0, np.linalg.norm(a))
    p = np.array(position, np.float64, copy=True)
    contacts = []
    for _ in range(SUBSTEPS):
        v = a.copy()
        if in_current(layout, p):
            v[1] -= layout.current_strength
        q = p + STEP / SUBSTEPS * v
        p = np.clip(q, RADIUS, 1 - RADIUS)
        if not np.array_equal(p, q):
            contacts.append("boundary")
    return p, a.astype(np.float32), list(dict.fromkeys(contacts))


def background(layout):
    n = IMAGE_SIZE * 3
    im = Image.new("RGB", (n, n), (242, 244, 247))
    draw = ImageDraw.Draw(im)
    x0 = layout.gap_width * (n - 1)
    x1 = (1 - layout.gap_width) * (n - 1)
    y0 = (1 - layout.top) * (n - 1)
    y1 = (1 - layout.bottom) * (n - 1)
    draw.rectangle(
        (x0, y0, x1, y1), fill=(167, 203, 251), outline=(75, 139, 223), width=1
    )
    for x in np.linspace(x0 + 12, x1 - 12, 9):
        c = (y0 + y1) / 2
        draw.line((x, c - 8, x, c + 5), fill=(33, 104, 208), width=3)
        draw.polygon(((x - 4, c + 2), (x + 4, c + 2), (x, c + 9)), fill=(33, 104, 208))
    draw.rectangle((0, 0, n - 1, n - 1), outline=(55, 64, 76), width=2)
    return im


def render_position(layout, position, base=None):
    im = (base if base is not None else background(layout)).copy()
    n = im.width
    x, y = position
    r = RADIUS * (n - 1)
    cx = x * (n - 1)
    cy = (1 - y) * (n - 1)
    ImageDraw.Draw(im).ellipse((cx - r, cy - r, cx + r, cy + r), fill=(31, 119, 220))
    return np.array(im.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS))


class CounterFlowEnv:
    """Counter-flow dynamics with image observations and goal-based termination."""

    def __init__(self, layout, max_steps=400):
        self.layout = layout
        self.max_steps = max_steps
        self.base = background(layout)
        self.position = None
        self.steps = 0

    def reset(self, *, start, goal):
        if not valid_position(self.layout, start) or not valid_position(
            self.layout, goal
        ):
            raise ValueError("Invalid reset")
        self.position = np.array(start, np.float64)
        self.goal = np.array(goal, np.float64)
        self.steps = 0
        return self.render(), {}

    def step(self, action):
        if self.position is None or self.steps >= self.max_steps:
            raise RuntimeError("Reset required")
        active = bool(in_current(self.layout, self.position))
        self.position, a, c = transition(self.layout, self.position, action)
        self.steps += 1
        hit = np.linalg.norm(self.position - self.goal) <= GOAL_RADIUS
        return (
            self.render(),
            float(hit),
            bool(hit),
            self.steps >= self.max_steps,
            dict(
                position=self.position.copy(),
                applied_action=a,
                contacts=c,
                step=self.steps,
                current_active=active,
            ),
        )

    def render(self):
        return render_position(self.layout, self.position, self.base)

    def goal_image(self):
        return render_position(self.layout, self.goal, self.base)
