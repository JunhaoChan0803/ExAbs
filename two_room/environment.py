"""Deterministic continuous disc physics and observation renderer (CPU only)."""

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

RADIUS = 0.02
STEP = 0.015
GOAL_RADIUS = 0.04
IMAGE_SIZE = 98


@dataclass(frozen=True)
class Layout:
    group: str
    difficulty: str
    wall_x: float
    upper_y: float
    lower_y: float
    lower_width: float
    mirror: bool = False
    upper_width: float = 0.12
    wall_width: float = 0.04

    @property
    def uid(self):
        return self.group + ("_mirror" if self.mirror else "_original")

    def canonical(self, p):
        p = np.asarray(p, dtype=np.float64).copy()
        if self.mirror:
            p[..., 0] = 1 - p[..., 0]
        return p

    def physical(self, p):
        return self.canonical(p)

    def rectangles(self):
        x0, x1 = (
            self.wall_x - 0.5 * self.wall_width,
            self.wall_x + 0.5 * self.wall_width,
        )
        lo0, lo1 = (
            self.lower_y - 0.5 * self.lower_width,
            self.lower_y + 0.5 * self.lower_width,
        )
        hi0, hi1 = (
            self.upper_y - 0.5 * self.upper_width,
            self.upper_y + 0.5 * self.upper_width,
        )
        return [(x0, 0.0, x1, lo0), (x0, lo1, x1, hi0), (x0, hi1, x1, 1.0)]


def _sweep_disc_rect(p, v, rect, radius=RADIUS):
    """Exact first contact with rectangle Minkowski-summed with a disc."""
    x0, y0, x1, y1 = rect
    hits = []
    # Four straight portions of the rounded rectangle.
    for axis, bound, sign, alo, ahi in (
        (0, x0 - radius, -1.0, y0, y1),
        (0, x1 + radius, 1.0, y0, y1),
        (1, y0 - radius, -1.0, x0, x1),
        (1, y1 + radius, 1.0, x0, x1),
    ):
        if v[axis] * sign >= -1e-14:
            continue
        t = (bound - p[axis]) / v[axis]
        q = p[1 - axis] + t * v[1 - axis]
        if -1e-10 <= t <= 1.0 and alo - 1e-10 <= q <= ahi + 1e-10:
            n = np.zeros(2)
            n[axis] = sign
            hits.append((max(0.0, t), n))
    a = float(v @ v)
    if a > 1e-20:
        for cx, cy, sx, sy in (
            (x0, y0, -1, -1),
            (x0, y1, -1, 1),
            (x1, y0, 1, -1),
            (x1, y1, 1, 1),
        ):
            d = p - np.array([cx, cy])
            b = 2 * float(d @ v)
            c = float(d @ d) - radius**2
            disc = b * b - 4 * a * c
            if disc < 0:
                continue
            t = (-b - math.sqrt(max(0.0, disc))) / (2 * a)
            q = d + t * v
            if -1e-10 <= t <= 1.0 and q[0] * sx >= -1e-10 and q[1] * sy >= -1e-10:
                n = q / max(np.linalg.norm(q), 1e-15)
                if v @ n < -1e-14:
                    hits.append((max(0.0, t), n))
    return min(hits, key=lambda x: x[0]) if hits else None


def valid_position(layout, position):
    p = layout.canonical(position)
    if np.any(p < RADIUS - 1e-9) or np.any(p > 1 - RADIUS + 1e-9):
        return False
    for x0, y0, x1, y1 in layout.rectangles():
        q = np.clip(p, [x0, y0], [x1, y1])
        if np.linalg.norm(p - q) < RADIUS - 1e-8:
            return False
    return True


def transition(layout, position, action):
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (2,) or not np.isfinite(action).all():
        raise ValueError("Action must be a finite two-dimensional vector")
    action = np.clip(action, -1, 1)
    action = action / max(1.0, np.linalg.norm(action))
    p = layout.canonical(position)
    v = action * STEP
    if layout.mirror:
        v[0] *= -1
    obstacles = [(r, "wall") for r in layout.rectangles()]
    obstacles += [
        ((-2, -2, 0, 2), "boundary"),
        ((1, -2, 2, 2), "boundary"),
        ((-2, -2, 2, 0), "boundary"),
        ((-2, 1, 2, 2), "boundary"),
    ]
    contacts = []
    for _ in range(6):
        if np.linalg.norm(v) < 1e-12:
            break
        candidates = []
        for rect, kind in obstacles:
            hit = _sweep_disc_rect(p, v, rect)
            if hit is not None:
                candidates.append((*hit, kind))
        # Block right-to-left travel across the gate segment.
        barrier = layout.wall_x + RADIUS
        if v[0] < -1e-14 and p[0] >= barrier - 1e-9:
            t = (barrier - p[0]) / v[0]
            y = p[1] + t * v[1]
            if (
                -1e-10 <= t <= 1
                and abs(y - layout.upper_y) <= layout.upper_width / 2 + RADIUS
            ):
                candidates.append((max(t, 0.0), np.array([1.0, 0.0]), "one_way"))
        if not candidates:
            p = p + v
            break
        t, n, kind = min(candidates, key=lambda x: x[0])
        p = p + t * v + n * 1e-10
        v = (1 - t) * v
        v = v - min(0.0, float(v @ n)) * n
        contacts.append(kind)
    result = layout.physical(p)
    if not valid_position(layout, result):
        raise RuntimeError("Collision solver returned penetrating position")
    return result, action.astype(np.float32), contacts


def background(layout):
    # Supersample the image to preserve subpixel motion.
    n = IMAGE_SIZE * 3
    image = Image.new("RGB", (n, n), (242, 244, 247))
    draw = ImageDraw.Draw(image)

    def pt(x, y):
        return x * (n - 1), (1 - y) * (n - 1)

    for x0, y0, x1, y1 in layout.rectangles():
        draw.rectangle([pt(x0, y1), pt(x1, y0)], fill=(55, 64, 76))
    draw.line(
        [
            pt(layout.wall_x - 0.075, layout.upper_y),
            pt(layout.wall_x + 0.075, layout.upper_y),
        ],
        fill=(225, 135, 25),
        width=4,
    )
    draw.polygon(
        [
            pt(layout.wall_x + 0.075, layout.upper_y),
            pt(layout.wall_x + 0.04, layout.upper_y + 0.026),
            pt(layout.wall_x + 0.04, layout.upper_y - 0.026),
        ],
        fill=(225, 135, 25),
    )
    draw.rectangle([(0, 0), (n - 1, n - 1)], outline=(55, 64, 76), width=2)
    if layout.mirror:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return image


def render_position(layout, position, base=None):
    if base is None:
        base = background(layout)
    image = base.copy()
    draw = ImageDraw.Draw(image)
    n = image.width
    x, y = position
    cx = x * (n - 1)
    cy = (1 - y) * (n - 1)
    r = RADIUS * (n - 1)
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(31, 119, 220))
    return np.asarray(
        image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    ).copy()


class TwoRoom:
    def __init__(self, layout, max_steps=300):
        self.layout = layout
        self.max_steps = max_steps
        self.base = background(layout)
        self.position = None
        self.goal = None
        self.steps = 0

    def reset(self, *, start, goal):
        if not valid_position(self.layout, start) or not valid_position(
            self.layout, goal
        ):
            raise ValueError("Invalid reset position")
        self.position = np.array(start, dtype=np.float64)
        self.goal = np.array(goal, dtype=np.float64)
        self.steps = 0
        return self.render(), {}

    def step(self, action):
        if self.position is None or self.steps >= self.max_steps:
            raise RuntimeError("Reset required")
        old = self.position.copy()
        self.position, applied, contacts = transition(
            self.layout, self.position, action
        )
        self.steps += 1
        success = np.linalg.norm(self.position - self.goal) <= GOAL_RADIUS
        a, b = self.layout.canonical(old), self.layout.canonical(self.position)
        crossed_lower = (a[0] - self.layout.wall_x) * (
            b[0] - self.layout.wall_x
        ) < 0 and abs(b[1] - self.layout.lower_y) < self.layout.lower_width / 2
        return (
            self.render(),
            float(success),
            bool(success),
            self.steps >= self.max_steps,
            {
                "position": self.position.copy(),
                "applied_action": applied,
                "contacts": contacts,
                "crossed_lower": bool(crossed_lower),
                "step": self.steps,
            },
        )

    def render(self):
        return render_position(self.layout, self.position, self.base)

    def goal_image(self):
        return render_position(self.layout, self.goal, self.base)
