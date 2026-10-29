"""Generate the Tachikoma-style top cover (v200) and its press-in detail parts for Sesame,
starting from Top-Cover-No-Ears-v100.

The original cover is kept intact (screw bosses, display slot, switch slot, wire
channels, USB opening, inner cavity). New geometry is only ever added OUTSIDE the
cover envelope, and wherever it overhangs the original footprint it stays above
Y = 43 mm with a 45 deg underside, so the hip joints (top at Y = 39.36 mm) keep the
clearance they have with the stock cover.

Coordinates are the shared Sesame assembly frame: +Y up, front of robot is -X.
usage (from the repo root):
    pip install trimesh manifold3d numpy rtree
    python hardware/cad/tachikoma/build_tachikoma.py hardware/printing/stl out/
"""
import sys, os
import numpy as np
import trimesh
import manifold3d as mf
from manifold3d import Manifold, CrossSection, Mesh

STL, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
mf.set_circular_segments(96)

# ---------------------------------------------------------------- helpers
def load(path):
    m = trimesh.load(path)
    return Manifold(Mesh(vert_properties=m.vertices.astype(np.float32),
                         tri_verts=m.faces.astype(np.uint32)))

def to_trimesh(man):
    me = man.to_mesh()
    return trimesh.Trimesh(np.asarray(me.vert_properties)[:, :3], np.asarray(me.tri_verts))

def plane_frame(n, d):
    """Affine (3x4) mapping local (u=Z, v=up-the-face, w=along n) onto the plane n.p = d."""
    w = n / np.linalg.norm(n)
    u = np.array([0.0, 0.0, 1.0])
    v = np.cross(w, u)
    R = np.column_stack([u, v, w])
    return np.column_stack([R, w * d])

def inv(T):
    R, t = T[:, :3], T[:, 3]
    return np.column_stack([R.T, -R.T @ t])

def box(x0, x1, y0, y1, z0, z1):
    return Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])

def ellipsoid(c, r):
    return Manifold.sphere(1.0, 128).scale(r).translate(c)

def superellipsoid(c, r, p, tilt_deg=0.0):
    def warp(v):
        x, y, z = v
        k = (abs(x) ** p + abs(y) ** p + abs(z) ** p) ** (1.0 / p)
        return (x / k, y / k, z / k)
    s = Manifold.sphere(1.0, 128).warp(warp).scale(r)
    return s.rotate([0, 0, -tilt_deg]).translate(c)

def barrel(c, r, e_round, n_vert, tilt_deg=0.0):
    """Superquadric: boxy-round in plan (exponent e_round), flat-topped in elevation (n_vert)."""
    def warp(v):
        x, y, z = v
        k = ((abs(x) ** e_round + abs(z) ** e_round) ** (n_vert / e_round) + abs(y) ** n_vert) ** (1.0 / n_vert)
        return (x / k, y / k, z / k)
    s = Manifold.sphere(1.0, 160).warp(warp).scale(r)
    return s.rotate([0, 0, -tilt_deg]).translate(c)

def cyl_along(p0, p1, r, r1=None):
    """Cylinder/cone from p0 to p1."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0; L = np.linalg.norm(d); d /= L
    c = Manifold.cylinder(L, r, r if r1 is None else r1, 64)        # along +Z
    z = np.array([0, 0, 1.0]); ax = np.cross(z, d); s = np.linalg.norm(ax)
    if s > 1e-9:
        ang = np.degrees(np.arctan2(s, z @ d)); ax /= s
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        Rm = np.eye(3) + np.sin(np.radians(ang)) * K + (1 - np.cos(np.radians(ang))) * K @ K
        c = c.transform(np.column_stack([Rm, np.zeros(3)]))
    elif z @ d < 0:
        c = c.mirror([0, 0, 1])
    return c.translate(p0)

def surface_hit(man, origin, direction):
    """First point where a ray from `origin` leaves solid `man` (origin inside)."""
    tm = to_trimesh(man)
    d = np.asarray(direction, float); d /= np.linalg.norm(d)
    locs, _, _ = tm.ray.intersects_location([origin], [d])
    dist = (locs - origin) @ d
    return locs[np.argmin(np.where(dist > 0, dist, np.inf))], d

# ---------------------------------------------------------------- base cover
C = load(f"{STL}/top-covers/Top-Cover-No-Ears-v100.stl")
assert C.status() == mf.Error.NoError
ZC = 2.385                                       # cover centreline (Z)
X0, X1, Z0, Z1 = -13.59, 65.63, -16.31, 21.08    # cover footprint
Y_BASE, Y_ROOF = 21.97, 55.62
Y_CLEAR = 43.0                                   # nothing outside the footprint below this

# Envelope of the cover ("cover filled in"): union of hulls between horizontal slices.
TO_Z = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0]], float)   # assembly Y -> slice Z
TO_Y = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0]], float)
Cz = C.transform(TO_Z)

def envelope(shrink):
    ys = np.arange(Y_BASE + 0.02, Y_ROOF - 0.02 - shrink, 0.25)
    ys = np.unique(np.concatenate([ys, [37.98, 38.02, 43.98, 44.02, Y_ROOF - 0.02 - shrink]]))
    secs = []
    for y in ys:
        pts = np.vstack([np.asarray(p) for p in Cz.slice(y).to_polygons()])
        h = CrossSection.hull_points(pts.tolist())
        if shrink:
            h = h.offset(-shrink, mf.JoinType.Miter)
        secs.append((y, np.asarray(h.to_polygons()[0])))
    secs.insert(0, (Y_BASE - 5, secs[0][1]))        # keep the open bottom open
    pieces = [Manifold.hull_points(np.vstack([np.column_stack([a, np.full(len(a), y0)]),
                                              np.column_stack([b, np.full(len(b), y1)])]).tolist())
              for (y0, a), (y1, b) in zip(secs[:-1], secs[1:])]
    return Manifold.batch_boolean(pieces, mf.OpType.Add).transform(TO_Y)

E_IN = envelope(0.8)      # additions may only overlap the outer 0.8 mm of the cover walls

# Allowed region: inside the footprint above Y_CLEAR, flaring out at 45 deg above it.
ALLOWED = box(X0, X1, Y_CLEAR, 200, Z0, Z1) + Manifold.hull_points(
    [[x, y, z] for y, g in [(Y_CLEAR, 0.0), (Y_CLEAR + 80, 80.0)]
     for x in (X0 - g, X1 + g) for z in (Z0 - g, Z1 + g)])

# ---------------------------------------------------------------- openings to keep clear
N_FACE = np.array([-0.94, 0.34, 0.0]); N_FACE /= np.linalg.norm(N_FACE); D_FACE = 22.35  # OLED face
N_SW = np.array([1.0, 1.0, 0.0]) / np.sqrt(2); D_SW = 77.37                          # switch chamfer

def wall_holes(n, d):
    """Through-holes of the cover wall in plane (n, d), as local 2D polygons + frame."""
    T = plane_frame(n, d - 0.5)
    cs = C.transform(inv(T)).slice(0.0)
    holes = []
    for p in cs.to_polygons():
        p = np.asarray(p)
        area = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
        if area < 0:
            holes.append(p[::-1])
    return holes, T

def tunnel(n, d, grow, depth=40.0, flare=0.0):
    holes, T = wall_holes(n, d)
    assert len(holes) == 1, f"expected one hole, got {len(holes)}"
    cs = CrossSection([holes[0].tolist()]).offset(grow, mf.JoinType.Round)
    a = np.asarray(cs.to_polygons()[0])
    b = np.asarray(cs.offset(flare * depth, mf.JoinType.Round).to_polygons()[0]) if flare else a
    pts = np.vstack([np.column_stack([a, np.full(len(a), -1.5)]),
                     np.column_stack([b, np.full(len(b), depth)])])
    return Manifold.hull_points(pts.tolist()).transform(T), holes[0], T

WINDOW, win_poly, T_WIN = tunnel(N_FACE, D_FACE, grow=0.0)
SWITCH, sw_poly, T_SW = tunnel(N_SW, D_SW, grow=2.0, flare=0.08)

# ---------------------------------------------------------------- Tachikoma shapes
FIT = 0.15            # radial clearance for press-in parts (tight push fit + optional glue)
PIN_R = 1.5           # pins on the body; detail parts get PIN_R + FIT holes

def unit(v):
    v = np.asarray(v, float); return v / np.linalg.norm(v)

def pin(base, d, length, r=PIN_R):
    """Pin standing on a seat at `base`, pointing along d, with a small lead-in chamfer."""
    return cyl_along(base - d * 0.5, base + d * (length - 0.35), r) + \
        cyl_along(base + d * (length - 0.45), base + d * length, r - 0.03, r - 0.35)

# Head: low, wide flattened dome over the front, cut flat by the OLED face plate.
HEAD_C, HEAD_R = np.array([5.0, 44.0, ZC]), np.array([22.0, 17.5, 24.0])
head_full = ellipsoid(HEAD_C, HEAD_R)
face_half = box(-300, 300, -300, 300, -600, 0).transform(plane_frame(N_FACE, D_FACE + 0.6))
head = head_full ^ face_half

# Dark-grey "skirt" band where the dome meets the body (paint it grey).
SKIRT = (ellipsoid(HEAD_C, HEAD_R + 1.2) - ellipsoid(HEAD_C, HEAD_R - 2.0)) ^ \
    box(-100, 200, 45.5, 48.5, -100, 100) ^ face_half

# Abdomen pod: upright, boxy-round barrel with a flatter top, leaning back slightly.
POD_C, POD_R, POD_TILT = np.array([44.0, 59.0, ZC]), np.array([21.0, 19.0, 20.0]), 6.0
pod = barrel(POD_C, POD_R, 3.2, 3.0, POD_TILT)
pod_grow = lambda g: barrel(POD_C, POD_R + g, 3.2, 3.0, POD_TILT)

# Pod panel lines: cap seam near the top, a waist seam, and a vertical seam down the back.
def seam_ring(y0, y1):
    return (pod_grow(1.0) - pod_grow(-0.7)) ^ box(-100, 200, y0, y1, -100, 100)
cap_y = POD_C[1] + POD_R[1] - 6.0
SEAMS = seam_ring(cap_y, cap_y + 0.9) + seam_ring(POD_C[1] - 4.0, POD_C[1] - 3.1) + \
    ((pod_grow(1.0) - pod_grow(-0.7)) ^ box(POD_C[0], 200, 30, cap_y, ZC - 0.45, ZC + 0.45))
# Raised cap rim just above the cap seam.
CAP_RIM = (pod_grow(0.8) - pod_grow(-1.0)) ^ box(-100, 200, cap_y + 0.9, cap_y + 2.4, -100, 100)

# Periscope rod on the top front of the pod (chunky so it survives bumps).
p_top, _ = surface_hit(pod, POD_C + [-9.0, 0, 0], [0, 1, 0])
PERISCOPE = cyl_along(p_top - [0, 2, 0], p_top + [0, 6.0, 0], 2.0) + \
    cyl_along(p_top + [0, 3.8, 0], p_top + [0, 5.2, 0], 2.8) + \
    Manifold.sphere(2.2, 48).translate(p_top + [0, 6.0, 0])

# Round side hatches (raised ring + boss), conforming to the pod surface.
hatch_c = POD_C + np.array([2.0, -2.0, 0])
ring = (cyl_along(hatch_c + [0, 0, -40], hatch_c + [0, 0, 40], 8.0)
        - cyl_along(hatch_c + [0, 0, -41], hatch_c + [0, 0, 41], 6.4))
boss = cyl_along(hatch_c + [0, 0, -40], hatch_c + [0, 0, 40], 2.6)
HATCH = (ring + boss) ^ pod_grow(0.9)

# Red lights: two stacked stubby stalks per side at the top front corners of the pod.
STALKS, LIGHT_PINS, LIGHT_CAPS = [], [], []
for s in (-1, 1):
    for dy in (6.0, 1.0):
        d = unit([-0.55, 0.15, s * 0.82])
        hit, _ = surface_hit(pod, POD_C + [-4.0, dy + 6.0, 0], d)
        base = hit + d * 1.6                                     # stalk end face
        STALKS.append(cyl_along(hit - d * 2.0, base, 2.4))
        LIGHT_PINS.append(pin(base, d, 2.6, 1.0))
        LIGHT_CAPS.append((base, d))

# Eyes: three white "bowling ball" sensors on the head (separate press-in parts).
EYE_SPECS = [  # (ray direction from head centre, eye radius)
    (unit([-0.62, 0.78, 0.0]), 6.2),          # big centre eye on the top front
    (unit([-0.52, 0.30, -0.80]), 5.4),        # side eyes
    (unit([-0.52, 0.30, 0.80]), 5.4),
]
EYE_SEATS, EYE_PINS, EYE_NECKS, EYES = [], [], [], []
for d, r in EYE_SPECS:
    hit, _ = surface_hit(head_full, HEAD_C, d)
    c = hit - d * (0.25 * r)                                # sphere centre, slightly sunk
    t0 = -0.3 * r                                           # flat back of the eye cap
    # never let an eye reach into the stock cover walls: slide it out along its axis
    while (Manifold.sphere(r + FIT, 64).translate(c) ^ C).volume() > 0.01:
        c = c + d * 0.25
    seat = c + d * t0
    EYE_NECKS.append(cyl_along(seat - d * 8.0, seat + d * 0.01, 0.72 * r))      # solid neck back into the head
    EYE_SEATS.append(cyl_along(seat, seat + d * 2 * r, r + FIT))
    EYE_PINS.append(pin(seat, d, 3.2))
    EYES.append((c, d, r, t0))

# Front arms: sockets are pins on the lower front face, below the OLED window.
ARM_PINS, ARM_MOUNTS = [], []
for s in (-1, 1):
    y = 29.5
    x = (D_FACE - N_FACE[1] * y) / N_FACE[0]          # point on the face plane at height y
    base = np.array([x, y, ZC + s * 9.5]) + N_FACE * 0.6
    boss_ = cyl_along(base - N_FACE * 1.5, base + N_FACE * 1.2, 3.3)
    ARM_PINS.append(boss_ + pin(base + N_FACE * 1.2, N_FACE, 3.5))
    ARM_MOUNTS.append((base + N_FACE * 1.2, s))

# ---------------------------------------------------------------- assemble
def drop_slivers(m, min_vol=1.0):
    return Manifold.batch_boolean([c for c in m.decompose() if c.volume() > min_vol], mf.OpType.Add)

def finish(m, allowed=True):
    if allowed:
        m = m ^ ALLOWED
    return drop_slivers((m - E_IN) - WINDOW - SWITCH)

shell = finish(head + pod + SKIRT + CAP_RIM + HATCH + PERISCOPE +
               Manifold.batch_boolean(STALKS + EYE_NECKS, mf.OpType.Add))
shell = shell - SEAMS - Manifold.batch_boolean(EYE_SEATS, mf.OpType.Add)
# pins/bosses are added after the seats are cut; arm bosses sit on the stock front face
details = Manifold.batch_boolean(EYE_PINS + LIGHT_PINS, mf.OpType.Add)
arm_bits = Manifold.batch_boolean(ARM_PINS, mf.OpType.Add)
ADD = shell + finish(details) + drop_slivers((arm_bits - E_IN) - WINDOW)
FINAL = drop_slivers(C + ADD)
assert FINAL.status() == mf.Error.NoError

# ---------------------------------------------------------------- press-in parts
def eye_part(c, d, r, t0):
    keep = cyl_along(c + d * t0, c + d * (r + 1), r + 1)          # everything in front of the flat back
    cap = Manifold.sphere(r, 96).translate(c) ^ keep
    hole = cyl_along(c + d * (t0 - 1), c + d * (t0 + 3.5), PIN_R + FIT)
    pin_holes = []
    u = unit(np.cross(d, [0, 1, 0]) if abs(d[1]) < 0.9 else np.cross(d, [1, 0, 0])); v = np.cross(d, u)
    for k in range(3):
        a = np.radians(90 + 120 * k)
        q = c + d * r + (u * np.cos(a) + v * np.sin(a)) * 0.33 * r
        pin_holes.append(cyl_along(q + d * 1.5, q - d * 1.4, 0.75))
    return cap - hole - Manifold.batch_boolean(pin_holes, mf.OpType.Add)

def light_cap(base, d):
    body = cyl_along(base, base + d * 3.0, 2.4) + Manifold.sphere(2.4, 48).translate(base + d * 3.0)
    return body - cyl_along(base - d, base + d * (2.6 + 0.3), 1.0 + FIT)

def arm_part(mount, s):
    """Tachikoma manipulator: shoulder ball, upper arm, elbow, forearm, three-finger claw.
    Built flat in its own plane (normal = world Z) so it prints lying down."""
    sh = mount + N_FACE * 3.5                 # shoulder ball rests on the boss
    fwd = unit([-1.0, -0.35, 0.0])
    elbow = sh + fwd * 8.0 + np.array([0, -3.0, 0])
    wrist = elbow + unit([-1.0, 0.25, 0]) * 7.0
    parts = [Manifold.sphere(3.4, 48).translate(sh), cyl_along(sh, elbow, 2.3),
             Manifold.sphere(2.7, 48).translate(elbow), cyl_along(elbow, wrist, 2.0),
             cyl_along(wrist - unit([-1, 0.25, 0]) * 0.6, wrist + unit([-1, 0.25, 0]) * 0.9, 2.8)]
    tip_base = wrist + unit([-1, 0.25, 0]) * 0.9
    for ang in (-50, 0, 50):
        a = np.radians(ang)
        d1 = unit([-np.cos(a), np.sin(a) + 0.25, 0])
        k = tip_base + d1 * 3.2
        d2 = unit(d1 + np.array([0.6 * np.sign(-ang) if ang else 0, 0.0 if ang else -0.6, 0]))
        parts += [cyl_along(tip_base, k, 1.1), Manifold.sphere(1.1, 24).translate(k),
                  cyl_along(k, k + d2 * 2.4, 1.1, 0.6)]
    arm = Manifold.batch_boolean(parts, mf.OpType.Add)
    arm = arm - cyl_along(mount - N_FACE * 1.0, mount + N_FACE * (3.5 + 0.3), PIN_R + FIT)
    # flat face for printing: shave 0.7 mm off the side facing the robot's centreline
    zc_side = ZC + s * 9.5
    flat = box(-200, 200, -200, 200, zc_side - s * 1.6 - (100 if s > 0 else 0), zc_side - s * 1.6 + (0 if s > 0 else 100))
    return arm - flat

EYE_PARTS = [eye_part(*e) for e in EYES]
LIGHT_PARTS = [light_cap(b, d) for b, d in LIGHT_CAPS]
ARM_PARTS = [arm_part(m, s) for m, s in ARM_MOUNTS]

# ---------------------------------------------------------------- checks
added = FINAL - C
interior = E_IN - C
print("added volume       %.1f cm^3" % (added.volume() / 1000))
leak = added ^ interior
print("added ∩ interior    %.4f mm^3  (must be ~0)" % leak.volume())
outside = added - box(X0, X1, -100, 300, Z0, Z1)
print("outside-footprint added material: min Y = %.2f, front arm bosses aside" % (outside - box(-40, X0 + 0.01, 0, 40, -100, 100)).bounding_box()[1])
print("window clear ∩ added  %.4f" % (added ^ (WINDOW - E_IN)).volume())
print("switch clear ∩ added  %.4f" % (added ^ (SWITCH - E_IN)).volume())
arms_world = Manifold.batch_boolean(ARM_PARTS, mf.OpType.Add)
print("arms ∩ cover %.3f mm^3, arms ∩ window tunnel %.3f" % ((arms_world ^ FINAL).volume(), (arms_world ^ WINDOW).volume()))
for i, (e, part) in enumerate(zip(EYES, EYE_PARTS)):
    print(f"eye {i}: overlap with cover {(part ^ FINAL).volume():.3f} mm^3, with window {(part ^ WINDOW).volume():.3f}")
print("final bbox", np.round(FINAL.bounding_box(), 2), "components", [(round(c.volume(), 1), np.round(c.bounding_box(), 1).tolist()) for c in FINAL.decompose()] if len(FINAL.decompose()) > 1 else 1)

def save(man, name):
    tm = to_trimesh(man)
    tm.merge_vertices()
    tm = trimesh.util.concatenate([p for p in tm.split(only_watertight=False) if abs(p.volume) > 1.0])
    tm.export(os.path.join(OUT, name))
    print(f"  wrote {name}: watertight={tm.is_watertight} faces={len(tm.faces)}")
    return tm

def lay_flat(man, down):
    """Rotate so direction `down` points to -Z, then drop onto z=0 for printing."""
    tm = to_trimesh(man); tm.merge_vertices()
    R = trimesh.geometry.align_vectors(unit(down), [0, 0, -1])
    tm.apply_transform(R)
    tm.apply_translation(-tm.bounds[0] * [0, 0, 1] - [tm.centroid[0], tm.centroid[1], 0])
    return tm

save(FINAL, "Top-Cover-Tachikoma-v200.stl")
save(Manifold.batch_boolean(ARM_PARTS, mf.OpType.Add), "Arms-Tachikoma-assembled.stl")   # for previews/checks
save(Manifold.batch_boolean(EYE_PARTS, mf.OpType.Add), "Eyes-Tachikoma-assembled.stl")
save(Manifold.batch_boolean(LIGHT_PARTS, mf.OpType.Add), "Lights-Tachikoma-assembled.stl")
for name, parts, downs in [
        ("Eyes-Tachikoma-v200-print-3-white.stl", EYE_PARTS, [-e[1] for e in EYES]),
        ("Lights-Tachikoma-v200-print-4-red.stl", LIGHT_PARTS, [-d for _, d in LIGHT_CAPS]),
        ("Arms-Tachikoma-v200-print-2.stl", ARM_PARTS, [np.array([0, 0, -s]) for _, s in ARM_MOUNTS])]:
    tms = []
    for i, (p, dn) in enumerate(zip(parts, downs)):
        tm = lay_flat(p, dn); tm.apply_translation([i * 26.0, 0, 0]); tms.append(tm)
    out = trimesh.util.concatenate(tms); out.export(os.path.join(OUT, name))
    print(f"  wrote {name}: watertight={out.is_watertight} parts={len(tms)} z-min={out.bounds[0][2]:.2f}")

# rough print-weight estimate: 2 perimeters (0.84 mm) + 10 % infill, PLA 1.24 g/cm^3
def grams(man):
    tm = to_trimesh(man); tm.merge_vertices()
    shell_ = tm.area * 0.84
    return (shell_ + 0.10 * max(tm.volume - shell_, 0)) * 1.24 / 1000
print("est. print mass: stock cover %.1f g -> tachikoma v200 %.1f g (+ details)" % (grams(C), grams(FINAL)))
