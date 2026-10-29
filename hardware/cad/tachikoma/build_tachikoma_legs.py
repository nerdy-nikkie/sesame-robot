"""Generate Tachikoma-style feet (lower legs) for Sesame from the stock L3/L4/R3/R4 parts.

Each foot keeps its servo cradle, screw holes and side profile exactly as stock.
Only the long lower-leg arm (beyond the servo tabs) is changed:
  * the lightening windows are filled and both faces get a recessed armour panel
    (1 mm deep, 45 deg walls), like the panel lines on the Tachikoma's tapered legs.
Both side faces stay flat, so the foot prints lying on either side face with no
supports; the recess walls are 45 deg and the recess floor is a short bridge.
Nothing extends past the stock silhouette in the plane the foot swings in, so the
ground contact point and effective leg length are unchanged.

Coordinates are the shared Sesame assembly frame (foot servo axis = X).
usage (from the repo root):
    pip install trimesh manifold3d numpy
    python hardware/cad/tachikoma/build_tachikoma_legs.py hardware/printing/stl out/
"""
import sys, os
import numpy as np
import trimesh
import manifold3d as mf
from manifold3d import Manifold, CrossSection, Mesh

STL, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
mf.set_circular_segments(96)

S_FILL = 82.5      # |Z| where the arm starts (servo tabs end at ~81.9)
S_PLATE = 86.0     # armour panels start here (clear of the tab screw heads)
PANEL_INSET = 2.5  # panel edge inset from the leg outline
PANEL_DEPTH = 1.0  # recess depth on each face (walls at 45 deg)

def load(path):
    m = trimesh.load(path)
    return Manifold(Mesh(vert_properties=m.vertices.astype(np.float32),
                         tri_verts=m.faces.astype(np.uint32)))

def to_trimesh(man):
    me = man.to_mesh()
    tm = trimesh.Trimesh(np.asarray(me.vert_properties)[:, :3], np.asarray(me.tri_verts))
    tm.merge_vertices()
    return tm

# Profile plane: local 2D (u, v) = (Z, Y); extrusion along X.
TO_LOCAL = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], float)   # (x,y,z)->(z,y,x)
TO_WORLD = TO_LOCAL                                                       # it is its own inverse

def extrude_x(cs, x0, x1):
    return cs.extrude(x1 - x0).translate([0, 0, x0]).transform(TO_WORLD)

def profile(part):
    """Filled side profile (YZ silhouette, windows filled) as a CrossSection in (Z, Y)."""
    shadow = part.transform(TO_LOCAL).project()
    outers = []
    for p in shadow.to_polygons():
        p = np.asarray(p)
        area = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
        if area > 0:
            outers.append(CrossSection([p.tolist()]))
    return CrossSection.batch_boolean(outers, mf.OpType.Add)

def band(s0, s1, sgn):
    """Region s0 <= |Z| <= s1 on the foot's side, as a 2D rectangle in (Z, Y)."""
    z0, z1 = sorted([sgn * s0, sgn * s1])
    return CrossSection.square([z1 - z0, 200]).translate([z0, -100])

def panel_recess(prof2d, x0, x1):
    """45 deg chamfered recess, PANEL_DEPTH deep, on both side faces of the leg."""
    a = np.asarray(prof2d.offset(-PANEL_INSET, mf.JoinType.Round).to_polygons()[0])
    b = np.asarray(prof2d.offset(-PANEL_INSET - PANEL_DEPTH, mf.JoinType.Round).to_polygons()[0])
    lift = lambda p, x: np.column_stack([p, np.full(len(p), x)])
    cut = lambda xf, d: Manifold.hull_points(
        np.vstack([lift(a, xf - d * 0.5), lift(a, xf), lift(b, xf + d * PANEL_DEPTH)]).tolist())
    return (cut(x0, 1.0) + cut(x1, -1.0)).transform(TO_WORLD)

def build(name):
    part = load(f"{STL}/{name}-v117.stl")
    bb = part.bounding_box()
    x0, x1 = bb[0], bb[3]
    sgn = -1.0 if abs(bb[2]) > abs(bb[5]) else 1.0          # which end is the tip
    tip = max(abs(bb[2]), abs(bb[5]))
    prof = profile(part)

    arm = prof ^ band(S_FILL, tip + 1, sgn)
    fill = extrude_x(arm, x0, x1)

    panel_prof = prof ^ band(S_PLATE, tip - 1.5, sgn)
    final = (part + fill) - panel_recess(panel_prof, x0, x1)
    assert final.status() == mf.Error.NoError

    # checks
    p_stock, p_new = profile(part), profile(final)
    grow = (p_new - p_stock).area()
    print(f"{name}: tip |Z|={tip:.2f}; side-profile growth {grow:.4f} mm^2 (must be ~0); X span "
          f"{final.bounding_box()[0]:.2f}..{final.bounding_box()[3]:.2f}; "
          f"added {(final - part).volume() / 1000:.2f} cm^3, removed {(part - final).volume():.1f} mm^3")
    return final

def save(man, path):
    tm = to_trimesh(man)
    tm = trimesh.util.concatenate([p for p in tm.split(only_watertight=False) if abs(p.volume) > 1.0])
    tm.export(path)
    print(f"  wrote {os.path.basename(path)}: watertight={tm.is_watertight}")

for name in ("L3", "L4", "R3", "R4"):
    save(build(name), f"{OUT}/{name}-Tachikoma-v100.stl")
