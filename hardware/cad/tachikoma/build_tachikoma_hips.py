"""Generate Tachikoma-style hips (upper legs) for Sesame from the stock L1/L2/R1/R2 parts.

Each hip becomes two prints plus a cap:
  * <name>-Tachikoma-v100.stl  the stock hip with a "shoulder" hump on the top plate. The
    hump is cut flush with the hip's outer face, so it still prints flat on that face
    like the stock part (no supports);
  * Thigh-Shell-Tachikoma-v100-{A,B}.stl  the rest of the thick arched thigh and the round
    knee housing, glued (CA) onto the hip's flat outer face; it prints on that flat face;
  * Knee-Cap-Tachikoma-v100.stl  white press-on cap over the two knee (foot-servo horn)
    screws; it pops off for servicing.

Kept exactly as stock: the frame-servo horn interface and both top-plate screw holes
(vertical access tunnels stay open through the hump), the foot-servo horn plate, and
every surface the foot servo or foot can sweep past. Verify with
check_tachikoma_clearance.py.

All parts share one design: R2 is the same shape as L1, and L2/R1 are its mirror image
(mirrored about Z = 0 in the exported STL coordinates). Shell A goes on L1/R2, B on L2/R1.
usage (from the repo root):
    pip install trimesh manifold3d numpy
    python hardware/cad/tachikoma/build_tachikoma_hips.py hardware/printing/stl out/
"""
import sys, os
import numpy as np
import trimesh
import manifold3d as mf
from manifold3d import Manifold, Mesh

STL, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
mf.set_circular_segments(64)

# --- interfaces measured from L1-v117 (exported pose, foot servo axis along X)
TOP_SCREWS = [(1.49, -23.02), (1.54, -34.43)]     # (X, Z) of the top-plate screw holes
TOP_PLATE_Y = 39.36
KNEE = np.array([15.24, -59.82])                  # (Y, Z) foot-servo shaft
KNEE_SCREW2 = np.array([15.24, -48.44])           # second horn screw
PLATE_FACE_X = -12.45                             # outer face of the knee plate

CAP_C = (KNEE + KNEE_SCREW2) / 2                  # knee cap centre between both screws
CAP_RECESS_R = 9.0                                # clears both screw heads
CAP_FIT = 0.15                                    # radial press-fit clearance
HOUSING_R = 11.5
HOUSING_X = -17.5                                 # outer face of the knee housing
TUNNEL_R = 3.4                                    # screwdriver access through the thigh

# thigh arch: chain of ellipsoids (centre X, Y, Z, radii X, Y, Z), body end -> knee
ARCH = [
    ((-3.0, 41.6, -36.5), (7.0, 3.2, 5.0)),
    ((-7.5, 44.5, -42.0), (6.0, 5.0, 5.5)),
    ((-13.0, 41.5, -47.5), (4.2, 6.0, 5.5)),
    ((-14.8, 32.0, -53.5), (3.4, 6.5, 6.0)),
    ((-14.8, 22.0, -55.5), (3.4, 6.5, 6.5)),
]

def load(path):
    m = trimesh.load(path)
    return Manifold(Mesh(vert_properties=m.vertices.astype(np.float32),
                         tri_verts=m.faces.astype(np.uint32)))

def to_trimesh(man):
    me = man.to_mesh()
    tm = trimesh.Trimesh(np.asarray(me.vert_properties)[:, :3], np.asarray(me.tri_verts))
    tm.merge_vertices()
    return trimesh.util.concatenate([p for p in tm.split(only_watertight=False) if abs(p.volume) > 1.0])

def ellipsoid(c, r):
    return Manifold.sphere(1.0, 64).scale(r).translate(c)

def x_cyl(y, z, r, x0, x1):
    return Manifold.cylinder(x1 - x0, r, r, 96).rotate([0, 90, 0]).translate([x0, y, z])

def rounded_x_disc(y, z, r, x0, x1, fillet):
    """Disc along X from x0 (inner) to x1 (outer, x1 < x0) with a rounded outer rim."""
    core = x_cyl(y, z, r - fillet, x1, x0)
    rim = x_cyl(y, z, r, x1 + fillet, x0)
    torus = Manifold.revolve(
        mf.CrossSection.circle(fillet, 32).translate([r - fillet, 0]), 96
    ).rotate([0, 90, 0]).translate([x1 + fillet, y, z])
    return core + rim + torus

def keep_outs():
    """Space that must stay empty: screw access, knee-cap recess, foot/servo sweep, hip interior."""
    k = [x_cyl(CAP_C[0], CAP_C[1], CAP_RECESS_R, HOUSING_X - 1, PLATE_FACE_X + 0.01)]
    for x, z in TOP_SCREWS:
        k.append(Manifold.cylinder(40, TUNNEL_R, TUNNEL_R, 48).rotate([-90, 0, 0])
                 .translate([x, TOP_PLATE_Y - 0.01, z]))
    # foot + servo sweep: everything at X > -9 below/outside Z -49.5, and r < 13 of the knee
    k.append(Manifold.cube([40, 120, 60]).translate([-9.0, -30, -49.5 - 60]))
    k.append(x_cyl(KNEE[0], KNEE[1], 13.0, -9.0, 30))
    # never grow under the top plate on the servo side (frame servo / hip interior)
    k.append(Manifold.cube([40, TOP_PLATE_Y - 0.01 + 30, 60]).translate([PLATE_FACE_X + 0.01, -30, -48.0]))
    return Manifold.batch_boolean(k, mf.OpType.Add)

def half(x_min=None, x_max=None):
    lo = -200 if x_min is None else x_min
    hi = 200 if x_max is None else x_max
    return Manifold.cube([hi - lo, 400, 400]).translate([lo, -200, -200])

def thigh_parts():
    """Return (hump fused to the hip, glue-on thigh shell), both in the L1 frame."""
    blobs = [ellipsoid(c, r) for c, r in ARCH]
    arch = Manifold.batch_boolean(
        [Manifold.batch_hull([a, b]) for a, b in zip(blobs[:-1], blobs[1:])], mf.OpType.Add)
    # hump: extend each blob straight down to the outer-face plane so it prints without support
    flat = lambda c, r: ellipsoid((PLATE_FACE_X, c[1], c[2]), (0.01, r[1], r[2]))
    hump = Manifold.batch_boolean([Manifold.batch_hull([b, flat(c, r)]) for b, (c, r) in zip(blobs, ARCH)]
                                  + [arch], mf.OpType.Add) ^ half(x_min=PLATE_FACE_X)
    housing = rounded_x_disc(CAP_C[0], CAP_C[1], HOUSING_R, PLATE_FACE_X + 0.5, HOUSING_X, 2.0)
    shell = (arch + housing) ^ half(x_max=PLATE_FACE_X)
    ko = keep_outs()
    return hump - ko, shell - ko

def knee_cap():
    """White press-on cap, modelled in the L1 frame (plug sits in the recess)."""
    plug_r = CAP_RECESS_R - CAP_FIT
    plug = x_cyl(CAP_C[0], CAP_C[1], plug_r, HOUSING_X - 0.01, HOUSING_X + 2.2)
    dome = ellipsoid((HOUSING_X, CAP_C[0], CAP_C[1]), (4.0, HOUSING_R - 1.0, HOUSING_R - 1.0)) \
        ^ Manifold.cube([10, 60, 60]).translate([HOUSING_X - 10, CAP_C[0] - 30, CAP_C[1] - 30])
    # three pinholes like the Tachikoma's sensor balls
    holes = []
    for k in range(3):
        a = np.radians(90 + 120 * k)
        y, z = CAP_C[0] + 3.2 * np.sin(a), CAP_C[1] + 3.2 * np.cos(a)
        holes.append(x_cyl(y, z, 0.7, HOUSING_X - 6, HOUSING_X - 2.4))
    return plug + dome - Manifold.batch_boolean(holes, mf.OpType.Add)

def cap_print_pose(cap):
    """Plug face down on the bed, dome up: +X (plug direction) -> -Z."""
    m = cap.translate([0, -CAP_C[0], -CAP_C[1]]).rotate([0, 90, 0])
    return m.translate([0, 0, -m.bounding_box()[2]])

def lay_on_face(m):
    """Shell print pose: glue face (x = PLATE_FACE_X) on the bed, thigh pointing up."""
    m = m.translate([-PLATE_FACE_X, 0, 0]).rotate([0, 90, 0])     # +X -> -Z ... glue face down
    bb = m.bounding_box()
    if bb[5] - bb[2] > 0 and abs(bb[2]) > 1e-3 and abs(bb[5]) < 1e-3:
        m = m.mirror([0, 0, 1])
    return m.translate([0, 0, -m.bounding_box()[2]])

if __name__ == "__main__":
    HUMP, SHELL = thigh_parts()
    CAP = knee_cap()
    for name in ("L1", "R2", "L2", "R1"):
        part = load(f"{STL}/{name}-v117.stl")
        a_side = name in ("L1", "R2")
        hump = HUMP if a_side else HUMP.mirror([0, 0, 1])
        shell = SHELL if a_side else SHELL.mirror([0, 0, 1])
        cap = CAP if a_side else CAP.mirror([0, 0, 1])
        final = part + hump
        assert final.status() == mf.Error.NoError
        tm = to_trimesh(final)
        tm.export(f"{OUT}/{name}-Tachikoma-v100.stl")
        # assembled reference (hip + glued shell + cap) for clearance checks and previews
        to_trimesh(final + shell + cap).export(f"{OUT}/{name}-Tachikoma-assembled.stl")
        bed = tm.area_faces[(tm.face_normals[:, 0] < -0.999) & (tm.triangles_center[:, 0] < PLATE_FACE_X + 0.01)].sum()
        print(f"{name}: added {(final - part).volume() / 1000:.2f} cm^3, removed {(part - final).volume():.3f} mm^3, "
              f"lowest X {final.bounding_box()[0]:.2f} (stock {part.bounding_box()[0]:.2f}), "
              f"flat print face {bed:.0f} mm^2, watertight={tm.is_watertight}, parts={len(tm.split(only_watertight=False))}")
    for tag, sh in (("A", SHELL), ("B", SHELL.mirror([0, 0, 1]))):
        t = to_trimesh(lay_on_face(sh))
        bed = t.area_faces[(t.face_normals[:, 2] < -0.999) & (t.triangles_center[:, 2] < 0.01)].sum()
        t.export(f"{OUT}/Thigh-Shell-Tachikoma-v100-{tag}.stl")
        print(f"thigh shell {tag}: watertight={t.is_watertight} parts={len(t.split(only_watertight=False))} "
              f"glue face {bed:.0f} mm^2 size {np.round(t.extents, 1)}")
    tmc = to_trimesh(cap_print_pose(CAP))
    flat = tmc.area_faces[(tmc.face_normals[:, 2] < -0.99) & (tmc.triangles_center[:, 2] < 0.01)].sum()
    tmc.export(f"{OUT}/Knee-Cap-Tachikoma-v100.stl")
    print(f"knee cap: watertight={tmc.is_watertight} size={np.round(tmc.extents, 2)} "
          f"bed contact {flat:.0f} mm^2 (print 4, white)")
