"""Range-of-motion clearance check for the Tachikoma parts.

Rule: the Tachikoma parts must not collide anywhere the stock parts don't. For each
pose we compare stock vs Tachikoma interference volumes:
  * hip swing (frame servo) from -100 to +100 deg at a front and a rear corner,
    against the top cover, frame and bottom cover;
  * foot swing (foot servo) from straight down to straight up, against the hip;
  * front and rear legs on the same side swinging toward each other.
The foot servo is approximated as an MG90 envelope (body, tabs and gear boss).

usage (from the repo root, after building the parts into out/):
    python hardware/cad/tachikoma/check_tachikoma_clearance.py hardware/printing/stl out/
"""
import sys
import numpy as np
import trimesh
from manifold3d import Manifold, Mesh

STL, OUT = sys.argv[1], sys.argv[2]
XMID = 26.02
PIVOT = np.array([1.49, 0.0, -23.02])         # front -Z frame-servo shaft (hip pivot)
KNEE = np.array([0.0, 15.24, -59.82])         # foot-servo shaft in the exported pose
TOL = 0.5                                     # mm^3 of numerical noise allowed

def load(path):
    m = trimesh.load(path)
    return Manifold(Mesh(vert_properties=m.vertices.astype(np.float32), tri_verts=m.faces.astype(np.uint32)))

def rot(axis, deg, about):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    R = {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
         "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])}[axis]
    return np.column_stack([R, about - R @ about])

def mirror_x(m):
    return m.mirror([1, 0, 0]).translate([2 * XMID, 0, 0])

def servo():
    body = Manifold.cube([22.9, 12.6, 22.5]).translate([-5.6, 9.0, -77.0])
    tabs = Manifold.cube([2.6, 12.6, 32.0]).translate([1.2, 9.0, -82.0])
    boss = Manifold.cylinder(2.4, 6.0, 6.0, 48).rotate([0, 90, 0]).translate([-8.0, KNEE[1], KNEE[2]])
    return body + tabs + boss

def pose(hip, foot, theta, alpha, rear):
    f = foot.transform(rot("x", alpha, KNEE))
    T = rot("y", theta, PIVOT)
    h, f = hip.transform(T), f.transform(T)
    return (mirror_x(h), mirror_x(f)) if rear else (h, f)

def main():
    stock_hip, stock_foot = load(f"{STL}/L1-v117.stl"), load(f"{STL}/L3-v117.stl") + servo()
    new_hip, new_foot = load(f"{OUT}/L1-Tachikoma-assembled.stl"), load(f"{OUT}/L3-Tachikoma-v100.stl") + servo()
    stock_body = load(f"{STL}/top-covers/Top-Cover-No-Ears-v100.stl")
    new_body = load(f"{OUT}/Top-Cover-Tachikoma-v200.stl")
    for extra in ("Internal-Frame-v121.stl", "Bottom-Cover-v121.stl"):
        stock_body = stock_body + load(f"{STL}/{extra}")
        new_body = new_body + load(f"{STL}/{extra}")
    try:
        new_body = new_body + load(f"{OUT}/Arms-Tachikoma-assembled.stl")
    except Exception:
        pass
    failures = 0

    print("hip swing vs body (stock / tachikoma mm^3):")
    for rear in (False, True):
        row = []
        for th in range(-100, 101, 5):
            for alpha in (-90, 0, 90):
                hs, fs = pose(stock_hip, stock_foot, th, alpha, rear)
                hn, fn = pose(new_hip, new_foot, th, alpha, rear)
                s = (hs ^ stock_body).volume() + (fs ^ stock_body).volume()
                n = (hn ^ new_body).volume() + (fn ^ new_body).volume()
                if s < TOL and n >= TOL:
                    failures += 1
                    row.append(f"FAIL th={th} a={alpha}: {n:.1f}")
            if th % 20 == 0:
                hs, _ = pose(stock_hip, stock_foot, th, 0, rear)
                row.append(f"{th}:{'free' if (hs ^ stock_body).volume() < TOL else 'blocked'}")
        print("  rear " if rear else "  front", " ".join(row))

    print("foot swing vs hip (Tachikoma additions only):")
    add = new_hip - stock_hip
    worst = 0.0
    for alpha in range(-90, 91, 5):
        f = new_foot.transform(rot("x", alpha, KNEE))
        v = (add ^ f).volume()
        worst = max(worst, v)
        if v >= TOL:
            failures += 1
            print(f"  FAIL alpha={alpha}: {v:.1f}")
    print(f"  worst overlap {worst:.3f} mm^3")

    print("front vs rear leg on the same side (both swinging toward each other):")
    for th in range(0, -101, -10):
        for alpha in (-90, 0, 90):
            hs_f, fs_f = pose(stock_hip, stock_foot, th, alpha, False)
            hs_r, fs_r = pose(stock_hip, stock_foot, th, alpha, True)
            hn_f, fn_f = pose(new_hip, new_foot, th, alpha, False)
            hn_r, fn_r = pose(new_hip, new_foot, th, alpha, True)
            s = ((hs_f + fs_f) ^ (hs_r + fs_r)).volume()
            n = ((hn_f + fn_f) ^ (hn_r + fn_r)).volume()
            if s < TOL and n >= TOL:
                failures += 1
                print(f"  FAIL th={th} a={alpha}: {n:.1f}")
    print("RESULT:", "PASS" if failures == 0 else f"{failures} new collisions")
    return failures

if __name__ == "__main__":
    sys.exit(1 if main() else 0)
