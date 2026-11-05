#!/usr/bin/env python3
"""Generate independent illustrative sections; no supplier input is read.

Blade EI targets: Moore et al. (2024), table 1,
doi:10.1088/1742-6596/2767/7/072025 (OWENS column).
All other dimensions/material properties below are explicit example assumptions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HEADER = ("LENFRACT_[-] MASSD_[kg/m] EIx_[N.m^2] EIy_[N.m^2] EA_[N] "
          "GJ_[N.m^2] GA_[N] STRPIT_[deg] KSX_[-] KSY_[-] RGX_[-] RGY_[-] "
          "XCM_[-] YCM_[-] XCE_[-] YCE_[-] XCS_[-] YCS_[-]")


def blade_section(samples=4097):
    chord, young, shear, density = 0.5, 30e9, 4e9, 2000.0
    x = chord * (1 - np.cos(np.linspace(0, math.pi, samples))) / 2

    def thickness(xpos):
        u = np.asarray(xpos) / chord
        return 5 * 0.18 * chord * (0.2969*np.sqrt(u) - 0.126*u
            - 0.3516*u**2 + 0.2843*u**3 - 0.1036*u**4)

    y = thickness(x)
    ds = np.hypot(np.diff(x), np.diff(y))
    perimeter = 2 * ds.sum()
    centroid = 2 * np.sum(ds * (x[:-1] + x[1:]) / 2) / perimeter
    ix_skin = 2 * np.sum(ds * (y[:-1]**2 + y[:-1]*y[1:] + y[1:]**2) / 3)
    dx = x - centroid
    iy_skin = 2 * np.sum(ds * (dx[:-1]**2 + dx[:-1]*dx[1:] + dx[1:]**2) / 3)
    enclosed = np.sum(np.diff(x) * (y[:-1] + y[1:]))
    # Four equal ideal longitudinal caps: fore/aft, upper/lower.
    # Their finite local second moment is neglected in this equivalent beam.
    cap_offset = 0.35 * chord
    cap_y = thickness(np.array([centroid-cap_offset, centroid+cap_offset]))
    cap_ix = float(np.mean(cap_y**2))
    target = np.array([1.21e5, 2.87e6]) / young
    skin, caps = np.linalg.solve([[ix_skin, cap_ix], [iy_skin, cap_offset**2]], target)
    assert skin > 0 and caps > 0
    area = float(perimeter * skin + caps)
    ix, iy = map(float, target)
    # Closed thin-wall Bredt torsion, skin only. Shear center approximated by
    # centroid, with no composite coupling or local insert/reinforcement model.
    gj = float(4 * enclosed**2 * shear * skin / perimeter)
    offset = float(centroid / chord - 0.25)  # aerodynamic pitch axis is c/4
    row = [density*area, young*ix, young*iy, young*area, gj, shear*area,
           0, 0.85, 0.85, math.sqrt(ix/area)/chord, math.sqrt(iy/area)/chord,
           offset, 0, offset, 0, offset, 0, 0.0011]
    return row, dict(chord_m=chord, E_pa=young, G_pa=shear, density_kg_m3=density,
        skin_thickness_m=float(skin), total_cap_area_m2=float(caps),
        cap_offset_from_centroid_m=cap_offset, centroid_from_le_m=float(centroid),
        enclosed_area_m2=float(enclosed), material_area_m2=area,
        EI_flap_nm2=young*ix, EI_lag_nm2=young*iy, mass_kg_m=density*area,
        GJ_nm2=gj, quadrature_points=samples)


def tube_section(outer, inner):
    assert 0 <= inner < outer
    young, poisson, density = 200e9, 0.3, 7850.0
    shear = young / (2 * (1+poisson))
    area = math.pi/4 * (outer**2-inner**2)
    inertia = math.pi/64 * (outer**4-inner**4)
    radius = math.sqrt(inertia/area) / outer
    return [density*area, young*inertia, young*inertia, young*area,
            shear*2*inertia, shear*area, 0, 0.85, 0.85, radius, radius,
            0, 0, 0, 0, 0, 0, outer]


def render():
    blade, details = blade_section()
    blade_text = ("// Public generic equivalent glass-fiber blade; see section_provenance.json\n"
        "0.0011 RAYLEIGHDMP\n1 STIFFTUNER\n1 MASSTUNER\n2 DISCTYPE\n1 DISC\n\n"
        + HEADER + " RAYLEIGHDMP_Vec_[-]\n")
    for station in (0, 0.05, 0.25, 0.5, 0.75, 0.95, 1):
        blade_text += " ".join(f"{v:.12g}" for v in [station, *blade]) + "\n"
    # Nominal solid lower shaft transitioning into a hollow steel torque tube.
    # These rounded assumed dimensions are not manufacturing specifications.
    profile = [(0, .10, 0), (.20, .14, 0), (.26, .15, 0),
               (.28, .65, .59), (.45, .65, .59), (.47, .65, .61),
               (.70, .65, .61), (1, .65, .61)]
    tube_text = ("// Public generic steel shaft/tube; see section_provenance.json\n"
        "0.03 RAYLEIGHDMP\n12 STIFFTUNER\n1 MASSTUNER\n10 DISC\n"
        "ADDMASS_0.0 1000 // nominal lumped equipment mass, kg\n\n"
        + HEADER + " Diameter_[m]\n")
    for station, outer, inner in profile:
        tube_text += " ".join(f"{v:.12g}" for v in [station, *tube_section(outer, inner)]) + "\n"
    provenance = dict(schema="vorlap-public-sections-v1",
        source="https://doi.org/10.1088/1742-6596/2767/7/072025",
        published_targets="Table 1 OWENS blade EI values; 0.5 m NACA0018 chord",
        assumptions="Uniform equivalent orthotropic shell and four ideal longitudinal caps; no local inserts. Closed-tail NACA four-digit analytic outline. Bredt skin-only torsion; centroid approximates shear center. Tube: nominal steel annuli, E=200 GPa, nu=0.3, rho=7850 kg/m^3; 1000 kg equipment lump. Section damping/tuners are retained example integration choices, not inferred material data.",
        blade=details, tube_profile_fraction_outer_inner_m=profile,
        applicability="Illustrative public replacement. Neither exact paper inputs nor a qualified manufacturer structural design.",
        files={})
    files = {"10m_HVAWT_reference_blade.str": blade_text,
             "10m_HVAWT_reference_torque_tube.str": tube_text}
    provenance["files"] = {k:hashlib.sha256(v.encode()).hexdigest() for k,v in files.items()}
    # Document the same 12-significant-digit precision used in the section
    # tables. Last-bit BLAS/NumPy reduction differences must not make an
    # otherwise identical generated input fail its provenance check.
    def documented_precision(value):
        if isinstance(value, float):
            return float(f"{value:.12g}")
        if isinstance(value, dict):
            return {k: documented_precision(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [documented_precision(v) for v in value]
        return value
    return files, documented_precision(provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify committed generated files")
    args = parser.parse_args()
    files, provenance = render()
    for name, content in files.items():
        path = HERE / "10m_HVAWT_reference_turbine/Structure" / name
        if args.check:
            if path.read_text() != content:
                raise SystemExit(f"Generated section differs: {path}")
        else:
            path.write_text(content)
    path = HERE / "section_provenance.json"
    content = json.dumps(provenance, indent=2) + "\n"
    if args.check:
        assert path.read_text() == content
    else:
        path.write_text(content)


if __name__ == "__main__":
    main()
