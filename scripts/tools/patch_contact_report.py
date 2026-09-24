# Copyright (c) 2026, r1_lab contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Add ``PhysxContactReportAPI`` to every rigid body of a URDF-converter asset.

Why: the Isaac Sim 6.0 URDF converter writes rigid-body links as a *nested* kinematic tree
(``Robot/Geometry/pelvis_link/left_hip_pitch_link/...``).  Isaac Lab 3.0's
``activate_contact_sensors`` stops descending at the first rigid body it meets, so only the
pelvis receives the contact-report API and contact sensors on the feet or torso fail to
resolve.  Baking the API into the asset's PhysX layer sidesteps that traversal.

Usage (any Python with ``pxr`` available, e.g. the Isaac Lab venv)::

    python patch_contact_report.py <asset_dir>/R1.usda

The script edits ``payloads/Physics/physx.usda`` next to the given interface file, authoring
``over`` prims that add the API schema and a zero force threshold.  It is idempotent.
"""

import argparse
import os
import sys

from pxr import Sdf, Usd, UsdPhysics

API = "PhysxContactReportAPI"
THRESHOLD_ATTR = "physxContactReport:threshold"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("interface_usd", help="Path to the asset's interface USD (e.g. R1.usda).")
    parser.add_argument("--threshold", type=float, default=0.0, help="Contact report force threshold (N).")
    args = parser.parse_args()

    interface = os.path.abspath(args.interface_usd)
    physx_layer_path = os.path.join(os.path.dirname(interface), "payloads", "Physics", "physx.usda")
    if not os.path.isfile(physx_layer_path):
        print(f"error: PhysX layer not found at {physx_layer_path}", file=sys.stderr)
        return 1

    # 1. Find every rigid body on the fully composed stage.
    stage = Usd.Stage.Open(interface)
    body_paths = [
        prim.GetPath()
        for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if not body_paths:
        print("error: no rigid bodies found on the composed stage", file=sys.stderr)
        return 1

    # 2. Author overs in the PhysX layer so the change lives with the other PhysX attributes.
    layer = Sdf.Layer.FindOrOpen(physx_layer_path)
    added = 0
    with Sdf.ChangeBlock():
        for path in body_paths:
            spec = Sdf.CreatePrimInLayer(layer, path)
            spec.specifier = Sdf.SpecifierOver
            schemas = spec.GetInfo("apiSchemas") if spec.HasInfo("apiSchemas") else Sdf.TokenListOp()
            if API not in schemas.prependedItems and API not in schemas.explicitItems:
                schemas.prependedItems = list(schemas.prependedItems) + [API]
                spec.SetInfo("apiSchemas", schemas)
                added += 1
            attr = spec.properties.get(THRESHOLD_ATTR)
            if attr is None:
                attr = Sdf.AttributeSpec(spec, THRESHOLD_ATTR, Sdf.ValueTypeNames.Float)
            attr.default = float(args.threshold)
    layer.Save()

    # 3. Verify on the composed stage.  Outside of Kit the PhysX schema plugin is not registered, so
    #    ``GetAppliedSchemas()`` would silently drop the token; inspect the composed metadata instead.
    def _has_api(prim: Usd.Prim) -> bool:
        list_op = prim.GetMetadata("apiSchemas")
        if list_op is None:
            return False
        items = list(list_op.explicitItems) + list(list_op.prependedItems) + list(list_op.appendedItems)
        return API in items

    with_api = sum(
        1
        for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and _has_api(prim)
    )
    print(f"rigid bodies: {len(body_paths)}  newly patched: {added}  with {API}: {with_api}")
    print(f"layer written: {physx_layer_path}")
    return 0 if with_api == len(body_paths) else 2


if __name__ == "__main__":
    sys.exit(main())
