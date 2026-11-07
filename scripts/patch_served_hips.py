#!/usr/bin/env python3
"""Patch on-disk HiPS under the :3005 serve root for Aladin Lite.

Fixes:
  * ``hips_pixel_bitpix`` (reproject labels float32 tiles as ``-64``)
  * stale ``index.html`` that loads Aladin from the CDS CDN

Requires a host with **read-write** access to the HiPS root (lwacalim09
currently mounts ``/lustre`` read-only).

Usage::

    PYTHONPATH=src python scripts/patch_served_hips.py \\
        /lustre/pipeline/calibration/hips
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lwa_healpix import upgrade_hips


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "hips_root",
        type=Path,
        nargs="?",
        default=Path("/lustre/pipeline/calibration/hips"),
        help="Directory containing *.hips surveys (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List surveys that would be patched without writing",
    )
    args = parser.parse_args(argv)

    root: Path = args.hips_root
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1

    surveys = sorted(p for p in root.glob("*.hips") if p.is_dir())
    if not surveys:
        print(f"No *.hips directories under {root}", file=sys.stderr)
        return 1

    for survey in surveys:
        props = survey / "properties"
        index = survey / "index.html"
        needs_bitpix = False
        needs_index = False
        if props.is_file():
            text = props.read_text()
            needs_bitpix = "hips_pixel_bitpix" in text and " = -64" in text
        if index.is_file():
            needs_index = "aladin.cds.unistra.fr" in index.read_text()
        elif not index.exists():
            needs_index = True

        status = []
        if needs_bitpix:
            status.append("bitpix")
        if needs_index:
            status.append("index")
        if not status:
            status.append("ok")

        print(f"{survey.name}: {', '.join(status)}")
        if args.dry_run or status == ["ok"]:
            continue
        upgrade_hips(survey, refresh_index=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
