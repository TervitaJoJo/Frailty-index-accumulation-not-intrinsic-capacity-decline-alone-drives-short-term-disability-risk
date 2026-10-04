"""Create ASCII hard-link paths for the three SHARE wave-6 files used by R.

R on this Windows installation cannot open the original non-ASCII SHARE
directory reliably. Hard links share the original file contents and do not
copy, edit, or export person-level data. The links can be removed safely after
the R pilot; rerun this helper before reproducing the pilot if needed.
"""
from pathlib import Path
import os

ROOT = Path(r"PATH_TO_STATISTICAL_MODELING")
OUT = Path(r"PATH_TO_IC_FRAILTY")
share_parent_candidates = [p for p in (ROOT / "SHARE").iterdir() if p.is_dir()]
if len(share_parent_candidates) != 1:
    raise RuntimeError(f"Expected one SHARE release directory; found {len(share_parent_candidates)}")
share_parent = share_parent_candidates[0]
target = OUT / "_share_ascii_real"
paths = [
    (share_parent / "Working_data" / "share.dta", target / "Working_data" / "share.dta"),
    (share_parent / "Raw_data" / "Wave 6 Release 9.0.0" / "sharew6_rel9-0-0_cf.dta",
     target / "Raw_data" / "Wave 6 Release 9.0.0" / "sharew6_rel9-0-0_cf.dta"),
    (share_parent / "Raw_data" / "Wave 6 Release 9.0.0" / "sharew6_rel9-0-0_gv_health.dta",
     target / "Raw_data" / "Wave 6 Release 9.0.0" / "sharew6_rel9-0-0_gv_health.dta"),
]
for source, dest in paths:
    if not source.is_file():
        raise FileNotFoundError(source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if dest.stat().st_ino == source.stat().st_ino:
            continue
        dest.unlink()
    os.link(source, dest)
print(f"Prepared {len(paths)} ASCII hard links under {target}")
