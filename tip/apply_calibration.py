"""Make the view-indexed CG-SSF the default configuration with the calibrated noise parameters
(rounded to three decimals, as printed in Section VI-B).

  python tip/apply_calibration.py results/calibration_view/calibration.json
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cal = json.load(open(sys.argv[1]))
assert cal.get("q_mode") == "view", "expected a view-indexed calibration"
f = cal["final"]
p = os.path.join(ROOT, "backend", "config.py")
s = open(p, encoding="utf-8").read()
vals = {"CGSSF_ETA_S": f["eta_s"], "CGSSF_ETA_B": f["eta_b"], "CGSSF_SIGMA_S": f["sigma_s"], "CGSSF_SIGMA_B": f["sigma_b"]}
for k, v in vals.items():
    s, n = re.subn(rf"(\n    {k} = )[0-9.eE+-]+", rf"\g<1>{round(v, 3)}", s)
    assert n == 1, k
if "CGSSF_Q_MODE" not in s:
    a = "    # Process noise eta_s, eta_b (per sqrt(s)), calibrated once on the held-out\n"
    assert a in s
    s = s.replace(a, "    # Process noise indexed by the change of view (rotation angle in rad + translation / median\n"
                     "    # scene depth); \"time\" restores the earlier time-indexed model (ablation (f)).\n"
                     "    CGSSF_Q_MODE = \"view\"\n" + a)
s = s.replace("    # Process noise eta_s, eta_b (per sqrt(s)), calibrated once on the held-out\n"
              "    # TUM fr2/xyz sequence with tip/calibrate_eta.py (see results/calibration).\n",
              "    # Process noise eta_s, eta_b (per sqrt(unit of view change)), calibrated once on the held-out\n"
              "    # TUM fr2/xyz sequence with tip/calibrate_eta.py --q-mode view (see results/calibration_view).\n")
open(p, "w", encoding="utf-8").write(s)
print("config:", {k: round(v, 3) for k, v in vals.items()}, "q_mode=view")
