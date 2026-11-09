"""Fill report_template.html with a collect.py JSON.

    python experiment/015_ckda_complex_eigenvalues/scripts/build_report.py <data.json> <out.html>
"""

import sys
from pathlib import Path

tpl = (Path(__file__).parent / "report_template.html").read_text()
data = Path(sys.argv[1]).read_text()
Path(sys.argv[2]).write_text(tpl.replace("/*DATA*/null", data))
print(sys.argv[2])
