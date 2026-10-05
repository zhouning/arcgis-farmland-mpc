from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[1]
DOI = "10.5281/zenodo.23163184"
files = {
    "main_pdf": ROOT / "01_main_document/01_main_manuscript_scirep.pdf",
    "anonymous_pdf": ROOT / "01_main_document/anonymous/manuscript_scirep_anonymous.pdf",
    "supplement_pdf": ROOT / "03_supplementary_information/02_supplementary_information_scirep_round1.pdf",
    "response_pdf": ROOT / "07_zenodo_release/point_by_point_response_round1.pdf",
    "main_source": ROOT / "05_source_editable/manuscript_scirep.tex",
    "anonymous_source": ROOT / "05_source_editable/manuscript_scirep_anonymous.tex",
    "response_source": ROOT / "07_zenodo_release/point_by_point_response_round1.tex",
}
checks = {name: {"exists": path.is_file(), "bytes": path.stat().st_size if path.is_file() else 0}
          for name, path in files.items()}
for name, rel in {
    "main_log": "05_source_editable/manuscript_scirep.log",
    "supplement_log": "05_source_editable/supplementary_information_scirep.log",
    "response_log": "07_zenodo_release/point_by_point_response_round1.log",
}.items():
    path = ROOT / rel
    text = path.read_text(errors="ignore") if path.is_file() else ""
    checks[name] = {"exists": path.is_file(), "undefined_refs": bool(re.search(r"undefined references|There were undefined", text, re.I))}
response = files["response_source"].read_text(errors="ignore") if files["response_source"].is_file() else ""
checks["response_author_input_needed"] = bool(re.search(r"AUTHOR\\?_INPUT\\?_NEEDED", response))
checks["doi_in_manuscript"] = DOI in files["main_source"].read_text(errors="ignore") if files["main_source"].is_file() else False
checks["doi_in_response"] = DOI in response
checks["doi_in_citation"] = DOI in (ROOT.parent.parent / "CITATION.cff").read_text(errors="ignore")
checks["pipeline_figure"] = (ROOT / "05_source_editable/figures/pipeline_workflow.pdf").is_file()
checks["local_inset"] = ((ROOT / "05_source_editable/figures/parcel_local_inset.pdf").is_file()
                         and (ROOT / "04_figures/parcel_local_inset.pdf").is_file())
report = ROOT / "07_zenodo_release/preflight_report.json"
report.write_text(json.dumps(checks, indent=2), encoding="utf-8")
print(json.dumps(checks, indent=2))

