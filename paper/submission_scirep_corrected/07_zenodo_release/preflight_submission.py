from pathlib import Path
import re, json, subprocess
ROOT=Path(__file__).resolve().parents[1]
checks={}
for p in [ROOT/'01_main_document/anonymous/manuscript_scirep_anonymous.pdf',ROOT/'03_supplementary_information/02_supplementary_information_scirep_round1.pdf',ROOT/'07_revision_response/point_by_point_response_round1.pdf']:
    checks[p.name]={'exists':p.exists(),'bytes':p.stat().st_size if p.exists() else 0}
for rel in ['05_source_editable/manuscript_scirep.log','05_source_editable/supplementary_information_scirep.log','07_revision_response/point_by_point_response_round1.log']:
    p=ROOT/rel; txt=p.read_text(errors='ignore') if p.exists() else ''; checks[rel]={'undefined_refs':bool(re.search(r'undefined references|There were undefined',txt,re.I))}
checks['response_author_input_needed']=len(re.findall(r'AUTHOR\\\\?_INPUT\\\\?_NEEDED', (ROOT/'07_revision_response/point_by_point_response_round1.tex').read_text(errors='ignore')))
checks['pipeline_figure']=(ROOT/'05_source_editable/figures/pipeline_workflow.pdf').exists()
checks['local_inset']=(ROOT/'05_source_editable/figures/parcel_local_inset.pdf').exists() and (ROOT/'04_figures/parcel_local_inset.pdf').exists()
out=ROOT/'08_revision_experiments/preflight_report.json'; out.write_text(json.dumps(checks,indent=2),encoding='utf-8'); print(json.dumps(checks,indent=2))


