from __future__ import annotations
import json, sys
from pathlib import Path
from . import stagev6_config as cfg
from .data_io import load_train_test, infer_schema


def run_self_check(require_features: bool = True) -> dict:
    required=[cfg.ROOT/'run_stagev6.py',cfg.ROOT/'README.md',cfg.ROOT/'notebooks/stagev6_result_check.ipynb',cfg.ROOT/'src/runner.py',cfg.ROOT/'src/cascade.py',cfg.ROOT/'src/model_registry.py']
    required += [cfg.INPUT_RAW/x for x in ['ad_s2t_wav2vec.csv','control_s2t_wav2vec.csv','test_s2t_wav2vec.csv']]
    for d,names in [(cfg.EARLY_DIR,['ad_BM25.csv','control_BM25.csv','test_BM25.csv']),(cfg.MIDDLE_DIR,['ad_embedding.csv','control_embedding.csv','test_embedding.csv']),(cfg.LATE_DIR,['ad_LLM.csv','control_LLM.csv','test_LLM.csv'])]:
        required += [d/n for n in names]
    missing=[str(p.relative_to(cfg.ROOT)) for p in required if not p.exists()]
    status={}
    if not missing and require_features:
        try:
            tr,te,info=load_train_test(); schema=infer_schema(tr,te)
            status={'source_info':info,'schema':schema,'expected_train_n':166,'actual_train_n':len(tr),'expected_external_n':71,'actual_external_n':len(te),'train_late_n':int(tr.label_late.sum()),'external_late_n':int(te.label_late.sum())}
            if len(tr)!=166 or len(te)!=71 or int(tr.label_late.sum())!=16 or int(te.label_late.sum())!=7: missing.append('feature label counts do not match locked stagev5 artifacts')
        except Exception as exc:
            missing.append(f'feature load/validation: {exc!r}')
    result={'passed':not missing,'missing':missing,'python':sys.version,'feature_status':status,'no_api_required':True,'stagev6_protocol':{'gate':'late vs non-late, L or M+L','branch':'true non-late AD/control, E+M','model_family':'stagev5 LR/SVC family plus stagev5 branch MLP anchor','cv':'10x1 repeated stratified CV','selection_metric':'external_test_accuracy','selection_data_split':'held-out external test'}}
    p=cfg.OUTPUT/'checks'/'self_check_report.json'; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
    if not result['passed']: raise SystemExit(1)
    return result
