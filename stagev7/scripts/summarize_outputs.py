from pathlib import Path
import pandas as pd
root=Path(__file__).resolve().parents[1]/'output'/'final_report'
for name in ['stagev7_final_primary_performance.csv','stagev7_cascade_ranking_external_exploratory.csv','stagev7_flat_multiclass_baseline_performance.csv']:
    p=root/name
    print('\n###',name)
    print(pd.read_csv(p).to_string(index=False))
