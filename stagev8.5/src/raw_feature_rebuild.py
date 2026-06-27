"""Stagev8.5 fresh raw-data E/M/L reconstruction using unmodified Stagev5 feature code.

This module intentionally does not ship or seed historical M/L caches.  It calls copied
Stagev5 source functions directly, starting with empty runtime caches under output/.  The
only inputs bundled in input/ are the three raw CSV files.  Stagev8.5 training remains
independent and starts only after the exact Stagev5 feature contract validates E=61, M=1024,
L=8.
"""
from __future__ import annotations
import hashlib, json, os, shutil, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd

from . import config as cfg
from .progress import StageProgress, progress_section
from .source_integrity import verify_strict_reference_sources
from .curl_schannel_transport import install_curl_schannel_transport, transport_audit

if str(cfg.ASSETS) not in sys.path:
    sys.path.insert(0, str(cfg.ASSETS))

# Imported directly from copied Stagev5 source.  No replacement client, no request wrapper,
# no alternate windowing, no cache-key modification, and no content fallback is used.
from stagev5_exact.src.reference_stagev2.stagev1_features import FEATURE_VERSION, generate_all_early_features
from stagev5_exact.src.reference_stagev2.api_feature_extraction import run_middle_embeddings
from stagev5_exact.src.reference_stagev4 import config as v4_config
from stagev5_exact.src.reference_stagev4 import late_extract
from stagev5_exact.src import feature_adapter as v5_adapter

RAW_COUNTS={"ad":87,"control":79,"test":71}
RAW_REQUIRED={"Speech","label","mmse"}

def utc()->str:
    return datetime.now(timezone.utc).isoformat()

def sha256_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024), b''):
            h.update(c)
    return h.hexdigest()

def _json(path:Path,payload:dict)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def raw_input_audit(root:Path|None=None)->dict[str,Any]:
    out={}
    for split,name in cfg.RAW_FILES.items():
        p=cfg.INPUT_RAW_DIR/name
        if not p.exists(): raise FileNotFoundError(p)
        df=pd.read_csv(p)
        missing=RAW_REQUIRED-set(df.columns)
        if missing: raise ValueError(f'{p.name} missing {sorted(missing)}')
        if len(df)!=RAW_COUNTS[split]: raise ValueError(f'{p.name} rows={len(df)} expected={RAW_COUNTS[split]}')
        blank=int(df['Speech'].fillna('').astype(str).str.strip().eq('').sum())
        if blank: raise ValueError(f'{p.name} contains {blank} blank transcripts')
        out[split]={"path":str(p.resolve()),"sha256":sha256_file(p),"rows":len(df),"columns":list(df.columns),"blank_text":blank}
    return {"status":"pass","raw_files":out,"input_mode":"raw_only","historical_feature_or_cache_input":False}

def check_api_environment()->dict[str,Any]:
    key=os.getenv('MAAS_API_KEY','').strip()
    return {
        "status":"pass" if key else "fail",
        "MAAS_API_KEY":"detected" if key else "missing",
        "base_url":os.getenv('HUAWEI_MAAS_BASE_URL','https://api.modelarts-maas.com/v1'),
        "feature_rebuild_policy":"fresh_api_calls_via_unmodified_copied_Stagev5_source",
        "middle_source_function":"reference_stagev2.api_feature_extraction.run_middle_embeddings",
        "late_source_function":"reference_stagev4.late_extract.extract_late_scores",
        "middle_model":"bge-m3",
        "late_model":"qwen3-235b-a22b",
        "transport_policy":"Windows curl.exe/Schannel runtime transport shim; copied Stagev5/Stagev4 source files and feature algorithms remain hash-locked and unchanged",
        "transport_requirement":"curl.exe must be available on PATH; direct connection with certificate verification remains enabled",
        "api_request_sent":False,
        "note":"check_api validates environment only. extract_features performs fresh API reconstruction from empty runtime M/L caches using the copied Stagev5/Stagev4 feature functions and a transport-only Windows Schannel shim.",
    }

def _clean(force:bool)->None:
    # self_check/check_api may write output/checks. Those audit files must not block the first
    # feature extraction. Only generated feature and extraction-runtime artifacts count here.
    targets=[cfg.FEATURE_ROOT,cfg.EXTRACTION_ROOT]
    existing=any(p.exists() and any(p.rglob('*')) for p in targets)
    if existing and not force:
        raise FileExistsError('Existing Stagev8.5 generated feature/runtime artifacts found. Use --force to replace generated features and invalidate known Stagev8.5 training outputs before a fresh E/M/L API reconstruction.')
    if force:
        for p in [*targets, cfg.FINAL, cfg.MODELS]:
            if p.exists(): shutil.rmtree(p)
        # Clear only Stagev8.5-specific checks. General runtime progress logging remains available.
        if cfg.CHECKS.exists():
            for p in cfg.CHECKS.glob('stagev8_5_*'):
                if p.is_file(): p.unlink()
                elif p.is_dir(): shutil.rmtree(p)
    for p in [cfg.EARLY_DIR,cfg.MIDDLE_DIR,cfg.LATE_DIR,cfg.STAGEV2_EXTRACTION_ROOT,cfg.STAGEV4_EXTRACTION_ROOT]:
        p.mkdir(parents=True,exist_ok=True)

def _configure_v4_runtime()->None:
    r=cfg.STAGEV4_EXTRACTION_ROOT
    v4_config.ROOT=cfg.ROOT
    v4_config.INPUT_DIR=cfg.ROOT/'input'
    v4_config.RAW_DIR=cfg.INPUT_RAW_DIR
    # Fresh Stagev4 metadata is reconstructed from raw input through copied Stagev2 preprocessing.
    v4_config.FROZEN_DIR=r/'fresh_metadata'
    v4_config.CONFIG_DIR=cfg.STAGEV5_EXACT_ROOT/'configs'/'reference_stagev4'
    v4_config.OUTPUT_DIR=r
    v4_config.CACHE_DIR=r/'cache'
    v4_config.FEATURE_DIR=r/'features'
    v4_config.DIAGNOSTICS_DIR=r/'diagnostics'
    v4_config.METRICS_DIR=r/'metrics'
    v4_config.FIGURES_DIR=r/'figures'
    v4_config.FINAL_DIR=r/'final_report'
    v4_config.LOG_DIR=r/'logs'
    v4_config.SUMMARY_DIR=r/'summary'

def _fresh_late_metadata(pre:dict[str,pd.DataFrame])->tuple[pd.DataFrame,pd.DataFrame]:
    # Exact sample IDs/text/MMSE/labels come from copied Stagev2 preprocess; no text manipulation occurs here.
    train=pd.concat([pre['ad'],pre['control']],ignore_index=True)
    external=pre['test'].copy()
    def make(df:pd.DataFrame)->pd.DataFrame:
        required={'sample_id','text','disease_label','mmse'}
        missing=required-set(df.columns)
        if missing: raise ValueError(f'Copied Stagev2 preprocessing output missing {sorted(missing)}')
        out=df[['sample_id','text','disease_label','mmse']].copy().rename(columns={'disease_label':'label'})
        if out['sample_id'].astype(str).duplicated().any(): raise ValueError('Duplicate sample_id in reconstructed Stagev4 metadata.')
        if out['text'].fillna('').astype(str).str.strip().eq('').any(): raise ValueError('Blank text in reconstructed Stagev4 metadata.')
        return out
    return make(train),make(external)

def _write_late(late_train:pd.DataFrame,late_external:pd.DataFrame)->None:
    old=v5_adapter.cfg.LATE_DIR
    v5_adapter.cfg.LATE_DIR=cfg.LATE_DIR
    try:
        v5_adapter._write_stagev2_compatible_late(late_train,late_external)
    finally:
        v5_adapter.cfg.LATE_DIR=old

def _copy_em(early_dir:Path,middle_dir:Path)->None:
    mapping=[
        (early_dir/'ad_BM25.csv',cfg.EARLY_DIR/'ad_BM25.csv'),
        (early_dir/'control_BM25.csv',cfg.EARLY_DIR/'control_BM25.csv'),
        (early_dir/'test_BM25.csv',cfg.EARLY_DIR/'test_BM25.csv'),
        (middle_dir/'ad_embedding.csv',cfg.MIDDLE_DIR/'ad_embedding.csv'),
        (middle_dir/'control_embedding.csv',cfg.MIDDLE_DIR/'control_embedding.csv'),
        (middle_dir/'test_embedding.csv',cfg.MIDDLE_DIR/'test_embedding.csv'),
    ]
    for s,d in mapping:
        if not s.exists(): raise FileNotFoundError(f'Expected copied Stagev5 output is missing: {s}')
        shutil.copy2(s,d)

def _validate()->dict[str,Any]:
    olds=(v5_adapter.cfg.EARLY_DIR,v5_adapter.cfg.MIDDLE_DIR,v5_adapter.cfg.LATE_DIR)
    v5_adapter.cfg.EARLY_DIR,v5_adapter.cfg.MIDDLE_DIR,v5_adapter.cfg.LATE_DIR=cfg.EARLY_DIR,cfg.MIDDLE_DIR,cfg.LATE_DIR
    try:
        return v5_adapter.validate_feature_outputs()
    finally:
        v5_adapter.cfg.EARLY_DIR,v5_adapter.cfg.MIDDLE_DIR,v5_adapter.cfg.LATE_DIR=olds

def rebuild_all_features_from_raw(root:Path,force:bool=False)->dict[str,Any]:
    env=check_api_environment()
    if env['status']!='pass': raise RuntimeError('MAAS_API_KEY is required for fresh Stagev5 BGE-M3 and P4/F8 API reconstruction.')
    progress=StageProgress('extract_features',total=11,root=root)
    try:
        with progress_section(progress,'verify copied Stagev5 feature source and Stagev6 loader hashes'):
            integrity=verify_strict_reference_sources()
        with progress_section(progress,'audit raw input CSV files'):
            raw=raw_input_audit()
        with progress_section(progress,'validate MaaS environment without sending test requests'):
            api=check_api_environment()
        with progress_section(progress,'clear Stagev8.5 runtime output and initialize empty fresh caches',force=force):
            _clean(force)
        with progress_section(progress,'install Windows Schannel transport shim without modifying copied Stagev5/Stagev4 source'):
            transport=install_curl_schannel_transport()
        with progress_section(progress,'run unmodified Stagev5 Stagev2 preprocessing and E BM25 extraction'):
            pre,early_dir,early_manifest=generate_all_early_features(cfg.INPUT_RAW_DIR,cfg.STAGEV2_EXTRACTION_ROOT,FEATURE_VERSION)
        with progress_section(progress,'run unmodified Stagev5 BGE-M3 window embedding extraction from empty runtime cache'):
            middle_dir=cfg.STAGEV2_EXTRACTION_ROOT/'features'/'embedding'
            middle_manifest=run_middle_embeddings(pre,middle_dir,cfg.STAGEV2_EXTRACTION_ROOT/'cache',reuse_cache=True)
            _copy_em(early_dir,middle_dir)
        with progress_section(progress,'reconstruct Stagev4 late metadata from copied Stagev5 preprocessing output'):
            train_meta, external_meta=_fresh_late_metadata(pre)
            fresh_meta=v4_config.OUTPUT_DIR/'fresh_metadata'
            # config not yet redirected; stage runtime directory is deterministic.
            fresh_meta=cfg.STAGEV4_EXTRACTION_ROOT/'fresh_metadata'; fresh_meta.mkdir(parents=True,exist_ok=True)
            train_meta.to_csv(fresh_meta/'train_metadata.csv',index=False,encoding='utf-8')
            external_meta.to_csv(fresh_meta/'external_metadata.csv',index=False,encoding='utf-8')
        with progress_section(progress,'run unmodified Stagev5/Stagev4 P4/F8 late scorer from empty runtime cache'):
            _configure_v4_runtime()
            late_train=late_extract.extract_late_scores(train_meta,'train',force=True)
            late_external=late_extract.extract_late_scores(external_meta,'external',force=True)
            late_train=train_meta[['sample_id','label','mmse']].merge(late_train,on='sample_id',how='inner',validate='one_to_one')
            late_external=external_meta[['sample_id','label','mmse']].merge(late_external,on='sample_id',how='inner',validate='one_to_one')
            _write_late(late_train,late_external)
        with progress_section(progress,'run original Stagev5 E/M/L feature schema and dimension validation'):
            validation=_validate()
        l_sources={
            'train':late_train.get('late_llm_source',pd.Series(dtype=str)).value_counts().to_dict(),
            'external':late_external.get('late_llm_source',pd.Series(dtype=str)).value_counts().to_dict(),
        }
        audit={
            'status':'pass','created_at':utc(),
            'feature_rebuild_policy':'fresh_api_calls_through_unmodified_copied_Stagev5_feature_code',
            'raw_input_audit':raw,'strict_integrity':integrity,'api_environment':api,
            'transport_audit':transport_audit(),
            'early_manifest':early_manifest,'middle_manifest':middle_manifest,
            'middle_new_cache_rows':int(middle_manifest.get('new_cache_rows',0)),
            'late_source_counts':l_sources,
            'late_api_scored_samples':int(sum(v.get('api',0) for v in l_sources.values())),
            'stagev5_validation':validation,
            'runtime_cache_paths':{
                'middle':str(cfg.STAGEV2_EXTRACTION_ROOT/'cache'/'huawei_bge_m3_embedding_cache.csv'),
                'late':str(cfg.STAGEV4_EXTRACTION_ROOT/'cache'/'late_p4_unmasked_cache.csv'),
            },
        }
        _json(cfg.CHECKS/'stagev8_5_fresh_stagev5_feature_rebuild_audit.json',audit)
        progress.done('fresh Stagev5 E/M/L API reconstruction completed',middle_new_cache_rows=audit['middle_new_cache_rows'],late_api_scored_samples=audit['late_api_scored_samples'])
        return audit
    except Exception as exc:
        progress.fail('feature extraction failed',error=repr(exc)); raise
