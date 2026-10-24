"""Application shell, dataset selection, content-aware caching and navigation."""
from io import BytesIO
from pathlib import Path
import os
import html

import pandas as pd
import streamlit as st

from retention.data import load_data, dataset_fingerprint
from retention.modeling import train_models, load_bundle, save_bundle, score_customers
from retention.dashboard.style import inject_style
from retention.dashboard import pages

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts" / "default_bundle.joblib"
NAV = ["Executive Overview", "Exploratory Analysis", "Model Comparison", "Explainability", "Customer Risk Explorer", "Retention Centre", "Economic Impact", "Methodology & Journey"]


@st.cache_data(show_spinner=False)
def read_dataset(contents: bytes, require_target=True):
    return load_data(BytesIO(contents), require_target=require_target)


@st.cache_resource(show_spinner=False)
def read_bundle(path: str, modified_ns: int):
    return load_bundle(path)


@st.cache_data(show_spinner=False)
def customer_scores(data_hash, model_name, artifact_token, _bundle, df):
    # df is included in the key; the data hash guards artifacts from another dataset.
    return score_customers(_bundle, df, model_name)


def _data_source():
    with st.sidebar.expander("Dataset & data quality", expanded=False):
        upload = st.file_uploader("Upload labelled customer CSV", type=["csv"], key="dataset_upload")
        path = st.text_input("Local dataset path", value=os.environ.get("TELCO_DATA_PATH", str(ROOT / "data" / "Telco-Customer-Churn.csv")))
        st.caption("A training dataset needs the Churn label. Use Customer Risk Explorer for unlabelled batch scoring.")
    fallback = False
    try:
        if upload is not None:
            raw, name = upload.getvalue(), upload.name
        else:
            target = Path(path).expanduser()
            if not target.exists():
                target = ROOT / "data" / "sample_telco.csv"
                fallback = True
            if not target.exists():
                st.error("No dataset was found. Upload an IBM Telco CSV or enter its local path in the sidebar.")
                st.stop()
            raw, name = target.read_bytes(), target.name
        df, quality = read_dataset(raw)
    except Exception as exc:
        st.error(f"Dataset validation failed: {exc}")
        st.stop()
    if fallback:
        st.warning("The configured dataset is unavailable. This view uses a clearly labelled fallback sample; results describe only that sample.")
    return df, quality, name, fallback


def _filters(df):
    with st.sidebar.expander("Business filters", expanded=False):
        contracts = st.multiselect("Contract", sorted(df.Contract.dropna().unique()), default=[])
        internet = st.multiselect("Internet service", sorted(df.InternetService.dropna().unique()), default=[])
        observed_tenure = df.tenure.dropna()
        lower, upper = (int(observed_tenure.min()), int(observed_tenure.max())) if len(observed_tenure) else (0,0)
        tenure = st.slider("Tenure (months)", lower, upper, (lower, upper)) if upper > lower else (lower, upper)
        include_missing = st.checkbox("Include records with missing tenure", value=True)
        st.caption("Applies to historical charts, recommendations and risk records. Held-out model evaluation stays fixed.")
    mask = df.tenure.between(*tenure)
    if include_missing:
        mask |= df.tenure.isna()
    if contracts:
        mask &= df.Contract.isin(contracts)
    if internet:
        mask &= df.InternetService.isin(internet)
    return df.loc[mask].copy()


def _model(df, data_hash):
    bundle = st.session_state.get("trained_bundle")
    if bundle is not None and bundle.get("data_hash") != data_hash:
        bundle = None
        st.session_state.pop("trained_bundle", None)
    if bundle is None and ARTIFACT.exists():
        try:
            candidate = read_bundle(str(ARTIFACT), ARTIFACT.stat().st_mtime_ns)
            if candidate.get("data_hash") == data_hash:
                bundle = candidate
        except Exception as exc:
            st.sidebar.warning(f"Saved model could not be loaded: {exc}")
    with st.sidebar.expander("Model & risk policy", expanded=bundle is None):
        if bundle is None:
            st.caption("No matching fitted artifact. Train the models for this exact dataset to unlock predictions.")
        else:
            st.caption(f"Artifact generated: {bundle.get('generated_at', 'unavailable')}")
        fast = st.checkbox("Quick demonstration training", value=False, help="Uses a smaller tuning search; still uses a fixed held-out test set.")
        if st.button("Train / retrain models", type="primary", width="stretch"):
            try:
                with st.spinner("Fitting preprocessing, tuning on training data and evaluating held-out customers…"):
                    bundle = train_models(df, random_state=42, fast=fast)
                    # Uploaded datasets remain in this session; bundled dataset artifacts are persisted.
                    st.session_state.trained_bundle = bundle
                    default_path = ROOT / "data" / "Telco-Customer-Churn.csv"
                    if default_path.exists() and dataset_fingerprint(load_data(default_path)[0]) == data_hash:
                        ARTIFACT.parent.mkdir(exist_ok=True)
                        save_bundle(bundle, ARTIFACT)
                st.success("Training complete. Model comparison uses one fixed held-out test set.")
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                with st.expander("Training diagnostic"):
                    st.exception(exc)
        if bundle is not None:
            options = list(bundle["models"])
            default = bundle.get("preferred_model", options[-1])
            model = st.selectbox("Prediction model", options, index=options.index(default) if default in options else 0)
        else:
            model = None
        low = st.slider("Low / medium boundary", 0.05, 0.75, 0.30, 0.05, format="%.2f")
        high = st.slider("Medium / high boundary", float(low + .05), 1.0, max(.60, float(low + .05)), .05, format="%.2f")
        st.caption("Risk bands are an editable outreach policy. They are not statistically established boundaries or certainty of churn.")
    return bundle, model, low, high


def main():
    st.set_page_config(page_title="Retention Lab · Customer Analytics", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
    inject_style()
    st.sidebar.markdown('<div class="brand">RETENTION <span>/ LAB</span></div><div class="brand-note">CUSTOMER INTELLIGENCE · POME</div>', unsafe_allow_html=True)
    st.sidebar.caption("WORKSPACE")
    page = st.sidebar.radio("Navigate", NAV, label_visibility="collapsed")
    st.sidebar.divider()
    df, quality, name, fallback = _data_source()
    data_hash = dataset_fingerprint(df)
    filtered = _filters(df)
    bundle, model, low, high = _model(df, data_hash)
    status = "Fitted model available" if bundle is not None else "Historical analysis ready"
    st.sidebar.markdown(f'<div class="status"><b>{status}</b><br>{html.escape(name)}<br>{len(df):,} validated historical records</div>', unsafe_allow_html=True)
    st.sidebar.caption("Academic prototype · IBM Telco sample\nNo live customer connection or automated outreach.")
    context = dict(df=df, filtered=filtered, quality=quality, name=name, fallback=fallback, bundle=bundle, model=model, low=low, high=high, data_hash=data_hash, score_fn=customer_scores)
    renderers = [pages.overview, pages.exploratory, pages.model_comparison, pages.explainability, pages.risk_explorer, pages.retention_centre, pages.economics, pages.methodology]
    renderers[NAV.index(page)](**context)
    st.markdown('<div class="footer">RETENTION / LAB &nbsp; · &nbsp; Principles of Management & Economics &nbsp; · &nbsp; Historical sample analysis, model estimates and editable business scenarios.</div>', unsafe_allow_html=True)
