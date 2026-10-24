"""Eight fully data-driven views for the retention decision-support workflow."""
from io import BytesIO
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, precision_score, recall_score
from sklearn.calibration import calibration_curve

from retention.data import load_data, FEATURE_COLUMNS
from retention.eda import add_tenure_band, segment_summary, executive_insights, descriptive_stats
from retention.modeling import score_customers
from retention.recommendations import customer_recommendations, segment_recommendations, risk_category
from retention.economics import economic_scenario, sensitivity_table
from retention.dashboard.style import chart, page_header, BLUE, TEAL, CORAL, PALETTE

LABELS = {0: "Stayed", 1: "Churned"}
COLORS = {"Stayed": BLUE, "Churned": CORAL, "Low": TEAL, "Medium": "#d7ac58", "High": CORAL}
NICE = {"InternetService":"Internet service", "MonthlyCharges":"Monthly charges", "TotalCharges":"Total charges", "TechSupport":"Technical support", "OnlineSecurity":"Online security", "PaymentMethod":"Payment method", "TenureBand":"Tenure group", "SeniorCitizen":"Senior citizen"}


def _csv(df):
    return df.to_csv(index=False).encode("utf-8")


def _money(value, currency):
    """Keep currency cards readable, with full values available in help/export."""
    for scale,suffix in [(1e12,"T"),(1e9,"B"),(1e6,"M")]:
        if abs(value)>=scale:
            return f"{currency} {value/scale:,.2f}{suffix}"
    return f"{currency} {value:,.0f}"


def _empty(df):
    if len(df) == 0:
        st.info("No historical records match the selected business filters. Adjust the filters in the sidebar.")
        return True
    return False


def _require_model(bundle):
    if bundle is None:
        st.info("Train the models from the sidebar to enable this view. A saved artifact is used only when it matches the loaded dataset.")
        return False
    return True


def _labelled(df):
    result = df.copy()
    result["Historical outcome"] = result.Churn.map(LABELS)
    return result


def _segment_chart(df, feature, title=None):
    summary = segment_summary(df, feature)
    summary["Churn rate (%)"] = summary.churn_rate * 100
    summary["Above"] = (summary.ci_upper - summary.churn_rate) * 100
    summary["Below"] = (summary.churn_rate - summary.ci_lower) * 100
    fig = px.bar(summary, x="segment", y="Churn rate (%)", error_y="Above", error_y_minus="Below", text=summary.customers.map(lambda n:f"n={n:,}"),
                 custom_data=["customers", "churned"], title=title or f"Churn by {NICE.get(feature, feature).lower()}", color_discrete_sequence=[TEAL])
    fig.update_traces(textposition="outside", cliponaxis=False, hovertemplate="%{x}<br>Churn: %{y:.1f}%<br>Customers: %{customdata[0]:,}<br>Churned: %{customdata[1]:,}<extra></extra>")
    fig.update_xaxes(title=NICE.get(feature, feature))
    fig.update_yaxes(range=[0, min(115, max(20, summary.ci_upper.max()*110+5))], ticksuffix="%")
    chart(fig)
    st.caption("Bars show observed churn; whiskers are 95% Wilson confidence intervals. Small groups have greater uncertainty.")
    return summary


def _scored(c):
    result = c["df"].copy()
    result["Predicted churn probability"] = c["score_fn"](c["data_hash"], c["model"], str(c["bundle"].get("generated_at")), c["bundle"], c["df"])
    result["Risk category"] = risk_category(result["Predicted churn probability"], c["low"], c["high"])
    return result


def _customer(df, key):
    labels = df.customerID.astype(str) if "customerID" in df else pd.Series([f"Record {i}" for i in df.index], index=df.index)
    idx = st.selectbox("Customer reference", list(df.index), format_func=lambda i:str(labels.loc[i]), key=key)
    return df.loc[[idx]]


def overview(**c):
    page_header("Customer retention, in focus.", "Understand the customer base. Find where churn concentrates. Turn evidence into a considered next step.")
    df = c["filtered"]
    a,b = st.columns([3,1])
    a.caption(f"DATASET  {c['name']}  ·  {len(df):,} of {len(c['df']):,} validated records in view")
    b.caption("OUTCOMES  Historical, observed")
    if _empty(df): return
    cards = [("Customers", f"{len(df):,}"),("Churned",f"{int(df.Churn.sum()):,}"),("Churn rate",f"{df.Churn.mean():.1%}"),("Stayed",f"{int((df.Churn==0).sum()):,}"),("Average tenure",f"{df.tenure.mean():.1f} mo"),("Average monthly charge",f"{df.MonthlyCharges.mean():.2f}")]
    for start in [0,3]:
        for col,(label,value) in zip(st.columns(3),cards[start:start+3]): col.metric(label,value)
    st.caption("Monthly charges are shown in the dataset's original charge units.")
    st.write("")
    left,right = st.columns([.85,1.15])
    with left:
        count = df.Churn.map(LABELS).value_counts().rename_axis("Historical outcome").reset_index(name="Customers")
        fig = px.pie(count,names="Historical outcome",values="Customers",hole=.72,color="Historical outcome",color_discrete_map=COLORS,title="The overall retention picture")
        fig.update_traces(textposition="outside", textinfo="label+percent",hovertemplate="%{label}<br>%{value:,} customers<br>%{percent}<extra></extra>")
        fig.add_annotation(text=f"<b>{df.Churn.mean():.1%}</b><br>churn rate",x=.5,y=.5,showarrow=False,font_size=20)
        chart(fig)
        st.caption("Observed historical outcomes in the current filtered sample.")
    with right: _segment_chart(df,"Contract","Contract type and observed churn")
    left,right = st.columns(2)
    with left: _segment_chart(df,"InternetService")
    with right: _segment_chart(add_tenure_band(df),"TenureBand")
    fig=px.histogram(_labelled(df),x="MonthlyCharges",color="Historical outcome",nbins=40,barmode="overlay",opacity=.7,color_discrete_map=COLORS,title="Monthly charges across historical outcomes",labels={"MonthlyCharges":"Monthly charges (dataset currency units)"})
    fig.update_yaxes(title="Customer records")
    chart(fig,300)
    st.caption("Count distributions reflect both charge levels and the different sizes of the churn and stayed groups.")
    st.subheader("What the sample tells us")
    for item in executive_insights(df): st.markdown(f'<div class="insight">{html.escape(str(item))}</div>',unsafe_allow_html=True)
    st.caption("These patterns describe the selected IBM sample. They do not establish why customers leave or what an intervention will change.")
    with st.expander("Data quality & preprocessing audit"):
        _quality(c)
    _report(c)


def _quality(c):
    quality=c["quality"]
    cols=st.columns(4)
    for col,label,key in zip(cols,["Rows loaded","Rows usable","Duplicates removed","Service inconsistencies"],["rows_loaded","rows_cleaned","duplicates_removed","inconsistent_service_records"]): col.metric(label,quality.get(key,"—"))
    missing=quality.get("missing_values",{})
    if missing: st.dataframe(pd.DataFrame({"Field":list(missing),"Missing after cleaning":list(missing.values())}),hide_index=True,width="stretch")
    for warning in quality.get("warnings",[]): st.warning(str(warning))
    for decision in quality.get("preprocessing_decisions",[]): st.write("• "+str(decision))
    st.download_button("Download data-quality audit",json.dumps(quality,indent=2,default=str),"data_quality.json","application/json")


def exploratory(**c):
    page_header("Explore the evidence.","Compare customer segments, inspect distributions and assess the strength of observed associations.")
    df=c["filtered"]
    if _empty(df): return
    st.caption(f"HISTORICAL ANALYSIS · {len(df):,} filtered records. All segment rates use observed Churn labels.")
    tabs=st.tabs(["Segments", "Charges & tenure", "Statistical evidence", "Data audit"])
    with tabs[0]:
        features=["Contract","PaymentMethod","InternetService","TechSupport","OnlineSecurity","TenureBand","SeniorCitizen","Partner","Dependents"]
        enriched=add_tenure_band(df)
        for start in range(0,len(features),2):
            for column,feature in zip(st.columns(2),features[start:start+2]):
                with column: _segment_chart(enriched,feature)
        st.caption("Tenure intervals: 0–12, 13–24, 25–48, 49–60 and 61+ completed months. The same definitions are used throughout the dashboard.")
        st.subheader("Compare two segments")
        x,y=st.columns(2)
        first=x.selectbox("Primary segment",["Contract","InternetService","PaymentMethod"],key="segfirst")
        second=y.selectbox("Comparison group",["TechSupport","OnlineSecurity","Partner"],key="segsecond")
        grouped=df.groupby([first,second],observed=True).Churn.agg(["mean","size"]).reset_index()
        grouped["Churn rate (%)"]=grouped["mean"]*100
        fig=px.bar(grouped,x=first,y="Churn rate (%)",color=second,barmode="group",custom_data=["size"],title="Observed churn across combined segments")
        fig.update_traces(hovertemplate="%{x}<br>Churn: %{y:.1f}%<br>Sample size: %{customdata[0]:,}<extra></extra>")
        chart(fig)
        st.caption("Combined segments can be small. Compare the sample sizes before interpreting differences.")
    with tabs[1]:
        labelled=_labelled(df)
        a,b=st.columns(2)
        with a:
            chart(px.box(labelled,x="Historical outcome",y="MonthlyCharges",color="Historical outcome",color_discrete_map=COLORS,title="Monthly charges by observed outcome",labels={"MonthlyCharges":"Monthly charges (currency units)"}))
        with b:
            chart(px.histogram(labelled,x="TotalCharges",color="Historical outcome",nbins=40,barmode="overlay",opacity=.6,color_discrete_map=COLORS,title="Total charges by observed outcome",labels={"TotalCharges":"Total charges (currency units)"}))
        st.caption("Box plots show medians and spread; histograms show counts. Total charges are partly a function of tenure and monthly charges.")
        a,b=st.columns([1.2,1])
        with a:
            chart(px.scatter(labelled,x="tenure",y="MonthlyCharges",color="Historical outcome",color_discrete_map=COLORS,opacity=.4,title="Tenure and monthly charges",labels={"tenure":"Tenure (months)","MonthlyCharges":"Monthly charges (currency units)"}),400)
            st.caption("Each point is one historical customer record; overlapping points can obscure density.")
        with b:
            corr=df[["tenure","MonthlyCharges","TotalCharges","Churn"]].corr()
            chart(px.imshow(corr,text_auto=".2f",zmin=-1,zmax=1,color_continuous_scale="RdBu_r",title="Pearson correlation · numeric fields",labels={"color":"Correlation"}),400)
            st.caption("Churn is coded 1 for churned and 0 for stayed. Linear correlations do not capture every relationship or imply causation.")
        st.subheader("Descriptive statistics")
        st.dataframe(descriptive_stats(df),hide_index=True,width="stretch")
    with tabs[2]:
        from retention.statistics import association_tests, logistic_odds_ratios
        st.subheader("Associations with historical churn")
        st.write("Categorical tests compare churn proportions; numeric Welch tests compare mean values between churn groups. Adjusted p-values account for multiple comparisons. Effect sizes and sample sizes give context beyond statistical significance. Numeric confidence intervals describe raw mean differences, while the standardised effect size is Hedges' g; consult each row's notes and assumptions.")
        try:
            result=association_tests(df)
            st.dataframe(result,hide_index=True,width="stretch")
            st.download_button("Download association results",_csv(result),"association_tests.csv","text/csv")
        except Exception as exc: st.warning(f"Association analysis unavailable for this selection: {exc}")
        st.subheader("Adjusted logistic odds ratios")
        st.caption("An odds ratio above 1 indicates higher estimated churn odds relative to the reference, holding included features fixed. It is not a probability ratio or causal effect.")
        if st.button("Fit statistical odds-ratio analysis",key="fit_stats"):
            with st.spinner("Fitting an explanatory logistic regression…"):
                try:
                    table,note=logistic_odds_ratios(df)
                    st.write(note)
                    st.dataframe(table,hide_index=True,width="stretch")
                    st.download_button("Download odds ratios",_csv(table),"logistic_odds_ratios.csv","text/csv")
                except Exception as exc: st.warning(f"Odds-ratio fit unavailable: {exc}")
        st.info("These observational comparisons describe this selected dataset. Filtering changes the population being tested. Significance alone does not imply a useful retention action.")
    with tabs[3]: _quality(c)


def model_comparison(**c):
    page_header("Compare the models fairly.","One held-out test set. Four fitted models. Clear trade-offs between finding churners and limiting false alarms.","MODEL ESTIMATES · FIXED HELD-OUT TEST RECORDS")
    bundle=c["bundle"]
    if not _require_model(bundle): return
    st.info(f"Preferred model: {bundle['preferred_model']}. Selected using training/validation results; dashboard filters do not alter the held-out test set. Ranking this table does not reselect the model.")
    metrics=bundle["metrics"].copy()
    ranking=st.selectbox("Rank the test-results table by",["pr_auc","roc_auc","f1","recall","precision","accuracy","brier"],format_func=lambda x:{"pr_auc":"PR-AUC","roc_auc":"ROC-AUC","f1":"F1-score","brier":"Brier score (lower is better)"}.get(x,x.capitalize()))
    metrics=metrics.sort_values(ranking,ascending=ranking=="brier")
    st.dataframe(metrics.style.format({col:"{:.3f}" for col in metrics.select_dtypes("number").columns}),hide_index=True,width="stretch")
    preferred=metrics.loc[metrics.model.eq(bundle["preferred_model"])].iloc[0]
    common_threshold=float(bundle.get("metadata",{}).get("metric_threshold",.5))
    st.write(f"At the common {common_threshold:.2f} threshold, the validation-selected {bundle['preferred_model']} identifies {preferred['recall']:.1%} of actual churners in the held-out test sample; {preferred['precision']:.1%} of its churn flags are correct.")
    components=metrics.loc[metrics.model.isin(["Logistic Regression","Random Forest","XGBoost"])]
    hybrid_rows=metrics.loc[metrics.model.eq("Hybrid Ensemble")]
    if not components.empty and not hybrid_rows.empty:
        strongest=components.loc[components.pr_auc.idxmax()]
        hybrid_pr=float(hybrid_rows.pr_auc.iloc[0])
        difference=hybrid_pr-float(strongest.pr_auc)
        if difference>1e-9:
            comparison=f"improves on the highest individual-model PR-AUC by {difference*100:.2f} percentage points"
        elif difference < -1e-9:
            comparison=f"does not improve on the highest individual-model PR-AUC; it is {abs(difference)*100:.2f} percentage points lower"
        else:
            comparison="matches the highest individual-model PR-AUC and does not improve on it"
        st.info(f"Final held-out diagnostic: Hybrid Ensemble PR-AUC is {hybrid_pr:.3f}, versus {strongest['pr_auc']:.3f} for {strongest['model']}. The hybrid {comparison}. This comparison describes the test sample and does not reselect the preferred model, which was chosen using validation data.")
    st.download_button("Download held-out model metrics",_csv(metrics),"model_comparison.csv","text/csv")
    melted=metrics.melt(id_vars="model",value_vars=["accuracy","precision","recall","f1","roc_auc","pr_auc"],var_name="Metric",value_name="Score")
    chart(px.bar(melted,x="Metric",y="Score",color="model",barmode="group",title="Performance on the same held-out customers"),350)
    st.caption("All tabulated classification metrics use the evaluation threshold saved by the training pipeline (normally 0.50). PR-AUC is particularly helpful with unequal class frequencies. Brier score assesses probability accuracy; lower is better.")
    with st.expander("What these metrics mean"):
        st.write("Precision: among historical customers flagged as churners, the share who actually churned. Recall: the share of actual churners identified. F1: a balance of precision and recall. Accuracy: the overall share correctly classified. ROC-AUC: ranking ability across thresholds. PR-AUC: the precision–recall trade-off, with the churn prevalence as a useful baseline. None measures causal campaign success or guarantees profit.")
        st.dataframe(bundle.get("validation_metrics",pd.DataFrame()),hide_index=True,width="stretch")
    y=np.asarray(bundle["y_test"])
    a,b=st.columns(2)
    with a:
        fig=go.Figure()
        for name,p in bundle["probabilities"].items():
            fpr,tpr,_=roc_curve(y,p);fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",name=name))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",line=dict(dash="dot",color="#a4b2c3"),name="Chance ranking"))
        fig.update_layout(title="ROC curves",xaxis_title="False positive rate",yaxis_title="True positive rate (recall)")
        chart(fig,390)
    with b:
        fig=go.Figure()
        for name,p in bundle["probabilities"].items():
            pre,rec,_=precision_recall_curve(y,p);fig.add_trace(go.Scatter(x=rec,y=pre,mode="lines",name=name))
        fig.add_hline(y=y.mean(),line_dash="dot",line_color="#a4b2c3")
        fig.update_layout(title=f"Precision–recall curves · prevalence {y.mean():.1%}",xaxis_title="Recall",yaxis_title="Precision")
        chart(fig,390)
    st.caption("Better ranking curves lie nearer the top; precision–recall depends on the churn prevalence in this historical test sample.")
    selected=st.selectbox("Model for detailed diagnostics",list(bundle["probabilities"]),index=list(bundle["probabilities"]).index(c["model"]),key="diagnosticmodel")
    threshold=float(bundle.get("metadata",{}).get("metric_threshold",.5))
    suggested=float(bundle.get("metadata",{}).get("thresholds",{}).get(selected,.5))
    st.caption(f"Validation maximum-F1 threshold for {selected}: {suggested:.3f}. Final test classification charts below use the common {threshold:.2f} threshold for fair model comparison.")
    p=np.asarray(bundle["probabilities"][selected]); predicted=(p>=threshold).astype(int)
    a,b=st.columns(2)
    with a:
        matrix=confusion_matrix(y,predicted,labels=[0,1])
        fig=px.imshow(matrix,x=["Predicted stayed","Predicted churned"],y=["Actually stayed","Actually churned"],text_auto=True,color_continuous_scale="Blues",title=f"Confusion matrix · threshold {threshold:.2f}",labels={"color":"Customers"})
        chart(fig,350)
        st.caption("Final held-out diagnostic at the fixed comparison threshold. Off-diagonal cells are classification errors.")
    with b:
        if "validation_probabilities" in bundle and "y_validation" in bundle:
            val_y=np.asarray(bundle["y_validation"]);val_p=np.asarray(bundle["validation_probabilities"][selected])
            grid=np.linspace(.01,.99,99)
            trade=pd.DataFrame({"Threshold":grid,"Precision":[precision_score(val_y,val_p>=t,zero_division=0) for t in grid],"Recall":[recall_score(val_y,val_p>=t,zero_division=0) for t in grid]})
            fig=px.line(trade,x="Threshold",y=["Precision","Recall"],title="Validation threshold and the trade-off")
            fig.add_vline(x=suggested,line_dash="dot")
            chart(fig,350)
            st.caption("This curve uses validation records only; the dotted line marks the validation maximum-F1 threshold. Select an operational policy with validation evidence and business capacity.")
        else: st.info("This artifact has no validation predictions. Retrain to inspect the threshold trade-off without using final test labels.")
    a,b=st.columns(2)
    with a:
        outcomes=pd.DataFrame({"Class":["Stayed","Churned"]*2,"Customers":[int((y==0).sum()),int((y==1).sum()),int((predicted==0).sum()),int((predicted==1).sum())],"Summary":["Actual"]*2+["Predicted"]*2})
        chart(px.bar(outcomes,x="Class",y="Customers",color="Summary",barmode="group",title="Actual versus predicted class counts"),300)
    with b:
        dist=pd.DataFrame({"Probability":p,"Historical outcome":pd.Series(y).map(LABELS)})
        chart(px.histogram(dist,x="Probability",color="Historical outcome",barmode="overlay",opacity=.6,nbins=30,color_discrete_map=COLORS,title="Probability distributions · test customers"),300)
    st.caption("Overlapping score distributions show uncertainty. Count summaries do not identify the same customers by themselves.")
    fig=go.Figure()
    for name,probs in bundle["probabilities"].items():
        frac,mean=calibration_curve(y,np.asarray(probs),n_bins=8,strategy="quantile")
        fig.add_trace(go.Scatter(x=mean,y=frac,mode="lines+markers",name=name))
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],name="Perfect calibration",line=dict(dash="dot",color="#a4b2c3")))
    fig.update_layout(title="Probability calibration · held-out diagnostic",xaxis_title="Mean predicted probability in bin",yaxis_title="Observed churn fraction")
    chart(fig,330)
    st.caption("Reliable probabilities follow the diagonal. Test calibration is a final diagnostic; any calibration fitting uses training data only.")
    _report(c)


def explainability(**c):
    from retention.explainability import global_importance, local_explanation, shap_global_summary
    page_header("Understand the score.","Inspect the patterns each model uses, then look at one customer's model-based explanation.","MODEL EXPLANATIONS · ASSOCIATIONS, NOT CAUSES")
    bundle=c["bundle"]
    if not _require_model(bundle): return
    st.info("An explanation describes how a fitted model produces its score. It does not identify a customer's reason for leaving or establish the effect of a retention offer.")
    global_tab,shap_tab,local_tab=st.tabs(["Model features", "SHAP summary", "Individual customer"])
    with global_tab:
        for name in ["Logistic Regression","Random Forest","XGBoost"]:
            st.subheader(name)
            importance=global_importance(bundle,name)
            if importance.empty:
                st.warning("No supported global feature explanation is available for this fitted model.")
                continue
            top=importance.nlargest(18,"importance")
            if name=="Logistic Regression" and "coefficient" in top:
                top=top.sort_values("coefficient")
                custom=[col for col in ["odds_ratio","reference"] if col in top]
                fig=px.bar(top,x="coefficient",y="feature",orientation="h",color="coefficient",color_continuous_scale="Tealrose",title="Logistic coefficients · transformed features",labels={"coefficient":"Change in model log-odds","feature":"Feature"},custom_data=custom)
                if custom==["odds_ratio","reference"]:
                    fig.update_traces(hovertemplate="%{y}<br>Coefficient: %{x:.3f}<br>Odds ratio: %{customdata[0]:.3f}<br>Reference: %{customdata[1]}<extra></extra>")
                fig.add_vline(x=0,line_color="#90a3b7")
                chart(fig,450)
                st.caption("Positive coefficients increase predicted churn odds, holding other encoded inputs fixed; negative coefficients decrease them. Odds ratios are exp(coefficient). Hover for the omitted reference category or numeric training-standard-deviation unit. Coefficients explain the underlying logistic estimator; calibration may change its final probabilities. Penalisation makes these predictive coefficients different from an unpenalised statistical inference model.")
            else:
                top=top.sort_values("importance")
                chart(px.bar(top,x="importance",y="feature",orientation="h",title=f"{name} · leading features",labels={"importance":"Relative model importance","feature":"Feature"},color_discrete_sequence=[TEAL]),450)
                st.caption("Tree importance indicates how the fitted model uses features. It has no positive/negative direction and may favour features with many possible values. Correlated inputs can share importance.")
            with st.expander(f"{name} feature values & method"):
                st.dataframe(importance,hide_index=True,width="stretch")
                st.download_button("Download feature importance",_csv(importance),f"{name.lower().replace(' ','_')}_importance.csv","text/csv",key=f"imp_{name}")
    with shap_tab:
        st.write("SHAP attributes an estimator's output relative to a background reference. It can be costly and is evaluated on a limited sample. The method label states the output scale; a raw estimator explanation need not match the calibrated final probability.")
        shap_model=st.selectbox("Model to explain with SHAP",list(bundle["models"]),key="shapmodel")
        if st.button("Calculate SHAP summary",key="run_shap"):
            with st.spinner("Calculating supported SHAP explanations…"):
                try:
                    result,note=shap_global_summary(bundle,c["df"],shap_model,max_rows=80)
                    st.session_state["shap_result"]=(c["data_hash"],bundle.get("generated_at"),shap_model,result,note)
                except Exception as exc:
                    st.warning(f"SHAP could not be calculated: {exc}. Use model features or the individual perturbation explanation.")
        saved=st.session_state.get("shap_result")
        if saved and saved[:3]==(c["data_hash"],bundle.get("generated_at"),shap_model):
            _,_,_,result,note=saved
            st.write(note)
            if result is not None and not result.empty:
                summary=result.assign(abs_value=result.shap_value.abs()).groupby("feature",as_index=False).abs_value.mean().nlargest(18,"abs_value").sort_values("abs_value")
                a,b=st.columns(2)
                with a:
                    chart(px.bar(summary,x="abs_value",y="feature",orientation="h",color_discrete_sequence=[TEAL],title="Mean absolute SHAP attribution",labels={"abs_value":"Mean |SHAP value|","feature":"Feature"}),480)
                with b:
                    points=result[result.feature.isin(summary.feature)].copy()
                    chart(px.strip(points,x="shap_value",y="feature",color="feature",title="SHAP attribution distribution",labels={"shap_value":"Signed SHAP attribution","feature":"Feature"}),480)
                st.caption("Left: average attribution magnitude across the explained sample. Right: the spread and direction of feature attributions, one point per explained record. Consult the method label for the estimator output scale.")
            else:
                st.warning("SHAP is unavailable for this model/runtime. The Model features tab and individual perturbation method provide the labelled fallback.")
        else: st.caption("Choose a model and calculate a summary. No simulated SHAP values are displayed.")
    with local_tab:
        df=c["filtered"]
        if _empty(df): return
        selected=_customer(df,"explaincustomer")
        probability=float(score_customers(bundle,selected,c["model"])[0])
        a,b,d=st.columns(3)
        a.metric("Predicted churn probability",f"{probability:.1%}")
        b.metric("Risk policy band",risk_category(probability,c["low"],c["high"]))
        d.metric("Historical outcome",LABELS[int(selected.Churn.iloc[0])])
        st.caption("Historical customer scores can include model training records. Use the Model Comparison page for unbiased held-out performance.")
        _local_chart(bundle,selected,c["model"])
        st.dataframe(selected[[f for f in ["customerID","tenure","Contract","InternetService","MonthlyCharges","TechSupport","OnlineSecurity"] if f in selected]],hide_index=True,width="stretch")


def _local_chart(bundle,selected,model):
    from retention.explainability import local_explanation
    try:
        explanation,method=local_explanation(bundle,selected,model)
        st.caption(method)
        if not explanation.empty:
            top=explanation.assign(magnitude=explanation.contribution.abs()).nlargest(10,"magnitude").sort_values("contribution")
            chart(px.bar(top,x="contribution",y="feature",orientation="h",color="contribution",color_continuous_scale="Tealrose",title="Factors affecting this model score",labels={"contribution":"Probability change versus feature reference","feature":"Input feature"}),390)
            st.caption("The fallback varies one input at a time against a training reference. Effects need not add to the score and can involve unrealistic combinations; treat them as model sensitivity, not causal effects.")
            with st.expander("Explanation details"):
                display_explanation=explanation.copy()
                for field in ["value","baseline"]:
                    if field in display_explanation:
                        display_explanation[field]=display_explanation[field].map(str)
                st.dataframe(display_explanation,hide_index=True,width="stretch")
    except Exception as exc: st.warning(f"Individual explanation unavailable: {exc}")


def _historical_risk_explorer(c):
    df=_scored(c).loc[c["filtered"].index]
    a,b=st.columns([2,1])
    bands=a.multiselect("Risk category",["Low","Medium","High"],default=["Low","Medium","High"])
    order=b.selectbox("Sort by risk",["Highest probability first","Lowest probability first"])
    df=df[df["Risk category"].isin(bands)].sort_values("Predicted churn probability",ascending=order.startswith("Lowest"))
    if _empty(df): return
    cols=st.columns(3)
    for col,band in zip(cols,["High","Medium","Low"]): col.metric(f"{band} risk in current view",f"{int((df['Risk category']==band).sum()):,}")
    fig=px.histogram(df,x="Predicted churn probability",color="Risk category",color_discrete_map=COLORS,nbins=30,title="Predicted risk across the selected customer view")
    fig.update_xaxes(tickformat=".0%",range=[0,1])
    chart(fig,290)
    st.caption("Probabilities express uncertainty. Risk boundaries follow the editable sidebar policy. Some historical records were used for model fitting; this view is not a performance estimate.")
    fields=["customerID","tenure","Contract","InternetService","MonthlyCharges","TechSupport","Churn","Predicted churn probability","Risk category"]
    display=df[[f for f in fields if f in df]].rename(columns={"Churn":"Actual historical churn (1=yes)"})
    st.dataframe(display,hide_index=True,width="stretch",column_config={"Predicted churn probability":st.column_config.ProgressColumn("Predicted churn probability",min_value=0,max_value=1,format="percent")})
    st.download_button("Export filtered risk records",_csv(display),"filtered_customer_risk.csv","text/csv")
    st.subheader("Customer detail & model scenario")
    selected=_customer(df,"riskcustomer")
    prob=float(selected["Predicted churn probability"].iloc[0])
    st.write(f"**Model:** {c['model']} · **Predicted probability:** {prob:.1%} · **Historical label:** {LABELS[int(selected.Churn.iloc[0])]}")
    _local_chart(c["bundle"],selected,c["model"])
    with st.expander("What if the contract or monthly price changed?"):
        st.warning("Model-based hypothetical scenario. Changing a model input does not prove that changing the customer's contract or price will reduce churn.")
        x,y=st.columns(2)
        values=["Month-to-month","One year","Two year"]
        original_contract=selected.Contract.iloc[0]
        default_contract=original_contract if original_contract in values else c["bundle"].get("training_baselines",{}).get("Contract",values[0])
        if default_contract not in values: default_contract=values[0]
        contract=x.selectbox("Hypothetical contract",values,index=values.index(default_contract))
        if original_contract not in values:
            st.caption("The original contract is missing and is imputed by the fitted pipeline. The scenario starts from the training reference contract; selecting a contract replaces the missing input.")
        original_charge=float(selected.MonthlyCharges.iloc[0])
        start_charge=original_charge if np.isfinite(original_charge) else 0.0
        monthly=y.number_input("Hypothetical monthly charges",min_value=0.0,max_value=max(1000.0,start_charge),value=start_charge,step=1.0)
        if not np.isfinite(original_charge):
            st.caption("The original monthly charge is missing and is imputed by the fitted pipeline. Entering a scenario charge replaces that missing input.")
        edited=selected.copy();edited["Contract"]=contract;edited["MonthlyCharges"]=monthly
        alternative=float(score_customers(c["bundle"],edited,c["model"])[0])
        st.metric("Scenario predicted churn",f"{alternative:.1%}",f"{(alternative-prob)*100:+.1f} percentage points",delta_color="inverse")
        st.caption("Only contract and current monthly charges change. Historical total charges and all other account attributes remain fixed. Inputs are restricted to valid contracts and non-negative charges; this is model sensitivity, with no simulated customer response.")


def risk_explorer(**c):
    page_header("Find the next conversation.","Explore predicted risk, inspect a customer record and score a new file using the fitted preprocessing pipeline.","MODEL ESTIMATES · HISTORICAL LABELS SHOWN SEPARATELY")
    if not _require_model(c["bundle"]): return
    history,batch=st.tabs(["Historical customer explorer", "Batch scoring · new records"])
    with history:
        _historical_risk_explorer(c)
    with batch:
        st.subheader("Score a CSV without a churn label")
        st.write("Use the same 19 account feature columns as the IBM dataset. customerID is optional for navigation and is excluded from predictors. The saved pipeline handles imputation and encoding.")
        with st.expander("Required feature schema"): st.code(", ".join(FEATURE_COLUMNS),language=None)
        example=c["df"].head(5).drop(columns=["Churn"],errors="ignore")
        st.download_button("Download scoring CSV template",_csv(example),"customer_scoring_template.csv","text/csv")
        upload=st.file_uploader("Upload customer scoring CSV",type=["csv"],key="batchupload")
        if upload is not None:
            try:
                incoming,quality=load_data(BytesIO(upload.getvalue()),require_target=False)
                scores=score_customers(c["bundle"],incoming,c["model"])
                scored=incoming.copy();scored["Predicted churn probability"]=scores
                scored["Risk category"]=risk_category(scores,c["low"],c["high"])
                st.success(f"Scored {len(scored):,} validated records with {c['model']}.")
                for message in quality.get("warnings",[]): st.warning(str(message))
                st.dataframe(scored.rename(columns={"Churn":"Actual supplied churn (1=yes)"}),hide_index=True,width="stretch")
                st.download_button("Download scored customers",_csv(scored),"scored_customers.csv","text/csv")
                st.caption("These are model estimates, not observed outcomes. New records can differ from the training population.")
                if len(scored):
                    selected=_customer(scored,"batchcustomer")
                    _local_chart(c["bundle"],selected,c["model"])
            except Exception as exc:
                st.error(f"The scoring file could not be processed: {exc}")


def retention_centre(**c):
    page_header("Evidence into thoughtful action.","Review transparent suggestions for a customer or segment. Every action has a rationale and a limitation.","DECISION SUPPORT · PROPOSED ACTIONS, NO OUTREACH SENT")
    df=c["filtered"]
    if _empty(df): return
    st.info("Recommendations are rules linked to observed segment patterns. Offer acceptance and causal intervention effects are unavailable in the IBM sample. Customer preferences and operational checks require further engagement.")
    if _require_model(c["bundle"]):
        selected=_customer(df,"retentioncustomer")
        probability=float(score_customers(c["bundle"],selected,c["model"])[0])
        a,b,d=st.columns(3)
        a.metric("Model churn probability",f"{probability:.1%}")
        b.metric("Risk policy band",risk_category(probability,c["low"],c["high"]))
        d.metric("Historical outcome",LABELS[int(selected.Churn.iloc[0])])
        st.caption("Scores for historical records may include fitting records. Model evaluation is reported separately on held-out test customers.")
        for i,suggestion in enumerate(customer_recommendations(selected.iloc[0],df,probability,c["low"],c["high"]),1):
            with st.container(border=True):
                st.markdown(f"**{i:02d} · {suggestion['action']}**")
                st.write(suggestion["rationale"])
                st.caption("EVIDENCE  "+suggestion["evidence"])
                st.caption("OBJECTIVE  "+suggestion["objective"])
                st.warning(suggestion["caveat"])
        with st.expander("Key model sensitivities for this customer"):
            _local_chart(c["bundle"],selected,c["model"])
    st.subheader("Segment-level retention opportunities")
    segments=segment_recommendations(df)
    st.dataframe(segments,hide_index=True,width="stretch")
    st.download_button("Download segment recommendations",_csv(segments),"segment_recommendations.csv","text/csv")
    st.caption("Sample sizes and observed churn rates are evidence of association. An absence of technical-support subscription is not evidence that the customer experienced a technical problem.")
    _report(c)


def economics(**c):
    try:
        _economics_view(**c)
    except (ValueError,OverflowError,ZeroDivisionError,FloatingPointError) as exc:
        st.error(f"The economic scenario or sensitivity range could not be calculated: {exc}. Reduce extreme assumptions and try again.")


def _economics_view(**c):
    page_header("Make the economics explicit.","Explore what a retention campaign would need to achieve under editable cost, contribution and incremental success assumptions.","HYPOTHETICAL SCENARIO · NOT OBSERVED FINANCIAL RESULTS")
    st.warning("The IBM dataset contains no campaign costs, contribution margins or treatment outcomes. Every financial input below is a hypothetical assumption. Incremental success means additional customers retained because of the intervention, excluding customers who would have stayed anyway.")
    inputs,results=st.columns([1,1.5],gap="large")
    with inputs:
        st.subheader("Campaign assumptions")
        target_mode=st.radio("Targeting basis",["Customer count","Share of current filtered view"],horizontal=False)
        if target_mode=="Customer count":
            targeted=st.number_input("Customers targeted",min_value=0,max_value=10000000,value=min(500,len(c["filtered"])),step=10)
        else:
            share=st.slider("Share of filtered customers targeted",0,100,20,1)
            targeted=int(round(len(c["filtered"])*share/100))
            st.caption(f"{targeted:,} customers from {len(c['filtered']):,} in the selected historical view.")
        contact=st.number_input("Contact / offer cost per targeted customer",min_value=0.0,max_value=1000000.0,value=15.0,step=1.0,format="%.2f")
        success=st.slider("Incremental retention success (%)",0.0,100.0,10.0,.5)/100
        contribution=st.number_input("Incremental contribution per retained customer",min_value=0.0,max_value=10000000.0,value=240.0,step=10.0,format="%.2f",help="Contribution after service costs over the stated period, before campaign costs. It is not revenue or the dataset's MonthlyCharges.")
        extra=st.number_input("Additional campaign costs",min_value=0.0,max_value=1000000000000.0,value=1000.0,step=100.0,format="%.2f")
        period=st.selectbox("Contribution period",["3 months","6 months","12 months","24 months"],index=2)
        currency=st.selectbox("Scenario currency",["USD","INR","EUR","GBP"],help="A display unit only; this control does not convert currency or infer the dataset's original monetary units.")
        try:
            scenario=economic_scenario(int(targeted),contact,success,contribution,extra,period,currency)
            if any(isinstance(value,(float,int)) and not np.isfinite(value) for value in scenario.values()):
                raise ValueError("The combination produces a non-finite return or break-even ratio. Use less extreme assumptions.")
        except (ValueError,OverflowError,ZeroDivisionError) as exc:
            st.error(f"The scenario could not be calculated: {exc}")
            st.session_state.pop("economic_scenario",None)
            return
        st.session_state["economic_scenario"]={**scenario,"reference_dataset":c["name"],"reference_data_hash":c["data_hash"],"reference_filtered_records":len(c["filtered"])}
    with results:
        st.subheader("Estimated incremental outcome")
        a,b=st.columns(2)
        a.metric("Targeted customers",f"{scenario['targeted']:,}")
        b.metric("Additional customers retained",f"{scenario['retained']:,.1f}")
        a,b=st.columns(2)
        a.metric("Total campaign cost",_money(scenario['campaign_cost'],currency),help=f"Exact assumed cost: {currency} {scenario['campaign_cost']:,.2f}")
        b.metric("Incremental contribution",_money(scenario['incremental_contribution'],currency),help=f"Exact estimated contribution: {currency} {scenario['incremental_contribution']:,.2f}")
        a,b=st.columns(2)
        a.metric("Estimated net benefit",_money(scenario['net_benefit'],currency),help=f"Exact estimated net benefit: {currency} {scenario['net_benefit']:,.2f}")
        b.metric("Estimated ROI","Undefined" if scenario["roi_pct"] is None else f"{scenario['roi_pct']:.1f}%")
        st.caption(f"Contribution and net benefit relate to {period}. Fractional retained customers are expected scenario values. M = million; B = billion; T = trillion. Hover a monetary card's help icon for its full value.")
        be=scenario["break_even_success_rate"]
        bev=scenario["break_even_value"]
        st.markdown(f"**Break-even incremental success:** {'undefined for zero targeted customers or contribution' if be is None else f'{be:.1%}'}")
        st.markdown(f"**Break-even retained-customer contribution:** {'undefined with zero incremental retention' if bev is None else f'{currency} {bev:,.2f} over {period}'}")
        if be is not None and be>1: st.info("The break-even success rate exceeds 100%. This campaign cannot break even within the current cost and contribution assumptions.")
        flows=pd.DataFrame({"Component":["Incremental contribution","Campaign cost","Net benefit"],"Amount":[scenario["incremental_contribution"],-scenario["campaign_cost"],scenario["net_benefit"]]})
        chart(px.bar(flows,x="Component",y="Amount",color="Component",title=f"Scenario contribution and cost · {currency}",color_discrete_sequence=[TEAL,CORAL,BLUE]),320)
    st.subheader("Sensitivity: where does the campaign break even?")
    a,b=st.columns(2)
    rates=np.linspace(0,min(1,max(.3,success*2)),31)
    values=np.linspace(max(0,contribution*.25),max(100,contribution*1.75),25)
    net=np.array([[economic_scenario(int(targeted),contact,float(rate),float(value),extra,period,currency)["net_benefit"] for rate in rates] for value in values])
    with a:
        fig=go.Figure(go.Heatmap(x=rates*100,y=values,z=net,colorscale="Tealrose",zmid=0,colorbar=dict(title=f"Net {currency}"),hovertemplate="Incremental success: %{x:.1f}%<br>Contribution: %{y:,.0f}<br>Net benefit: %{z:,.0f}<extra></extra>"))
        fig.update_layout(title="Incremental success × retained-customer value",xaxis_title="Incremental success (%)",yaxis_title=f"Contribution per retained customer ({currency})")
        chart(fig,390)
        st.caption("Costs are held at the entered assumptions. The heatmap varies incremental success and contribution; the centre of the colour scale is zero net benefit.")
    with b:
        lines=[]
        for multiplier in [.5,1,1.5]:
            for rate in rates:
                s=economic_scenario(int(targeted),contact*multiplier,float(rate),contribution,extra,period,currency)
                lines.append({"Incremental success (%)":rate*100,"Net benefit":s["net_benefit"],"Contact cost":f"{multiplier:.1f}× base contact cost"})
        fig=px.line(pd.DataFrame(lines),x="Incremental success (%)",y="Net benefit",color="Contact cost",title="Contact cost and success-rate sensitivity")
        fig.add_hline(y=0,line_dash="dot",line_color="#96a7b9")
        chart(fig,390)
        st.caption("Retained-customer contribution and additional costs are held fixed. Crossing the zero line means breaking even under those assumptions.")
    with st.expander("Formulas, assumptions & scenario table"):
        st.markdown("**Campaign cost** = targeted × cost per contact + additional costs.\n\n**Additional retained customers** = targeted × assumed incremental success.\n\n**Incremental contribution** = additional retained customers × contribution per retained customer.\n\n**Net benefit** = incremental contribution − campaign cost.\n\n**ROI** = net benefit ÷ campaign cost × 100%, undefined when campaign cost is zero.\n\n**Break-even success** = campaign cost ÷ (targeted × contribution). **Break-even value** = campaign cost ÷ additional retained customers, with zero-denominator results undefined.")
        st.caption("The calculation assumes common cost and contribution within the chosen target group. For segment-specific value assumptions, set the sidebar segment filters and run a separate scenario; exports retain those assumptions. No causal uplift is estimated from churn probabilities.")
        table=sensitivity_table(int(targeted),contact,success,contribution,extra,period,currency)
        st.dataframe(table,hide_index=True,width="stretch")
        st.download_button("Download sensitivity scenarios",_csv(table),"economic_sensitivity.csv","text/csv")
    st.download_button("Download scenario assumptions & results",json.dumps(scenario,indent=2),"economic_scenario.json","application/json")
    _report(c)


def methodology(**c):
    page_header("A transparent research workflow.","The data, methods and limits behind the project — and how its customer-centred journey would work in practice.","METHODOLOGY · ACADEMIC PROTOTYPE")
    st.subheader("The seven-stage customer journey")
    stages=[
        ("Customer uses services","Historical internet, phone and account records represent the subscribed customer."),
        ("Process existing data","Validate available account features. No invented telemetry or manual customer questionnaire."),
        ("Predict churn risk","A fitted pipeline estimates churn probability from available subscription information."),
        ("Classify risk","Apply an editable policy: low, medium or high; probability does not mean certainty."),
        ("Suggest retention actions","Present evidence-linked options for an analyst to review; no offer is automatically sent."),
        ("Customer responds","Conceptual future stage. The sample records contain no outcomes for these proposed offers."),
        ("Monitor and learn","Re-run uploaded data and evaluate labelled outcomes; no live automatic monitoring is claimed."),
    ]
    st.graphviz_chart('digraph journey { rankdir=LR; graph [bgcolor="transparent",pad="0.2"]; node [shape=box,style="rounded,filled",fillcolor="#e9f4f5",color="#badcde",fontname="Arial",fontsize=10]; edge [color="#6f9caf"]; n1[label="1. Use services"]; n2[label="2. Collect account data"]; n3[label="3. Predict risk"]; n4[label="4. Classify risk"]; n5[label="5. Suggest actions"]; n6[label="6. Customer response\\n(conceptual)"]; n7[label="7. Monitor & learn\\n(future labelled data)"]; n1->n2->n3->n4->n5->n6->n7; n7->n2[style=dashed,label="future evidence"]; }',width="stretch")
    for start in [0,4]:
        columns=st.columns(4 if start==0 else 3)
        for column,(i,(title,description)) in zip(columns,enumerate(stages[start:start+4],start+1)):
            with column: st.markdown(f'<div class="step"><div class="step-num">STAGE {i:02d}</div><h4>{title}</h4><p>{description}</p></div>',unsafe_allow_html=True)
    st.caption("Solid arrows show the proposed sequence; the dashed loop is a future deployment concept. This local prototype implements analysis, prediction and recommendation views, not customer outreach or observed offer response.")
    tabs=st.tabs(["Data & preparation","Models & evaluation","Decisions & economics","Limitations"])
    with tabs[0]:
        st.markdown("The project uses the **IBM Telco Customer Churn** sample. [IBM's source repository](https://github.com/IBM/telco-customer-churn-on-icp4d) describes the historical sample; results are conditional on the file loaded here. Common versions contain around 7,043 rows, but the dashboard reads actual file size and schema instead of assuming a fixed population. Customer IDs support navigation and are excluded from predictors.")
        st.markdown("**Preparation.** Required account fields are validated. Churn labels are mapped to 0 (stayed) or 1 (churned), senior status is a category, and numeric fields including whitespace in TotalCharges are parsed. Missing values, duplicates, inconsistent service combinations and unexpected fields are audited. Remaining feature missingness is imputed by the predictive preprocessing fitted on training records only. Categorical inputs are encoded with unseen-category handling; numeric features are scaled where appropriate.")
        st.markdown("**Exploratory statistics.** Churn rates use actual labels and Wilson 95% intervals. Numeric summaries report spread and missingness. Categorical associations use chi-square where expected counts permit, Fisher's test for sparse two-by-two comparisons, or withhold unsuitable p-values. Numeric comparisons report effect sizes and assumptions. False-discovery correction addresses multiple testing. Statistical logistic odds ratios provide adjusted associations under model assumptions, not causal effects.")
        _quality(c)
    with tabs[1]:
        st.markdown("**Leakage prevention.** Split customers into stratified development and final test samples using a fixed seed. Fit imputers, encoders, scalers, tuning choices and any probability calibrators using training records only. Hyperparameters, ensemble weights, the preferred model and policy choices must not use final test labels. All four models are compared on exactly the same held-out customers.")
        st.markdown("**The four predictors.** Logistic Regression is a linear baseline in log-odds space; Random Forest captures nonlinear patterns across many decision trees; XGBoost combines sequential boosted trees. The **Hybrid Ensemble is an actual soft-voting probability combination**, initially using equal non-negative weights summing to one. Its score is the weighted mean of the three individual probabilities. The hybrid can perform worse than an individual model; the dashboard reports the measured results.")
        st.markdown("**Evaluation.** Precision measures the reliability of churn flags; recall measures how many churners are found; F1 balances them. ROC-AUC measures ranking; PR-AUC provides a prevalence-sensitive view for imbalanced outcomes; Brier score and calibration curves assess probability quality. Confusion matrices show false positives and false negatives. Training or validation results determine the preferred model; test diagnostics describe generalisation on this one sample.")
        if c["bundle"] is not None:
            with st.expander("Actual fitted artifact metadata"):
                st.json({"generated_at":c["bundle"].get("generated_at"),"data_hash":c["bundle"].get("data_hash"),"preferred_model":c["bundle"].get("preferred_model"),"metadata":c["bundle"].get("metadata",{})},expanded=False)
        st.markdown("**Explanations.** Logistic coefficients show direction on the estimator's log-odds scale, while tree importances show relative use without causal direction. Optional supported SHAP summaries identify their model and output scale. A deterministic single-feature perturbation method is a reliable labelled fallback: replace one input with a training reference and measure score change. Such effects need not add to the prediction and can be sensitive to correlated features.")
    with tabs[2]:
        st.markdown(f"**Current risk policy.** Low: below **{c['low']:.0%}**; medium: **{c['low']:.0%} to below {c['high']:.0%}**; high: **{c['high']:.0%} or above**. The initial 30% and 60% boundaries are illustrative outreach priorities. They are configurable business rules, not established statistical cut-offs. Separately, a binary classifier threshold is used for performance diagnostics. A high score never establishes that a customer will leave.")
        st.markdown("**Recommendations.** Transparent rules link actual customer features and observed segment results to possible account reviews, onboarding conversations or support information. They show the evidence, objective and caveat. No missing subscription is treated as evidence of a complaint. There are no automatic discounts, offers or contacts.")
        st.markdown("**Economics.** Editable inputs describe the target count, contact cost, incremental retention uplift, contribution per additional retained customer over a named period and extra costs. Net benefit subtracts campaign cost from incremental contribution. The scenario excludes customers who would have stayed without treatment. It does not estimate causal success from predictive churn probability. A controlled pilot or randomised experiment would be needed to estimate intervention uplift and real campaign profitability.")
        st.markdown("**Future monitoring.** A deployed system should monitor feature drift, missingness, calibration, precision/recall at a fixed policy, segment performance and delayed labels. Track consented offer exposure and response separately, run controlled campaign evaluation and approve retraining on new labelled outcomes. Here, analysts can upload new data and explicitly retrain; automatic live learning is not implemented.")
    with tabs[3]:
        for item in [
            "The IBM sample is historical and observational. It may not represent a current provider, geography, pricing system or customer population.",
            "The data does not contain real-time network telemetry, internet speed, complaints, satisfaction, reasons for cancellation or outcomes of proposed retention offers.",
            "Predictive associations and explanations do not identify causal drivers. Changing a model input in the what-if view does not establish the benefit of a real action.",
            "Some subgroups are small and estimates uncertain. Multiple testing, correlated features and potential omitted variables limit inference. Consider fairness and segment-level error rates before operational use.",
            "Historical customer explorer scores may include fitting records. Only the dedicated held-out evaluation view is suitable for assessing predictive performance.",
            "Scenario costs, intervention success and contribution margins are hypothetical. Revenue is different from contribution, and predicted churn is different from incremental retention uplift.",
            "Model artifacts and caches are keyed to dataset content. Changing the loaded file requires a matching model or retraining. Only load trusted local joblib files.",
        ]: st.write("• "+item)
    _report(c)


def _report(c):
    with st.expander("Export a concise analysis report"):
        st.caption("Exports identify historical findings, model estimates and any hypothetical economics scenario saved in this session.")
        report={
            "title":"Customer Retention in Broadband and Internet Service Plans",
            "dataset":c["name"],"data_fingerprint":c["data_hash"],"fallback_sample":c["fallback"],
            "validated_records":len(c["df"]),"filtered_records":len(c["filtered"]),
            "historical_insights":executive_insights(c["filtered"]) if len(c["filtered"]) else [],
            "selected_model":c["model"],"risk_policy":{"low_below":c["low"],"high_at_or_above":c["high"]},
            "model_generated_at":c["bundle"].get("generated_at") if c["bundle"] else None,
            "held_out_metrics":c["bundle"]["metrics"].to_dict("records") if c["bundle"] else [],
            "segment_recommendations":segment_recommendations(c["filtered"]).to_dict("records") if len(c["filtered"]) else [],
            "hypothetical_economic_scenario":st.session_state.get("economic_scenario"),
            "limitations":"Historical IBM sample; associations are not causes. Scores are estimates, risk bands are illustrative and scenario assumptions are hypothetical. No live outreach or causal offer-response data.",
        }
        md=f"# {report['title']}\n\nDataset: {report['dataset']} ({report['validated_records']:,} validated records; {report['filtered_records']:,} filtered).\n\n## Historical findings\n\n"+"\n".join(f"- {i}" for i in report["historical_insights"])
        md+=f"\n\n## Model estimates\n\nSelected model: {report['selected_model'] or 'Not trained'}. Risk boundaries: {c['low']:.0%} / {c['high']:.0%}.\n\n"
        if c["bundle"] is not None: md+=c["bundle"]["metrics"].to_csv(index=False)+"\n"
        md+="\n## Hypothetical economics\n\n"+(json.dumps(report["hypothetical_economic_scenario"],indent=2) if report["hypothetical_economic_scenario"] else "No scenario configured in this session.")
        md+="\n\n## Limitations\n\n"+report["limitations"]
        a,b=st.columns(2)
        a.download_button("Download report · JSON",json.dumps(report,indent=2,default=str),"retention_analysis_report.json","application/json")
        b.download_button("Download report · Markdown",md,"retention_analysis_report.md","text/markdown")
