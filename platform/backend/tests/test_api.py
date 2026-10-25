"""Genuine in-process HTTP workflows with fitted generic models and durable storage."""
from __future__ import annotations

import io
import json
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.persistence.store import new_id


def data_fixture(rows=180):
    rng=np.random.default_rng(94)
    fees=rng.uniform(15,95,rows)
    activity=rng.choice(["Regular","Occasional","New"],rows)
    chance=np.where(activity=="Occasional",.65,.25)
    labels=np.where(rng.random(rows)<chance,"left","stayed")
    return pd.DataFrame({"Ref":[f"C-{index:04d}" for index in range(rows)],"Fee":fees,"Activity":activity,"Result":labels})


def configuration(labelled=True):
    return {"target_column":"Result" if labelled else None,"positive_label":"left" if labelled else None,
            "customer_id":"Ref","excluded_columns":[],"features":[{"column":"Fee","kind":"numeric"},{"column":"Activity","kind":"categorical"}],
            "business_domain":"subscription","prediction_horizon":"Next renewal; synthetic test fixture only",
            "semantics":[{"column":"Fee","concept":"pricing","meaning":"Current subscription price","confirmed":True,"actionable":True,"protected":False},
                         {"column":"Activity","concept":"engagement","meaning":"Recorded account activity category","confirmed":True,"actionable":True,"protected":False}]}


def upload(client,df):
    result=client.post("/api/v1/datasets/upload",files={"file":("customers.csv",df.to_csv(index=False).encode(),"text/csv")})
    assert result.status_code==201,result.text
    return result.json()


def wait_job(client,identifier):
    deadline=time.monotonic()+90
    observed=[]
    while time.monotonic()<deadline:
        response=client.get(f"/api/v1/jobs/{identifier}")
        assert response.status_code==200,response.text
        job=response.json()
        observed.append(job["status"])
        if job["status"] in {"completed","failed"}:
            return job,observed
        time.sleep(.05)
    pytest.fail("The actual background training job did not finish within 90 seconds.")


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    home=tmp_path_factory.mktemp("platform-api")
    app=create_app(home)
    with TestClient(app) as client:
        dataset=upload(client,data_fixture())
        saved=client.post(f"/api/v1/datasets/{dataset['id']}/configuration",json=configuration())
        assert saved.status_code==200,saved.text
        started=time.monotonic()
        response=client.post("/api/v1/analyses/train",json={"dataset_id":dataset["id"],"fast":True})
        assert response.status_code==202,response.text
        assert time.monotonic()-started<5,"Training must return a job instead of blocking for model fitting."
        result=response.json()
        job,observed=wait_job(client,result["job_id"])
        assert job["status"]=="completed",job
        assert job["progress"]==100
        yield client,result["analysis_id"],dataset,app,home


def test_full_train_evaluate_and_matching_version(trained):
    client,analysis_id,dataset,_,_=trained
    result=client.get(f"/api/v1/analyses/{analysis_id}/models")
    assert result.status_code==200,result.text
    report=result.json()
    assert report["dataset_id"]==dataset["id"]
    assert report["model_version"]==analysis_id
    assert len(report["metrics"])==4
    assert set(report["models"])=={"Logistic Regression","Random Forest","XGBoost","Hybrid Ensemble"}
    assert report["preferred_model"] in report["models"]
    assert all(0<=row["roc_auc"]<=1 for row in report["metrics"])
    assert all(row["calibration"] for row in report["curves"].values())


def test_customers_paging_filters_details_and_explanations(trained):
    client,analysis_id,_,_,_=trained
    result=client.get(f"/api/v1/analyses/{analysis_id}/customers",params={"limit":5,"filters":json.dumps({"Activity":["Regular"]})})
    assert result.status_code==200,result.text
    page=result.json()
    assert len(page["rows"])<=5
    assert all(row["Activity"]=="Regular" for row in page["rows"])
    assert all(0<=row["churn_probability"]<=1 for row in page["rows"])
    detail=client.get(f"/api/v1/analyses/{analysis_id}/customers/{page['rows'][0]['row_index']}").json()
    assert detail["explanation"]["drivers"]
    assert detail["customer"]["actual_outcome"] in {"left","stayed"}
    assert client.get(f"/api/v1/analyses/{analysis_id}/customers/999999").status_code==404
    assert client.get(f"/api/v1/analyses/{analysis_id}/customers?model=not-this-model").status_code==422


def test_immutable_analysis_schema_survives_dataset_reconfiguration(trained):
    client,analysis_id,dataset,_,_=trained
    changed=configuration()
    changed["business_domain"]="banking"
    assert client.post(f"/api/v1/datasets/{dataset['id']}/configuration",json=changed).status_code==200
    overview=client.get(f"/api/v1/analyses/{analysis_id}/overview").json()
    assert overview["configuration"]["business_domain"]=="subscription"
    listing=client.get("/api/v1/analyses").json()["analyses"]
    assert next(item for item in listing if item["id"]==analysis_id)["configuration"]["business_domain"]=="subscription"


def test_batch_scoring_without_labels_and_unseen_categories_preserves_rows(trained):
    client,analysis_id,_,_,_=trained
    incoming=data_fixture(6).drop(columns=["Result","Ref"])
    incoming.loc[0,"Activity"]="Never seen before"
    incoming=pd.concat([incoming,incoming.iloc[[0]]],ignore_index=True)
    result=client.post(f"/api/v1/analyses/{analysis_id}/batch-score",files={"file":("batch.csv",incoming.to_csv(index=False).encode(),"text/csv")})
    assert result.status_code==200,result.text
    scored=result.json()
    assert scored["operating_mode"]=="B"
    assert scored["total"]==7
    assert len(scored["rows"])==7
    assert scored["warnings"]
    assert scored["explanation_limit"]==7
    assert set(scored["explanations"])=={str(index) for index in range(7)}
    assert scored["explanations"]["0"]["probability"]==pytest.approx(scored["rows"][0]["churn_probability"])
    assert all("actual_outcome" not in row for row in scored["rows"])
    broken=incoming.drop(columns="Fee")
    result=client.post(f"/api/v1/analyses/{analysis_id}/batch-score",files={"file":("broken.csv",broken.to_csv(index=False).encode(),"text/csv")})
    assert result.status_code==422
    assert "Fee" in result.json()["detail"]


def test_actionable_what_if_and_invalid_changes(trained):
    client,analysis_id,_,_,_=trained
    result=client.post(f"/api/v1/analyses/{analysis_id}/what-if",json={"row_index":0,"changes":{"Fee":50}})
    assert result.status_code==200,result.text
    assert 0<=result.json()["scenario_probability"]<=1
    assert result.json()["caveats"]
    for changes in [{"Ref":"new"},{"Result":"stayed"},{"Age":22},{"Fee":1000000},{"Activity":"unknown"}]:
        assert client.post(f"/api/v1/analyses/{analysis_id}/what-if",json={"row_index":0,"changes":changes}).status_code==422


def test_recommendations_economics_and_valid_downloads(trained):
    client,analysis_id,_,_,_=trained
    recommendations=client.get(f"/api/v1/analyses/{analysis_id}/recommendations")
    assert recommendations.status_code==200,recommendations.text
    assert recommendations.json()["limitations"]
    economics=client.post(f"/api/v1/analyses/{analysis_id}/economics",json={"targeted":100,"budget":350,"intervention_cost":10,"incremental_impact":.1,"contribution":200,"fixed_cost":100})
    assert economics.status_code==200,economics.text
    assert economics.json()["scenario"]["targeted"]==25
    assert economics.json()["scenario"]["incremental_retained"]==2.5
    assert economics.json()["ranking"][0]["expected_net_value"]==10
    for kind in ["customers","metrics","recommendations","report"]:
        download=client.get(f"/api/v1/analyses/{analysis_id}/export",params={"kind":kind})
        assert download.status_code==200,download.text
        assert "attachment" in download.headers["content-disposition"]
        if kind=="report":
            assert download.json()["hypothetical_economics"]
        else:
            pd.read_csv(io.BytesIO(download.content))


def test_exploratory_mode_never_invents_predictions(tmp_path):
    app=create_app(tmp_path)
    with TestClient(app) as client:
        dataset=upload(client,data_fixture().drop(columns="Result"))
        assert client.post(f"/api/v1/datasets/{dataset['id']}/configuration",json=configuration(False)).status_code==200
        result=client.post("/api/v1/analyses",json={"dataset_id":dataset["id"]})
        assert result.status_code==201,result.text
        analysis=result.json()["analysis_id"]
        assert result.json()["operating_mode"]=="C"
        assert client.post("/api/v1/analyses/train",json={"dataset_id":dataset["id"]}).status_code==409
        assert client.get(f"/api/v1/analyses/{analysis}/models").status_code==409
        customers=client.get(f"/api/v1/analyses/{analysis}/customers").json()
        assert all(row["churn_probability"] is None for row in customers["rows"])
        assert all(row["risk_category"]=="Unscored" for row in customers["rows"])
        report=client.get(f"/api/v1/analyses/{analysis}/export?kind=report").json()
        assert report["models"] is None


def test_invalid_files_schema_and_ids_are_actionable(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        for filename,contents in [("empty.csv",b""),("program.py",b"print(1)"),("corrupt.xlsx",b"not a workbook"),("duplicate.csv",b"A,A\n1,2\n")]:
            result=client.post("/api/v1/datasets/upload",files={"file":(filename,contents)})
            assert result.status_code==422,result.text
        assert client.get("/api/v1/datasets/missing/profile").status_code==404
        assert client.get("/api/v1/analyses/missing/customers").status_code==404
        assert client.get("/api/v1/not-a-route").status_code==404
        dataset=upload(client,data_fixture())
        bad=configuration()
        bad["features"].append({"column":"Ref","kind":"categorical"})
        assert client.post(f"/api/v1/datasets/{dataset['id']}/configuration",json=bad).status_code==422
        assert client.post("/api/v1/analyses/train",json={"dataset_id":dataset["id"]}).status_code==409


def test_excel_upload_and_sheet_selection(tmp_path):
    import openpyxl
    book=openpyxl.Workbook()
    book.active.title="Description"
    book.active.append(["Notes"])
    book.active.append(["Synthetic fixture metadata"])
    sheet=book.create_sheet("Customers")
    sheet.append(["id","value"])
    sheet.append(["one",10])
    binary=io.BytesIO()
    book.save(binary)
    with TestClient(create_app(tmp_path)) as client:
        result=client.post("/api/v1/datasets/upload",files={"file":("sample.xlsx",binary.getvalue())},data={"sheet_name":"Customers"})
        assert result.status_code==201,result.text
        assert result.json()["profile"]["rows"]==1
        assert result.json()["profile"]["worksheets"]==["Description","Customers"]


def test_real_training_failure_is_persisted_not_reported_as_success(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        small=data_fixture(30)
        dataset=upload(client,small)
        configured=client.post(f"/api/v1/datasets/{dataset['id']}/configuration",json=configuration())
        assert configured.status_code==200,configured.text
        started=client.post("/api/v1/analyses/train",json={"dataset_id":dataset["id"],"fast":True})
        assert started.status_code==202,started.text
        job,_=wait_job(client,started.json()["job_id"])
        assert job["status"]=="failed"
        assert job["error"]
        assert client.get(f"/api/v1/analyses/{started.json()['analysis_id']}/models").status_code==409


def test_sqlite_and_model_version_survive_app_restart(trained):
    _,analysis_id,dataset,_,home=trained
    with TestClient(create_app(home)) as client:
        assert client.get(f"/api/v1/datasets/{dataset['id']}/profile").status_code==200
        assert client.get(f"/api/v1/analyses/{analysis_id}/models").status_code==200


def test_interrupted_jobs_marked_failed_on_restart(tmp_path):
    app=create_app(tmp_path)
    job_id,analysis_id=new_id(),new_id()
    app.state.runtime.store.put("analysis",analysis_id,{"id":analysis_id,"status":"running"})
    app.state.runtime.store.put("job",job_id,{"id":job_id,"analysis_id":analysis_id,"dataset_id":"test","status":"running","progress":15,"message":"fitting","error":None})
    with TestClient(app) as client:
        job=client.get(f"/api/v1/jobs/{job_id}").json()
        assert job["status"]=="failed"
        assert "interrupted" in job["error"]


def test_empty_recommendation_download_is_still_valid_csv(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        dataset=upload(client,data_fixture().drop(columns="Result"))
        assert client.post(f"/api/v1/datasets/{dataset['id']}/configuration",json=configuration(False)).status_code==200
        analysis=client.post("/api/v1/analyses",json={"dataset_id":dataset["id"]}).json()["analysis_id"]
        download=client.get(f"/api/v1/analyses/{analysis}/export?kind=recommendations")
        assert download.status_code==200
        table=pd.read_csv(io.BytesIO(download.content))
        assert table.empty
        assert {"action","evidence_status","sample_size"} <= set(table.columns)
        for suffix in ["recommendations?model=unavailable", "export?kind=report&model=unavailable"]:
            assert client.get(f"/api/v1/analyses/{analysis}/{suffix}").status_code==409


def test_export_population_and_economics_honor_filters_without_changing_metrics(trained):
    client,analysis_id,_,_,_=trained
    filters={"Activity":["Regular"]}
    original=data_fixture()
    selected=original.loc[original.Activity.eq("Regular")]
    before=client.get(f"/api/v1/analyses/{analysis_id}/models").json()["metrics"]
    report=client.get(f"/api/v1/analyses/{analysis_id}/export",params={"kind":"report","filters":json.dumps(filters)}).json()
    assert report["population"]["selected_rows"]==len(selected)
    assert report["overview"]["rows"]==len(selected)
    for record in report["recommendations"]["segments"]:
        assert record["overall_churn_rate"]==pytest.approx(selected.Result.eq("left").mean())
        assert record["sample_size"]<=len(selected)
    result=client.post(f"/api/v1/analyses/{analysis_id}/economics",params={"filters":json.dumps(filters)},json={"targeted":20})
    assert result.status_code==200,result.text
    body=result.json()
    assert body["population"]["available_customers"]==len(selected)
    expected_refs=set(selected.Ref)
    assert all(row["customer_reference"] in expected_refs for row in body["ranking"])
    after=client.get(f"/api/v1/analyses/{analysis_id}/models",params={"filters":json.dumps(filters)}).json()["metrics"]
    assert before==after


def test_disk_mutation_cannot_silently_reuse_cached_dataset(tmp_path):
    application=create_app(tmp_path)
    with TestClient(application) as client:
        dataset=upload(client,data_fixture())
        rt=application.state.runtime
        stored=rt.store.get("dataset",dataset["id"])
        assert len(rt.frame(stored))==180
        (rt.home/stored["storage_path"]).write_bytes(data_fixture(20).to_csv(index=False).encode())
        with pytest.raises(ValueError,match="contents changed"):
            rt.frame(stored)


def test_artifact_from_different_analysis_is_rejected_even_with_matching_schema(trained):
    _,analysis_id,_,application,_=trained
    rt=application.state.runtime
    original=rt.store.get("analysis",analysis_id)
    different=dict(original,id=new_id())
    with pytest.raises(ValueError,match="different analysis version"):
        rt.bundle(different)


def test_numeric_categorical_scenario_options_use_saved_label_canonicalization():
    from app.api.routes import actionable_features
    frame=pd.DataFrame({"Choice":[1.0,2.0,np.nan]})
    config={"features":[{"column":"Choice","kind":"categorical"}],
            "semantics":[{"column":"Choice","concept":"contract","meaning":"Plan code", "confirmed":True,"actionable":True,"protected":False}]}
    assert actionable_features(frame,config)[0]["categories"]==["1","2"]


def test_openapi_documents_typed_payloads_and_versioned_routes(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        schema=client.get("/openapi.json").json()
        routes=schema["paths"]
        assert routes["/api/v1/datasets/upload"]["post"]["responses"]["201"]["content"]["application/json"]["schema"]["$ref"].endswith("DatasetPayload")
        assert routes["/api/v1/analyses/{analysis_id}/models"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith("ModelPayload")
        assert routes["/api/v1/analyses/{analysis_id}/batch-score"]["post"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith("BatchPayload")


def test_recommendations_use_the_requested_models_actual_global_importance(trained):
    client,analysis_id,_,_,_=trained
    response=client.get(f"/api/v1/analyses/{analysis_id}/recommendations",params={"model":"Logistic Regression"})
    assert response.status_code==200,response.text
    segments=response.json()["segments"]
    assert segments
    assert all(record["global_model_evidence"]["model"]=="Logistic Regression" for record in segments)
    assert all(record["segment_mean_probability"] is not None for record in segments)


def test_corrupted_artifact_is_an_explicit_not_ready_error(trained):
    client,analysis_id,_,application,_=trained
    rt=application.state.runtime
    identifier=new_id()
    original=rt.store.get("analysis",analysis_id)
    record={**original,"id":identifier,"analysis_id":identifier,"artifact_path":f"models/{identifier}.joblib"}
    (rt.home/record["artifact_path"]).write_bytes(b"broken artifact")
    rt.store.put("analysis",identifier,record)
    response=client.get(f"/api/v1/analyses/{identifier}/models")
    assert response.status_code==409
    assert "could not be loaded" in response.json()["detail"]


def test_training_queue_remains_bounded_under_concurrent_submissions(tmp_path):
    from concurrent.futures import Future,ThreadPoolExecutor
    from app.core.runtime import Runtime
    class PendingExecutor:
        def submit(self,*args):
            return Future()
        def shutdown(self,**kwargs):
            pass
    rt=Runtime(tmp_path)
    rt.executor.shutdown(wait=True)
    rt.executor=PendingExecutor()
    dataset={"id":"queue-fixture","filename":"synthetic.csv","fingerprint":"fixture",
             "configuration":{"target_column":"Outcome","features":[]}}
    def submit(_):
        analysis=rt.create_analysis(dataset,trained=True)
        try:
            rt.submit(analysis,{"fast":True})
            return True
        except ValueError:
            return False
    with ThreadPoolExecutor(max_workers=10) as pool:
        accepted=list(pool.map(submit,range(10)))
    assert sum(accepted)==5
    assert len(rt.futures)==5
    rt.shutdown()


def test_economic_ranking_uses_explicit_selected_model_in_body(trained):
    client,analysis_id,_,_,_=trained
    chosen="Logistic Regression"
    expected=client.get(f"/api/v1/analyses/{analysis_id}/customers",params={"model":chosen,"limit":1}).json()["rows"][0]
    response=client.post(f"/api/v1/analyses/{analysis_id}/economics",json={"model":chosen,"targeted":10})
    assert response.status_code==200,response.text
    result=response.json()
    assert result["selected_model"]==chosen
    assert result["scenario"]["selected_model"]==chosen
    assert result["ranking"][0]["row_index"]==expected["row_index"]
    assert result["ranking"][0]["churn_probability"]==pytest.approx(expected["churn_probability"])
    bad=client.post(f"/api/v1/analyses/{analysis_id}/economics",json={"model":"not-a-saved-model"})
    assert bad.status_code==422
    conflicting=client.post(f"/api/v1/analyses/{analysis_id}/economics",params={"model":"XGBoost"},json={"model":chosen})
    assert conflicting.status_code==422
