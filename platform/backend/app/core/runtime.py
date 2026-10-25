"""Dataset/model access isolated by immutable analysis version."""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import copy
import logging
import threading
from pathlib import Path

from ..persistence.store import Store, new_id, utcnow
from ..services.data_engine import read_table, frame_fingerprint, configuration_fingerprint

logger = logging.getLogger(__name__)


class Runtime:
    def __init__(self, home):
        self.store = Store(Path(home))
        self.home = self.store.home
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="retention-training")
        self.lock = threading.RLock()
        self.frames = OrderedDict()
        self.bundles = OrderedDict()
        self.futures = {}

    def frame(self, dataset):
        key = dataset["id"]
        path = (self.home / dataset["storage_path"]).resolve()
        if not path.is_relative_to(self.home) or not path.is_file():
            raise ValueError("The saved dataset file is missing or outside platform storage.")
        stamp = path.stat()
        signature = (str(path), stamp.st_mtime_ns, stamp.st_size, dataset["fingerprint"])
        with self.lock:
            cached = self.frames.get(key)
            if cached is not None and cached[0] == signature:
                self.frames.move_to_end(key)
                return cached[1].copy()
        frame = read_table(path.read_bytes(), dataset["filename"], dataset.get("sheet_name"))
        if frame_fingerprint(frame) != dataset["fingerprint"]:
            raise ValueError("The stored dataset contents changed. Upload a new dataset version rather than reusing this analysis.")
        with self.lock:
            self.frames[key] = (signature, frame)
            while len(self.frames) > 3:
                self.frames.popitem(last=False)
        return frame.copy()

    def bundle(self, analysis):
        from ..services.modeling import load_bundle
        if analysis.get("status") != "completed" or not analysis.get("artifact_path"):
            raise ValueError("This analysis has no completed fitted model. Start training or inspect the job's real failure message.")
        identifier = analysis["id"]
        path = (self.home / analysis["artifact_path"]).resolve()
        if not path.is_relative_to(self.home) or not path.is_file():
            raise ValueError("The fitted artifact is unavailable; retrain this dataset.")
        stamp = path.stat()
        signature = (str(path), stamp.st_mtime_ns, stamp.st_size, analysis["dataset_fingerprint"], analysis["configuration_fingerprint"])
        with self.lock:
            cached = self.bundles.get(identifier)
            if cached is not None and cached[0] == signature:
                self.bundles.move_to_end(identifier)
                return cached[1]
        try:
            bundle = load_bundle(path)
        except Exception as error:
            logger.exception("Could not read the locally generated model artifact for analysis %s", identifier)
            raise ValueError(f"The saved model could not be loaded ({type(error).__name__}); retrain this dataset. The server log contains the diagnostic.") from error
        if bundle["dataset_fingerprint"] != analysis["dataset_fingerprint"] or bundle["configuration_fingerprint"] != analysis["configuration_fingerprint"]:
            raise ValueError("The saved model does not match this dataset and schema version.")
        if bundle.get("analysis_id") != identifier:
            raise ValueError("The saved model belongs to a different analysis version. Retrain or select the matching analysis.")
        with self.lock:
            self.bundles[identifier] = (signature, bundle)
            while len(self.bundles) > 4:
                self.bundles.popitem(last=False)
        return bundle

    def create_analysis(self, dataset, trained=False):
        config = copy.deepcopy(dataset["configuration"])
        identifier = new_id()
        mode = "historical" if config.get("target_column") else "exploratory"
        return self.store.put("analysis", identifier, {
            "id": identifier, "analysis_id": identifier, "dataset_id": dataset["id"],
            "filename": dataset["filename"], "configuration": config,
            "dataset_fingerprint": dataset["fingerprint"],
            "configuration_fingerprint": configuration_fingerprint(config),
            "mode": mode, "operating_mode": "A" if mode == "historical" else "C",
            "status": "queued" if trained else "ready", "model_version": None,
            "artifact_path": None, "job_id": None, "error": None,
        })

    def submit(self, analysis, settings):
        with self.lock:
            self.futures = {key: future for key, future in self.futures.items() if not future.done()}
            if len(self.futures) >= 5:
                raise ValueError("Five training jobs are already queued/running. Wait for completion before adding another.")
            job_id = new_id()
            job = {"id": job_id, "analysis_id": analysis["id"], "dataset_id": analysis["dataset_id"],
                   "status": "queued", "progress": 0, "message": "Queued for the local training worker", "error": None,
                   "settings": settings}
            self.store.put("job", job_id, job)
            self.store.update("analysis", analysis["id"], job_id=job_id)
            future = self.executor.submit(self._train, analysis["id"], job_id, settings)
            self.futures[job_id] = future
        return job

    def _train(self, analysis_id, job_id, settings):
        from ..services.modeling import train_dataset, save_bundle
        try:
            self.store.update("job", job_id, status="running", progress=1, message="Validating the immutable dataset and schema snapshot")
            self.store.update("analysis", analysis_id, status="running")
            analysis = self.store.get("analysis", analysis_id)
            dataset = self.store.get("dataset", analysis["dataset_id"])
            frame = self.frame(dataset)
            def progress(percent, message):
                self.store.update("job", job_id, progress=max(1, min(99, int(percent))), message=str(message))
            bundle = train_dataset(frame, analysis["configuration"], settings=settings, progress=progress)
            if bundle["dataset_fingerprint"] != analysis["dataset_fingerprint"] or bundle["configuration_fingerprint"] != analysis["configuration_fingerprint"]:
                raise ValueError("Training returned an incompatible dataset/schema fingerprint; artifact was not accepted.")
            bundle["analysis_id"] = analysis_id
            artifact_path = Path("models") / f"{analysis_id}.joblib"
            save_bundle(bundle, self.home / artifact_path)
            self.store.update("analysis", analysis_id, status="completed", artifact_path=str(artifact_path),
                              model_version=analysis_id, preferred_model=bundle["preferred_model"], completed_at=utcnow())
            self.store.update("job", job_id, status="completed", progress=100, message="All four fitted models evaluated on the same held-out records", error=None)
        except Exception as error:
            logger.exception("Training failed for analysis %s", analysis_id)
            self.store.update("job", job_id, status="failed", message="Training failed; review the diagnostic", error=f"{type(error).__name__}: {error}")
            self.store.update("analysis", analysis_id, status="failed", error=f"{type(error).__name__}: {error}")

    def shutdown(self):
        self.executor.shutdown(wait=True, cancel_futures=False)
