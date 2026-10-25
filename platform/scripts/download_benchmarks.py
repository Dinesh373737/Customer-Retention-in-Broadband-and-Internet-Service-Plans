"""Download the named public benchmark sources; never substitute synthetic data."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import urllib.error
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "iranian": {
        "page": "https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset",
        "download": "https://archive.ics.uci.edu/static/public/563/iranian+churn+dataset.zip",
        "license": "CC BY 4.0; UCI Iranian Churn (2020), DOI 10.24432/C5JW3Z",
    },
    "banking": {
        "page": "https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction",
        "download": "https://www.kaggle.com/api/v1/datasets/download/shantanudhakadd/bank-customer-churn-prediction",
        "metadata": "https://www.kaggle.com/api/v1/datasets/view/shantanudhakadd/bank-customer-churn-prediction",
        "license": "Consult the dataset card; downloads remain local and are excluded from Git.",
    },
    "ecommerce": {
        "page": "https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction",
        "download": "https://www.kaggle.com/api/v1/datasets/download/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction",
        "metadata": "https://www.kaggle.com/api/v1/datasets/view/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction",
        "license": "Consult the dataset card; downloads remain local and are excluded from Git.",
    },
}


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "RetentionLab-Academic-Benchmark/2.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read(50 * 1024 * 1024 + 1)
    if len(body) > 50 * 1024 * 1024:
        raise ValueError("Download exceeded the 50 MB benchmark limit.")
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(SOURCES))
    args = parser.parse_args()
    records = []
    for key, source in SOURCES.items():
        if args.dataset and key != args.dataset:
            continue
        destination = ROOT / "benchmarks/data" / key
        destination.mkdir(parents=True, exist_ok=True)
        record = {"dataset": key, "source": source["page"], "license_note": source["license"]}
        try:
            body = fetch(source["download"])
            if not zipfile.is_zipfile(io.BytesIO(body)):
                raise ValueError("Source returned a non-ZIP response; authentication or manual download may be required.")
            files = []
            with zipfile.ZipFile(io.BytesIO(body)) as archive:
                if sum(item.file_size for item in archive.infolist()) > 100 * 1024 * 1024:
                    raise ValueError("Benchmark archive exceeds the uncompressed size limit.")
                for item in archive.infolist():
                    if Path(item.filename).suffix.lower() not in {".csv", ".xlsx"}:
                        continue
                    filename = Path(item.filename).name
                    payload = archive.read(item)
                    (destination / filename).write_bytes(payload)
                    files.append({"filename": filename, "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
            if not files:
                raise ValueError("Archive contains no supported CSV/Excel dataset files.")
            record.update(status="downloaded", files=files, archive_sha256=hashlib.sha256(body).hexdigest())
            if source.get("metadata"):
                try:
                    metadata = json.loads(fetch(source["metadata"]))
                    record["license_metadata"] = metadata.get("licenses", metadata.get("licenseName", metadata.get("license")))
                except (ValueError, urllib.error.URLError) as error:
                    record["metadata_note"] = str(error)
        except (ValueError, urllib.error.URLError, zipfile.BadZipFile) as error:
            record.update(status="unavailable", error=str(error))
        (destination / "provenance.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        records.append(record)
        print(json.dumps(record), flush=True)
    (ROOT / "benchmarks/data/acquisition.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
