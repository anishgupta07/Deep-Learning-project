"""Integration tests for GenomeDL v3 FastAPI service."""

from fastapi.testclient import TestClient
import pytest
from app import app

client = TestClient(app)


def test_healthz_endpoint():
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "GenomeDL" in data["service"]


def test_predict_variant_brca1_pathogenic():
    payload = {
        "chrom": "17",
        "pos": 43106487,
        "ref": "A",
        "alt": "C",
        "gene": "BRCA1",
        "consequence": "missense_variant",
        "ref_aa": "C",
        "alt_aa": "G",
        "aa_pos": 61,
        "af": 0.00001,
        "genotype": "0/1"
    }
    response = client.post("/v1/predict/variant", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["gene"] == "BRCA1"
    assert data["tier"] == "Pathogenic"
    assert len(data["actionable_cancer_risks"]) > 0
    assert "disclaimer" in data


def test_predict_variant_common_benign():
    payload = {
        "chrom": "13",
        "pos": 32338780,
        "ref": "A",
        "alt": "G",
        "gene": "BRCA2",
        "consequence": "synonymous_variant",
        "af": 0.045,  # > 1% gnomAD
        "genotype": "0/1"
    }
    response = client.post("/v1/predict/variant", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["tier"] == "Benign"
