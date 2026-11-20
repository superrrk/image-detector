import io
from PIL import Image
import pytest
from server.app import create_app, ModelUnavailable


class StubPredictor:
    def predict(self, image):
        return {"label": "Real-like", "fake_score": 0.2, "decision_threshold": 0.5,
                "model": "test-stub", "saliency_data_url": "", "disclaimer": "Test only"}


@pytest.fixture
def client():
    return create_app(StubPredictor()).test_client()


def image_bytes():
    output = io.BytesIO()
    Image.new("RGB", (12, 12), "red").save(output, format="PNG")
    output.seek(0)
    return output


def test_index(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Image Forensics Lab" in response.data


def test_valid_image(client):
    response = client.post("/api/analyze", data={"image": (image_bytes(), "face.png")})
    assert response.status_code == 200
    assert response.json["prediction"]["fake_score"] == 0.2
    assert response.json["file"]["format"] == "PNG"
    assert "metrics" in response.json


def test_missing_image(client):
    response = client.post("/api/analyze")
    assert response.status_code == 400


def test_invalid_image(client):
    response = client.post("/api/analyze", data={"image": (io.BytesIO(b"not a photo"), "fake.png")})
    assert response.status_code == 415


def test_oversized_image(client):
    response = client.post("/api/analyze", data={"image": (io.BytesIO(b"a" * (8 * 1024 * 1024 + 1)), "x.png")})
    assert response.status_code == 413


def test_model_unavailable():
    class Missing:
        def predict(self, image):
            raise ModelUnavailable("Weights missing")
    client = create_app(Missing()).test_client()
    response = client.post("/api/analyze", data={"image": (image_bytes(), "face.png")})
    assert response.status_code == 200
    assert response.json["prediction"] is None
    assert response.json["model_status"] == "not_installed"
