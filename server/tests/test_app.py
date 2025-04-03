import pytest
import os
import io
from app import app
from PIL import Image
import numpy as np

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def create_test_image():
    """Create a simple test image"""
    img = Image.new('RGB', (128, 128), color='red')
    img_io = io.BytesIO()
    img.save(img_io, 'JPEG')
    img_io.seek(0)
    return img_io

def test_predict_endpoint_no_file(client):
    """Test predict endpoint with no file"""
    response = client.post('/predict')
    assert response.status_code == 400
    assert b'No image file' in response.data

def test_predict_endpoint_with_file(client):
    """Test predict endpoint with valid file"""
    data = {'image': (create_test_image(), 'test.jpg')}
    response = client.post('/predict', 
                         content_type='multipart/form-data',
                         data=data)
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'label' in json_data
    assert 'confidence' in json_data
    assert 'saliency_url' in json_data

def test_static_file_serving(client):
    """Test static file serving"""
    # First upload an image
    data = {'image': (create_test_image(), 'test.jpg')}
    response = client.post('/predict', 
                         content_type='multipart/form-data',
                         data=data)
    json_data = response.get_json()
    
    # Then try to access the saliency map
    saliency_url = json_data['saliency_url']
    response = client.get(saliency_url)
    assert response.status_code == 200
