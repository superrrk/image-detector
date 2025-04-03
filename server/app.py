from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from captum.attr import Saliency
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/*": {"origins": "*"}})

# Create directories if they don't exist
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
UPLOAD_FOLDER = os.path.join(STATIC_DIR, 'uploads')
SALIENCY_FOLDER = os.path.join(STATIC_DIR, 'saliency')

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SALIENCY_FOLDER, exist_ok=True)

# Model definition
class FakeImageCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 30 * 30, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.model(x))

# Load model
try:
    model = FakeImageCNN()
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, '..', 'models', 'fake_detector.pth'), 
                                   map_location='cpu'))
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

# Image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def generate_saliency_map(img_tensor, original_img):
    """Generate saliency map"""
    try:
        saliency = Saliency(model)
        attributions = saliency.attribute(img_tensor, target=0)
        
        saliency_map = attributions.squeeze().cpu().detach().numpy()
        saliency_map = np.abs(saliency_map).mean(axis=0)
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
        
        # Convert to heatmap
        heatmap = Image.fromarray((saliency_map * 255).astype(np.uint8)).convert('RGB')
        heatmap = heatmap.resize(original_img.size, Image.Resampling.LANCZOS)
        
        # Blend with original image
        result = Image.blend(original_img, heatmap, 0.5)
        return result
    except Exception as e:
        logger.error(f"Error generating saliency map: {e}")
        return original_img  # Return original image if saliency fails

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file"}), 400
            
        file = request.files['image']
        if not file:
            return jsonify({"error": "Empty file"}), 400

        # Process image
        img = Image.open(file).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        
        # Get prediction
        with torch.no_grad():
            output = model(img_tensor)
            prob = output.item()
            
        # Adjust threshold (make it harder to classify as fake)
        threshold = 0.7
        is_fake = prob > threshold
        
        # Calculate confidence
        raw_confidence = prob if is_fake else (1 - prob)
        
        # Scale confidence to be less extreme
        scaled_confidence = (raw_confidence - 0.5) * 2
        scaled_confidence = (np.tanh(scaled_confidence) + 1) / 2
        
        label = "FAKE" if is_fake else "REAL"
        
        # Generate and save saliency map
        unique_id = uuid.uuid4().hex
        saliency_filename = f"saliency_{unique_id}.png"
        saliency_path = os.path.join(SALIENCY_FOLDER, saliency_filename)
        
        saliency_result = generate_saliency_map(img_tensor, img)
        saliency_result.save(saliency_path)
        
        response_data = {
            "label": label,
            "confidence": round(scaled_confidence, 3),
            "raw_score": round(prob, 3),
            "saliency_url": f"/static/saliency/{saliency_filename}"
        }
        
        logger.info(f"Prediction: {response_data}")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in predict: {str(e)}")
        return jsonify({"error": "Failed to process image"}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)

if __name__ == "__main__":
    logger.info("Starting server on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
