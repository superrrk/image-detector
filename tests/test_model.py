import unittest
import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import sys
sys.path.append('..')  # Add parent directory to path

from train_simple import create_model, transform

class TestImageDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cls.model = create_model().to(cls.device)
        
        # Load model if exists
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pth')
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=cls.device)
            cls.model.load_state_dict(checkpoint['model_state_dict'])
        cls.model.eval()
        
        # Set up correct paths
        cls.base_dir = os.path.join(os.path.dirname(__file__), '..')
        cls.data_dir = os.path.join(cls.base_dir, 'real_vs_fake', 'real-vs-fake')

    def test_model_output_shape(self):
        """Test if model outputs correct shape"""
        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, 224, 224).to(self.device)
        with torch.no_grad():
            output = self.model(dummy_input)
        self.assertEqual(output.shape, (batch_size, 1))

    def test_model_output_range(self):
        """Test if model outputs are between 0 and 1"""
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            output = self.model(dummy_input)
        self.assertTrue(torch.all((output >= 0) & (output <= 1)))

    def test_transform_consistency(self):
        """Test if transform produces consistent tensor shapes"""
        dummy_image = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        transformed = transform(dummy_image)
        self.assertEqual(transformed.shape, (3, 224, 224))

    def test_data_directory_structure(self):
        """Test if the data directory structure is correct"""
        print("\nChecking directory structure:")
        data_dir = self.data_dir
        print(f"Base data directory: {data_dir}")
        
        # Check if directory exists
        self.assertTrue(os.path.exists(data_dir), f"Data directory not found: {data_dir}")
        
        # List contents of data directory
        contents = os.listdir(data_dir)
        print(f"Contents of data directory: {contents}")
        
        # Check for test directory
        test_dir = os.path.join(data_dir, 'test')
        self.assertTrue(os.path.exists(test_dir), f"Test directory not found: {test_dir}")
        
        # List contents of test directory
        test_contents = os.listdir(test_dir)
        print(f"Contents of test directory: {test_contents}")

    def test_real_images(self):
        """Test model predictions on known real images"""
        test_dir = os.path.join(self.data_dir, 'test')
        real_dir = os.path.join(test_dir, 'real')
        
        if not os.path.exists(real_dir):
            self.skipTest(f"Real images directory not found: {real_dir}")
        
        real_images = [f for f in os.listdir(real_dir) if f.endswith(('.jpg', '.jpeg', '.png'))][:5]
        if not real_images:
            self.skipTest("No image files found in real directory")
        
        results = []
        for img_name in real_images:
            img_path = os.path.join(real_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(img_tensor)
            
            pred = output.item()
            results.append({
                'image': img_name,
                'prediction': pred,
                'expected': 1,
                'correct': pred >= 0.5
            })
        
        accuracy = sum(r['correct'] for r in results) / len(results)
        print("\nReal Image Test Results:")
        for r in results:
            print(f"Image: {r['image']}, Pred: {r['prediction']:.3f}, Correct: {r['correct']}")
        print(f"Accuracy on real images: {accuracy:.2%}")
        
        self.assertGreater(accuracy, 0.6, "Model performs poorly on real images")

    def test_fake_images(self):
        """Test model predictions on known fake images"""
        test_dir = os.path.join(self.data_dir, 'test')
        fake_dir = os.path.join(test_dir, 'fake')
        
        if not os.path.exists(fake_dir):
            self.skipTest(f"Fake images directory not found: {fake_dir}")
        
        fake_images = [f for f in os.listdir(fake_dir) if f.endswith(('.jpg', '.jpeg', '.png'))][:5]
        if not fake_images:
            self.skipTest("No image files found in fake directory")
        
        results = []
        for img_name in fake_images:
            img_path = os.path.join(fake_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(img_tensor)
            
            pred = output.item()
            results.append({
                'image': img_name,
                'prediction': pred,
                'expected': 0,
                'correct': pred < 0.5
            })
        
        accuracy = sum(r['correct'] for r in results) / len(results)
        print("\nFake Image Test Results:")
        for r in results:
            print(f"Image: {r['image']}, Pred: {r['prediction']:.3f}, Correct: {r['correct']}")
        print(f"Accuracy on fake images: {accuracy:.2%}")
        
        self.assertGreater(accuracy, 0.6, "Model performs poorly on fake images")

    def test_edge_cases(self):
        """Test model behavior with edge cases"""
        # Test with all-black image
        black_image = torch.zeros(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            black_output = self.model(black_image)
        
        # Test with all-white image
        white_image = torch.ones(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            white_output = self.model(white_image)
        
        # Test with random noise
        noise_image = torch.rand(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            noise_output = self.model(noise_image)
        
        print("\nEdge Case Results:")
        print(f"Black image prediction: {black_output.item():.3f}")
        print(f"White image prediction: {white_output.item():.3f}")
        print(f"Noise image prediction: {noise_output.item():.3f}")

if __name__ == '__main__':
    unittest.main(verbosity=2) 