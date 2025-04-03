import os
import shutil
from sklearn.model_selection import train_test_split
import random

def prepare_dataset(input_dir, output_dir, test_size=0.2):
    """
    Prepare the dataset from the given input directory structure.
    
    Args:
        input_dir: Path to the directory containing the test directory
        output_dir: Path where the processed dataset will be saved
        test_size: Fraction of data to use for validation
    """
    # Create directory structure
    os.makedirs(output_dir, exist_ok=True)
    for split in ['train', 'val']:
        for label in ['real', 'fake']:
            os.makedirs(os.path.join(output_dir, split, label), exist_ok=True)
    
    # Get paths for real and fake images from the test directory
    test_dir = os.path.join(input_dir, 'test')
    real_dir = os.path.join(test_dir, 'real')
    fake_dir = os.path.join(test_dir, 'fake')
    
    # Collect file paths
    real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"Found {len(real_images)} real images and {len(fake_images)} fake images")
    
    # Split datasets
    real_train, real_val = train_test_split(real_images, test_size=test_size, random_state=42)
    fake_train, fake_val = train_test_split(fake_images, test_size=test_size, random_state=42)
    
    # Copy files
    def copy_files(file_list, dest_subdir):
        for src in file_list:
            dst = os.path.join(output_dir, dest_subdir, os.path.basename(src))
            shutil.copy2(src, dst)
            
    print("Copying files...")
    copy_files(real_train, os.path.join('train', 'real'))
    copy_files(real_val, os.path.join('val', 'real'))
    copy_files(fake_train, os.path.join('train', 'fake'))
    copy_files(fake_val, os.path.join('val', 'fake'))
    print("Dataset preparation completed!")

if __name__ == "__main__":
    prepare_dataset(
        input_dir='real_vs_fake/real-vs-fake',
        output_dir='data'
    ) 