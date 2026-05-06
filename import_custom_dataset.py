import os
import sys
import numpy as np
try:
    from PIL import Image
except ImportError:
    print("[!] Please install Pillow to process custom images: pip install Pillow")
    sys.exit(1)

IMAGE_SIZE = 64  # High resolution for real-life images!

def build_dataset(dataset_dir):
    print(f"Scanning directory: {dataset_dir}")
    
    # Get all subdirectories (each represents a class)
    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    classes.sort()
    
    if not classes:
        print("[!] No class folders found in the directory!")
        print("    Ensure your dataset is structured like this:")
        print("    my_photos/")
        print("       cat/")
        print("       dog/")
        sys.exit(1)
        
    print(f"Found {len(classes)} classes: {classes}")
    
    # Save class names for C++ to read (optional, but let's write to a text file)
    with open("custom_classes.txt", "w") as f:
        for c in classes:
            f.write(c + "\n")
            
    images = []
    labels = []
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_dir, class_name)
        img_files = os.listdir(class_dir)
        print(f"  - Loading {len(img_files)} images from '{class_name}'...")
        
        for img_name in img_files:
            img_path = os.path.join(class_dir, img_name)
            try:
                # Open, convert to grayscale, and resize
                with Image.open(img_path) as img:
                    img = img.convert('L')
                    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    images.append(img_array.flatten())
                    labels.append(class_idx)
            except Exception as e:
                # Not an image or unreadable
                pass
                
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32).reshape(-1, 1)
    
    num_samples = len(images)
    print(f"Successfully processed {num_samples} total images.")
    
    # Split into train (80%) and test (20%)
    indices = np.random.permutation(num_samples)
    split_idx = int(num_samples * 0.8)
    
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    train_images, test_images = images[train_idx], images[test_idx]
    train_labels, test_labels = labels[train_idx], labels[test_idx]
    
    print(f"Train split: {len(train_images)} images")
    print(f"Test split:  {len(test_images)} images")
    
    print("Writing binary .mat files for GradientX...")
    train_images.tofile("custom_train_images.mat")
    train_labels.tofile("custom_train_labels.mat")
    test_images.tofile("custom_test_images.mat")
    test_labels.tofile("custom_test_labels.mat")
    
    with open("custom_meta.txt", "w") as f:
        f.write(f"{len(train_images)} {len(test_images)} {IMAGE_SIZE*IMAGE_SIZE} {len(classes)}\n")
        
    print("Done! You can now select 'Custom Dataset' in main.exe.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python import_custom_dataset.py <path_to_image_folder>")
        sys.exit(1)
    build_dataset(sys.argv[1])
