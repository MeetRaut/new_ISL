import os
import random
import shutil

# Define your data directory
data_dir = 'Data'

# Define your output directories for train and test sets
train_dir = 'Split/Train'
test_dir = 'Split/Test'

# Define the percentage of data you want to allocate for training
train_ratio = 0.8

# Iterate over each class directory
for class_label in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_label)
    
    # Create train and test directories for each class
    train_class_dir = os.path.join(train_dir, class_label)
    test_class_dir = os.path.join(test_dir, class_label)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)
    
    # Get a list of all image files for this class
    image_files = os.listdir(class_dir)
    
    # Shuffle the list of image files
    random.shuffle(image_files)
    
    # Split the image files into train and test sets
    num_train_samples = int(train_ratio * len(image_files))
    train_files = image_files[:num_train_samples]
    test_files = image_files[num_train_samples:]
    
    # Move train files to the train directory
    for train_file in train_files:
        src = os.path.join(class_dir, train_file)
        dest = os.path.join(train_class_dir, train_file)
        shutil.copyfile(src, dest)
    
    # Move test files to the test directory
    for test_file in test_files:
        src = os.path.join(class_dir, test_file)
        dest = os.path.join(test_class_dir, test_file)
        shutil.copyfile(src, dest)
