import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import TensorDataset


def load_data(sample_size):
    if sample_size == -1:
        df = pd.read_csv('../data/labels.csv')
    else:
        df = pd.read_csv('../data/labels.csv').sample(sample_size)

    # Get image file paths and labels
    image_paths = df['image'].values
    labels = df['label'].values
    num_classes = 21  # We specify the max number of classes in the dataset, regardless of the initial subsampling
    print(f'Loaded {len(image_paths)} images, with {num_classes} different classes.')

    # Load the mapping between integer labels and class names
    label_mapping = {}
    with open('../data/label_mapping.txt', 'r') as file:
        for line in file:
            value, key = line.strip().split(': ')
            label_mapping[int(key)] = value

    return image_paths, labels, num_classes, label_mapping


def load_images(file_paths, transform, folder_path='../data/Images/'):
    images = []
    for file_path in tqdm(file_paths, desc='Loading images'):
        # Load the image
        with Image.open(folder_path + file_path) as img:
            # Convert image to RGB if it's not and apply the same basic transformations
            img = img.convert('RGB')
            img = transform(img)
            images.append(img)
    return torch.stack(images)


def calculate_mean_std(stacked_images):
    # Mean and std are calculated across the height and width dimensions (2 and 3)
    mean = stacked_images.view(stacked_images.size(0), stacked_images.size(1), -1).mean(dim=2).mean(dim=0)
    std = stacked_images.view(stacked_images.size(0), stacked_images.size(1), -1).std(dim=2).mean(dim=0)
    return mean, std

def transform_data(train_paths, train_labels, val_paths, val_labels, test_paths, test_labels):
    # Basic image transformations to load the training dataset
    basic_transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),  # 224x224 is a common "historical" format for benchmarking CNNs
        transforms.ToTensor()
    ])

    # Load images for the mean/std calculation
    train_images = load_images(train_paths, basic_transform)

    # Calculate mean and std
    mean, std = calculate_mean_std(train_images)

    # Normalize the training dataset
    normalize_transform = transforms.Normalize(mean=mean, std=std)

    # Apply the normalization to each training image
    train_images = torch.stack([normalize_transform(image) for image in train_images])

    # Transformation with normalization for validation and test dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),  # 224x224 is a common "historical" format for benchmarking CNNs
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Load and normalize the validation and test datasets
    val_images = load_images(val_paths, transform)
    test_images = load_images(test_paths, transform)

    # Convert labels to tensors
    train_labels = torch.tensor(train_labels)
    val_labels = torch.tensor(val_labels)
    test_labels = torch.tensor(test_labels)

    # Create TensorDatasets
    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset = TensorDataset(val_images, val_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    return train_dataset, val_dataset, test_dataset
