import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class GroupedImageDataset(Dataset):
    def __init__(self, excel_file, image_root_dir, transform=None):
        # Read the Excel file
        self.data = pd.read_excel(excel_file)

        # Group by index
        self.groups = self.data.groupby('index')

        # Store the root path of the images and transformation operations
        self.image_root_dir = image_root_dir
        self.transform = transform

    def __len__(self):
        # Return the number of unique groups (indices)
        return len(self.groups)

    def __getitem__(self, idx):
        # Get the grouped data
        index = list(self.groups.groups.keys())[idx]  # Get the corresponding index
        group = self.groups.get_group(index)  # Get all rows corresponding to this index

        # Get x, y, z coordinates (assuming these values are the same for each index)
        x = group['x'].iloc[0]
        y = group['y'].iloc[0]
        z = group['z'].iloc[0]
        coordinates = (x, y, z)

        # Construct image paths
        image_paths = [
            os.path.join(self.image_root_dir, f"{int(row['index']):06}_{int(row['pos'])}.jpg")
            for _, row in group.iterrows()
        ]

        # Open images and apply transformations
        images = []
        for img_path in image_paths:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)

        # Get descriptions
        descriptions = group['text_description'].tolist()

        return index, coordinates, images, descriptions


# Define a collate_fn to avoid PyTorch's default behavior
def custom_collate_fn(batch):
    indices, coordinates, images, descriptions = zip(*batch)

    # Keep indices and coordinates as lists to avoid being converted into Tensors
    indices = list(indices)
    coordinates = list(coordinates)

    # Keep images and descriptions as lists of lists
    images = list(images)
    descriptions = list(descriptions)

    return indices, coordinates, images, descriptions


# Define image transformation operations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image
    transforms.ToTensor(),  # Convert to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])
