import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class GroupedImageDataset(Dataset):
    def __init__(self, excel_file, image_root_dir, transform=None, drop_last=False):
        self.data = pd.read_excel(excel_file)
        self.groups = self.data.groupby('index')
        self.image_root_dir = image_root_dir
        self.transform = transform
        self.drop_last = drop_last

    def __len__(self):
        length = len(self.groups)
        if self.drop_last:
            return length - (length % 2)
        return length

    def __getitem__(self, idx):
        index = list(self.groups.groups.keys())[idx]
        group = self.groups.get_group(index)
        x = group['x'].iloc[0]
        y = group['y'].iloc[0]
        z = group['z'].iloc[0]
        coordinates = (x, y, z)
        image_paths = [
            os.path.join(self.image_root_dir, f"{int(row['index']):06}_{int(row['pos'])}.jpg")
            for _, row in group.iterrows()
        ]
        images = []
        for img_path in image_paths:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)
        descriptions = group['text_description'].tolist()
        return index, coordinates, images, descriptions


def custom_collate_fn(batch):
    indices, coordinates, images, descriptions = zip(*batch)
    indices = list(indices)
    coordinates = list(coordinates)
    images = list(images)
    descriptions = list(descriptions)
    return indices, coordinates, images, descriptions


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
