import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader.Text4VPR_dataloader import GroupedImageDataset
from models.backbone import DINOv2
from models.Image_Aggregation import image_aggregation
from models.T5_model import LanguageEncoder
from models.loss import ContrastiveLoss
from tqdm import tqdm
import logging
import numpy as np

torch.cuda.empty_cache()
batch_size = 64

# Initialize logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("test.log"), logging.StreamHandler()])

# Define transformations
transform = transforms.Compose([
    transforms.Resize((252, 252)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# File paths for test dataset
excel_file_test = r"your path to /dataset/test_description.xlsx"
image_root_dir = r"your path to /dataset/images"

# Initialize dataset and DataLoader for testing
dataset_test = GroupedImageDataset(excel_file_test, image_root_dir, transform=transform, drop_last=True)

def custom_collate_fn(batch):
    indices, coordinates, images, descriptions = zip(*batch)
    indices = list(indices)
    coordinates = list(coordinates)
    images = list(images)
    descriptions = list(descriptions)
    return indices, coordinates, images, descriptions

dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

def validate_on_testset(image_encoder, text_encoder, test_loader, device, top_k=[1, 5, 10], distance_thresholds=[5, 10, 15, 30]):
    image_encoder.eval()
    text_encoder.eval()
    all_text_encodings = []
    all_image_encodings = []
    all_positions = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Encoding Test Data"):
            indices, coordinates, images, descriptions = batch

            if not isinstance(images, (list, tuple)) or not all(isinstance(img_group, list) for img_group in images):
                print(f"Skipping batch due to invalid image format: {type(images)}")
                continue

            images = [[img.to(device) for img in img_group] for img_group in images]

            image_descriptors = []
            for img_group in images:
                group_descriptors = []
                for img in img_group:
                    img = img.unsqueeze(0)
                    image_descriptor = image_encoder(img).squeeze(0)
                    group_descriptors.append(image_descriptor)
                concatenated_image_descriptor = torch.cat(group_descriptors, dim=-1)
                image_descriptors.append(concatenated_image_descriptor)

            image_descriptors = torch.stack(image_descriptors)
            text_descriptors = []
            for group_descriptions in descriptions:
                group_text_encodings = []
                for desc in group_descriptions:
                    text_descriptor = text_encoder([desc]).squeeze(0)
                    group_text_encodings.append(text_descriptor)
                concatenated_text_descriptor = torch.cat(group_text_encodings, dim=-1)
                text_descriptors.append(concatenated_text_descriptor)
            text_descriptors = torch.stack(text_descriptors)

            all_text_encodings.append(text_descriptors.cpu())
            all_image_encodings.append(image_descriptors.cpu())
            all_positions.extend(coordinates)

    all_text_encodings = torch.cat(all_text_encodings, dim=0).numpy()
    all_image_encodings = torch.cat(all_image_encodings, dim=0).numpy()
    all_positions = np.array(all_positions)

    num_queries = len(all_text_encodings)
    accuracies = {k: {p: 0 for p in distance_thresholds} for k in top_k}

    for query_idx in tqdm(range(num_queries), desc="Evaluating Top-K Accuracy"):
        text_encoding = all_text_encodings[query_idx]
        query_position = all_positions[query_idx]
        similarities = np.dot(all_image_encodings, text_encoding)
        sorted_indices = np.argsort(-similarities)
        distances = np.linalg.norm(all_positions[sorted_indices] - query_position, axis=1)

        for k in top_k:
            top_k_indices = sorted_indices[:k]
            top_k_distances = distances[:k]
            for p in distance_thresholds:
                if any(top_k_distances <= p):
                    accuracies[k][p] += 1

    for k in top_k:
        for p in distance_thresholds:
            accuracies[k][p] /= num_queries
            print(f"Top-{k} accuracy within {p} meters: {accuracies[k][p]:.4f}")

    return accuracies

# Initialize the models
dino = DINOv2(model_name='dinov2_vitb14', num_trainable_blocks=2, norm_layer=False, return_token=True)
image_aggregation_model = image_aggregation(num_channels=768,
                                            num_clusters=16,
                                            cluster_dim=128,
                                            token_dim=256,
                                            dropout=0.3)
image_encoder = torch.nn.Sequential(dino, image_aggregation_model)
text_encoder = LanguageEncoder(embedding_dim=2048)

# Load saved weights for the models
image_encoder.load_state_dict(torch.load("path_to_saved_image_encoder_weights.pth"))
text_encoder.load_state_dict(torch.load("path_to_saved_text_encoder_weights.pth"))

# Define the loss function and optimizer (for consistency)
contrastive_loss = ContrastiveLoss(temperature=0.1)
optimizer = optim.Adam(list(image_encoder.parameters()) + list(text_encoder.parameters()), lr=1e-4)

# Define device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_encoder.to(device)
text_encoder.to(device)

# Run the testing
logging.info('Testing model on test set...')
accuracies_test = validate_on_testset(image_encoder, text_encoder, dataloader_test, device)

logging.info(f'Test Accuracies:')
for k in [1, 5, 10]:
    for p in [5, 10, 15, 30]:
        logging.info(f"Top-{k} accuracy within {p} meters on test set: {accuracies_test[k][p]:.4f}")
