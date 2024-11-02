import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader.Text4VPR_dataloader import GroupedImageDataset
from models.backbone import DINOv2
from models.Image_Aggregation import image_aggregation
from models.T5_model import LanguageEncoder
from tqdm import tqdm
import logging

def custom_collate_fn(batch):
    indices, coordinates, images, descriptions = zip(*batch)
    indices = list(indices)
    coordinates = list(coordinates)
    images = list(images)
    descriptions = list(descriptions)
    return indices, coordinates, images, descriptions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("test.log"), logging.StreamHandler()])


transform = transforms.Compose([
    transforms.Resize((252, 252)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

excel_file_test = r"your absolute path to dataset/test_description.xlsx"
image_root_dir = r"your absolute path to /dataset/Street360Loc_images"

dataset_test = GroupedImageDataset(excel_file_test, image_root_dir, transform=transform, drop_last=True)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

def load_model(model_path, device):
    dino = DINOv2(model_name='dinov2_vitb14', num_trainable_blocks=2, norm_layer=False, return_token=True)
    image_aggregation_model = image_aggregation(num_channels=768, num_clusters=16, cluster_dim=128, token_dim=256, dropout=0.3)
    image_encoder = torch.nn.Sequential(dino, image_aggregation_model)
    text_encoder = LanguageEncoder(embedding_dim=2048)

    checkpoint = torch.load(model_path)
    image_encoder.load_state_dict(checkpoint['image_encoder_state_dict'])
    text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])

    image_encoder.to(device)
    text_encoder.to(device)

    return image_encoder, text_encoder

def evaluate_on_testset(image_encoder, text_encoder, test_loader, device, top_k=[1, 5, 10], distance_thresholds=[5, 10, 15, 30]):
    image_encoder.eval()
    text_encoder.eval()
    all_text_encodings = []
    all_image_encodings = []
    all_positions = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Encoding Test Data"):
            indices, coordinates, images, descriptions = batch
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
            logging.info(f"Top-{k} accuracy within {p} meters: {accuracies[k][p]:.4f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = r'your weight file path under train/checkpoints/'
    image_encoder, text_encoder = load_model(model_path, device)
    evaluate_on_testset(image_encoder, text_encoder, dataloader_test, device)
    logging.info('Test evaluation completed!')
