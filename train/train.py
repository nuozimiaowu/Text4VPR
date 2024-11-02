import os
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()])

transform = transforms.Compose([
    transforms.Resize((252, 252)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

excel_file_train = r"your absolute path to /dataset/train_description.xlsx"
excel_file_val = r"your absolute path to /dataset/val_description.xlsx"
image_root_dir = r"your absolute path to /dataset/Street360Loc_images"

dataset_train = GroupedImageDataset(excel_file_train, image_root_dir, transform=transform, drop_last=True)
dataset_val = GroupedImageDataset(excel_file_val, image_root_dir, transform=transform, drop_last=True)


def custom_collate_fn(batch):
    indices, coordinates, images, descriptions = zip(*batch)
    indices = list(indices)
    coordinates = list(coordinates)
    images = list(images)
    descriptions = list(descriptions)
    return indices, coordinates, images, descriptions

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

def validate_on_valset(image_encoder, text_encoder, val_loader, device, top_k=[1, 5, 10], distance_thresholds=[5, 10, 15, 30]):
    image_encoder.eval()
    text_encoder.eval()
    all_text_encodings = []
    all_image_encodings = []
    all_positions = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Encoding Validation Data"):
            indices, coordinates, images, descriptions = batch

            print(f"images type: {type(images)}")
            print(f"coordinates: {coordinates}")

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

dino = DINOv2(model_name='dinov2_vitb14', num_trainable_blocks=2, norm_layer=False, return_token=True)
image_aggregation_model = image_aggregation(num_channels=768,
                                            num_clusters=16,
                                            cluster_dim=128,
                                            token_dim=256,
                                            dropout=0.3)
image_encoder = torch.nn.Sequential(dino, image_aggregation_model)
text_encoder = LanguageEncoder(embedding_dim=2048)
contrastive_loss = ContrastiveLoss(temperature=0.1)
optimizer = optim.Adam(list(image_encoder.parameters()) + list(text_encoder.parameters()), lr=1e-4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_encoder.to(device)
text_encoder.to(device)

num_epochs = 10

best_accuracy = 0 

for epoch in range(num_epochs):
    image_encoder.train()
    text_encoder.train()
    total_loss = 0

    with tqdm(total=len(dataloader_train), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
        for indices, coordinates, images, descriptions in dataloader_train:
            # Process images
            image_descriptors = []
            for img_group in images:
                group_descriptors = []
                for img in img_group:
                    img = img.to(device)
                    img = img.unsqueeze(0)
                    image_descriptor = image_encoder(img)
                    group_descriptors.append(image_descriptor.squeeze(0))
                concatenated_group_descriptor = torch.cat(group_descriptors, dim=-1)
                image_descriptors.append(concatenated_group_descriptor)
            image_descriptors = torch.stack(image_descriptors)

            # Process text
            text_descriptors = []
            for group_descriptions in descriptions:
                group_text_encodings = []
                for desc in group_descriptions:
                    text_descriptor = text_encoder([desc])
                    group_text_encodings.append(text_descriptor.squeeze(0))
                concatenated_text_descriptor = torch.cat(group_text_encodings, dim=-1)
                text_descriptors.append(concatenated_text_descriptor)
            text_descriptors = torch.stack(text_descriptors)

            # Calculate loss
            image_descriptors_2048 = image_descriptors.view(-1, 2048)
            text_descriptors_2048 = text_descriptors.view(-1, 2048)
            loss = contrastive_loss(image_descriptors_2048, text_descriptors_2048)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'Loss': loss.item()})
            pbar.update(1)

    avg_loss = total_loss / len(dataloader_train)
    logging.info(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

    logging.info(f'Validating model on validation set after epoch {epoch + 1}')
    accuracies_val = validate_on_valset(image_encoder, text_encoder, dataloader_val, device)

    # Calculate current top1 accuracy
    current_top1_accuracy = accuracies_val[1][5]

    logging.info(f'Validation Accuracies after epoch {epoch + 1}:')
    for k in [1, 5, 10]:
        for p in [5, 10, 15, 30]:
            logging.info(f"Top-{k} accuracy within {p} meters on validation set: {accuracies_val[k][p]:.4f}")

    # If current top1 accuracy exceeds previous best, save the model immediately
    if current_top1_accuracy > best_accuracy:
        best_accuracy = current_top1_accuracy
        model_filename = os.path.join('checkpoints', f'model_top1_accuracy_{best_accuracy:.4f}.pth')
        torch.save({
            'image_encoder_state_dict': image_encoder.state_dict(),
            'text_encoder_state_dict': text_encoder.state_dict(),
        }, model_filename)
        logging.info(f'New best model saved as: {model_filename}')

<<<<<<< HEAD
logging.info('Training complete!')
=======
logging.info('Training complete!')
>>>>>>> a8237670e9185a5435fa9e8e3376ac083c568e38
