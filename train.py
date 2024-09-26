import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=8):  # Adjust `out_channels` to match number of classes
        super(UNet, self).__init__()

        # Contracting path
        self.encoder1 = self.contract_block(in_channels, 64)
        self.encoder2 = self.contract_block(64, 128)
        self.encoder3 = self.contract_block(128, 256)
        self.encoder4 = self.contract_block(256, 512)

        # Bottleneck
        self.bottleneck = self.contract_block(512, 1024)

        # Expanding path
        self.upconv4 = self.expand_block(1024, 512)
        self.upconv3 = self.expand_block(512, 256)
        self.upconv2 = self.expand_block(256, 128)
        self.upconv1 = self.expand_block(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Contracting path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # Expanding path
        up4 = self.upconv4(bottleneck)
        up4 = torch.cat((up4, enc4), dim=1)
        up3 = self.upconv3(up4)
        up3 = torch.cat((up3, enc3), dim=1)
        up2 = self.upconv2(up3)
        up2 = torch.cat((up2, enc2), dim=1)
        up1 = self.upconv1(up2)
        up1 = torch.cat((up1, enc1), dim=1)

        return self.final_conv(up1)

    def contract_block(self, in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size),
            nn.ReLU()
        )

    def expand_block(self, in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
        )

import torch
import torch.nn as nn

# Example hierarchy for COCO dataset
hierarchy = {
    "animal": ["dog", "cat", "horse"],
    "vehicle": ["car", "truck", "motorcycle"],
    "furniture": ["chair", "couch"],
    # Add more as needed
}

class_to_id = {
    "dog": 0, "cat": 1, "horse": 2, "car": 3, "truck": 4, "motorcycle": 5,
    "chair": 6, "couch": 7
}

def find_class_group(class_name, hierarchy):
    """Find the group (parent) for a class."""
    for parent, children in hierarchy.items():
        if class_name in children:
            return parent
    return None

def path_distance(pred_class, true_class, hierarchy):
    """Calculate the path-based distance between two classes."""
    pred_parent = find_class_group(pred_class, hierarchy)
    true_parent = find_class_group(true_class, hierarchy)

    if pred_parent == true_parent:
        return 0
    else:
        return 1

class PathBasedLoss(nn.Module):
    def __init__(self, hierarchy, class_to_id):
        super(PathBasedLoss, self).__init__()
        self.hierarchy = hierarchy
        self.class_to_id = class_to_id

    def forward(self, predictions, targets):
        batch_size = predictions.size(0)
        loss = 0.0
        probs = torch.softmax(predictions, dim=1)

        for i in range(batch_size):
            pred_class_idx = torch.argmax(probs[i]).item()
            pred_class_name = list(self.class_to_id.keys())[list(self.class_to_id.values()).index(pred_class_idx)]

            true_class_idx = targets[i].item()
            true_class_name = list(self.class_to_id.keys())[list(self.class_to_id.values()).index(true_class_idx)]

            dist = path_distance(pred_class_name, true_class_name, self.hierarchy)

            loss += dist
        
        return loss / batch_size


def installation_coco():

    import os
    import requests
    import zipfile

    # URLs de téléchargement pour les images et les annotations de COCO
    coco_train_images_url = "http://images.cocodataset.org/zips/train2017.zip"
    coco_annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

    # Chemins où les fichiers zip seront téléchargés
    images_zip_path = "train2017.zip"
    annotations_zip_path = "annotations_trainval2017.zip"

    # Répertoires où les fichiers seront extraits
    images_dir = "path_to_coco/images/train2017"
    annotations_dir = "path_to_coco/annotations"

    # Fonction pour télécharger un fichier
    def download_file(url, save_path):
        print(f"Téléchargement depuis {url} ...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Téléchargement terminé : {save_path}")

    # Fonction pour extraire un fichier zip
    def extract_zip(zip_path, extract_to):
        print(f"Extraction de {zip_path} dans {extract_to} ...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extraction terminée : {extract_to}")

    # Création des répertoires si nécessaires
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    # Télécharger et extraire les images
    download_file(coco_train_images_url, images_zip_path)
    extract_zip(images_zip_path, images_dir)

    # Télécharger et extraire les annotations
    download_file(coco_annotations_url, annotations_zip_path)
    extract_zip(annotations_zip_path, annotations_dir)

    # Supprimer les fichiers zip après extraction (optionnel)
    os.remove(images_zip_path)
    os.remove(annotations_zip_path)

    print("Téléchargement et extraction terminés pour COCO.")






import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader, random_split

# Define paths to COCO images and annotations
coco_train_images_path = 'path_to_coco\\images\\train2017\\train2017'
coco_train_annotations_path = "path_to_coco\\annotations\\annotations\\instances_train2017.json"

# Transformation applied to images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Load COCO dataset
full_coco_dataset = datasets.CocoDetection(
    root=coco_train_images_path,
    annFile=coco_train_annotations_path,
    transform=transform
)
img, _ = full_coco_dataset[5]  # Get the first image
print(img)
print(f"Resized image size: {img.size()}")

# Visualize the resized image
plt.imshow(img.permute(1, 2, 0))  # PyTorch uses (C, H, W), but plt.imshow needs (H, W, C)
plt.title(f"Resized Image: {img.size()}")
plt.axis('off')
plt.show()

# Define a subset size (for example, 5% of the dataset)

train_size = int(0.8 * len(full_coco_dataset))
test_size = len(full_coco_dataset) - train_size
# Split into train and test subsets
train_subset, test_subset = random_split(full_coco_dataset, [train_size, test_size])

print(train_subset[0])

# Display the first 10 indices of the train subset
# Create data loaders for training and testing
train_loader = DataLoader(dataset=train_subset, batch_size=8, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_subset, batch_size=8, shuffle=False, num_workers=0)
print(train_loader.dataset[0])
print(f"Total Dataset Size: {len(full_coco_dataset)}")


## Visualize the first few samples from the training set



import torchvision
import torch

# Function to display images from a batch of data
def show_images_from_loader(loader, num_images=8):
    # Get a batch of training data
    dataiter = iter(loader)
    images, labels = next(dataiter)  # Get one batch of data
    
    # Optionally, select only the first `num_images` images
    images = images[:num_images]
    
    # Create a grid of images
    grid_img = torchvision.utils.make_grid(images, nrow=4)
    
    # Convert the images to numpy format and transpose for plotting
    npimg = grid_img.numpy()
    plt.figure(figsize=(10, 10))  # Set the figure size
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # Transpose to (H, W, C) for plotting
    plt.axis('off')  # Hide axis labels
    plt.show()

# Example usage: Plot images from the train loader
# Assuming train_loader is your DataLoader object
show_images_from_loader(train_loader)




# Initialize model, loss function, and optimizer
model = UNet(in_channels=3, out_channels=8)  # Adjust based on number of classes
criterion = PathBasedLoss(hierarchy=hierarchy, class_to_id=class_to_id)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, targets in train_loader:
        images = images.float()

        # Forward pass
        outputs = model(images)
        
        # Convert target segmentation masks to class indices
        target_class_indices = [class_to_id[label['category_id']] for label in targets]
        target_class_indices = torch.tensor(target_class_indices).long()

        # Compute loss
        loss = criterion(outputs, target_class_indices)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

# Save the trained model
torch.save(model.state_dict(), 'unet_trained_subset.pth')

import torch
import numpy as np
from sklearn.metrics import jaccard_score




# Switch the model to evaluation mode
model = UNet(in_channels=3, out_channels=8)  # Adjust output channels
model.load_state_dict(torch.load('unet_trained_subset.pth'))
model.eval()

# Function to compute IoU
def compute_iou(pred_mask, true_mask, num_classes):
    iou_per_class = []
    
    for cls in range(num_classes):
        pred = (pred_mask == cls).astype(np.uint8)
        true = (true_mask == cls).astype(np.uint8)
        
        if np.sum(true) == 0 and np.sum(pred) == 0:
            iou = 1.0
        else:
            iou = jaccard_score(true.flatten(), pred.flatten(), average='binary')
        iou_per_class.append(iou)
    
    return np.mean(iou_per_class)

# Evaluate on test dataset
iou_scores = []
with torch.no_grad():
    for images, targets in test_loader:
        images = images.float()
        outputs = model(images)
        
        predicted_masks = torch.argmax(outputs, dim=1).cpu().numpy()
        for i in range(len(images)):
            # Extract the ground truth mask and predicted mask for this image
            true_mask = np.array([label['category_id'] for label in targets[i]])
            pred_mask = predicted_masks[i]

            # Compute IoU for this sample
            iou = compute_iou(pred_mask, true_mask, num_classes=8)  # Adjust number of classes
            iou_scores.append(iou)
mean_iou = np.mean(iou_scores)
print(f"Mean IoU over test set: {mean_iou:.4f}")

# Function to visualize segmentation results
import matplotlib.pyplot as plt
def visualize_segmentation(image, true_mask, pred_mask, color_map):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(image.permute(1, 2, 0))  # Display original image
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(true_mask, cmap='gray')  # Display ground truth mask
    ax[1].set_title('Ground Truth Mask')
    ax[1].axis('off')

    ax[2].imshow(pred_mask, cmap='gray')  # Display predicted mask
    ax[2].set_title('Predicted Mask')
    ax[2].axis('off')

    plt.show()

# Visualize the first few samples from the test set
with torch.no_grad():
    for images, targets in test_loader:
        images = images.float()
        outputs = model(images)
        predicted_masks = torch.argmax(outputs, dim=1).cpu().numpy()

        for i in range(len(images)):
            image = images[i].cpu()
            true_mask = np.array([label['category_id'] for label in targets[i]])
            pred_mask = predicted_masks[i]

            # Visualize the original image, true mask, and predicted mask
            if i < 5:  # Visualize the first 5 samples
                visualize_segmentation(image, true_mask, pred_mask, color_map=None)

