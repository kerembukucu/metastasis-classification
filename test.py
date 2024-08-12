import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import timm
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve
import numpy as np
from numpy import sqrt, argmax

expname = 13
fold_number = 4
change = "g-mean"

# Define transformations for the test dataset to match validation
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom dataset for test images
class TestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images, self.labels = self.make_dataset()

    def make_dataset(self):
        images = []
        labels = []
        class_to_idx = {'tumor': 1, 'non-tumor': 0}  # Class mapping

        for root, dirs, files in os.walk(self.root):
            for file in files:
                if file.lower().endswith(('jpg', 'jpeg', 'png')):
                    image_path = os.path.join(root, file)
                    images.append(image_path)
                    # Extract class label from the parent folder name
                    parent_folder = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
                    labels.append(class_to_idx.get(parent_folder, -1))  # Assign -1 if parent_folder not found

        # Check for any labels assigned -1, indicating a problem with class_name extraction
        if -1 in labels:
            print("Warning: Some images have been assigned an invalid label (-1). Please check your folder structure.")

        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        label = self.labels[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, path

# Load the model
num_classes = 1  # Use 1 if your saved model has 1 output node
model = timm.create_model('resnet34', pretrained=False)
num_features = model.get_classifier().in_features
model.fc = nn.Linear(num_features, num_classes)
model.load_state_dict(torch.load(f'data/cv/fold_{fold_number}/resnet34_final_model_{expname}.pth'))  # Load the trained weights

# Move the model to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Set the model to evaluation mode
model.eval()

# Create the test data loader
test_dataset = TestDataset(root='data/test', transform=transform_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# Test the model
def test(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []
    all_paths = []
    all_outputs = []  # Store raw outputs for inspection
    pbar = tqdm(test_loader, total=len(test_loader))
    with torch.no_grad():
        for images, labels, paths in pbar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images).sigmoid().cpu().numpy()  # Forward pass
            all_outputs.extend(outputs)  # Collect raw outputs
            preds = (outputs > 0.5).astype(float)  # Use 0.5 threshold for initial prediction
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_paths.extend(paths)
    
    # Convert all outputs and labels to numpy arrays
    all_outputs_np = np.array(all_outputs).flatten()
    all_labels_np = np.array(all_labels).flatten()

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels_np, all_outputs_np)

    # Calculate G-Mean for each threshold
    gmeans = sqrt(tpr * (1 - fpr))
    
    # Locate the index of the largest G-Mean
    ix = argmax(gmeans)
    best_threshold = thresholds[ix]
    best_gmean = gmeans[ix]
    print(f'Best Threshold={best_threshold:.6f}, G-Mean={best_gmean:.3f}')
    
    # Apply the best threshold to predictions
    preds_adjusted = (all_outputs_np > best_threshold).astype(float)

    # Compute confusion matrix
    cm = confusion_matrix(all_labels_np, preds_adjusted)

    # Print the distribution of raw output probabilities
    print("Raw output probabilities:")
    print(all_outputs_np)
    print("Mean of raw outputs:", np.mean(all_outputs_np))
    print("Standard deviation of raw outputs:", np.std(all_outputs_np))
    print("Min of raw outputs:", np.min(all_outputs_np))
    print("Max of raw outputs:", np.max(all_outputs_np))

    # Print unique predictions and their counts
    unique_preds, counts_preds = np.unique(preds_adjusted, return_counts=True)
    print(f'Prediction distribution: {dict(zip(unique_preds, counts_preds))}')
    
    # Print unique labels and their counts
    unique_labels, counts_labels = np.unique(all_labels_np, return_counts=True)
    print(f'True label distribution: {dict(zip(unique_labels, counts_labels))}')
    
    return preds_adjusted, all_labels_np, all_paths, cm, fpr, tpr, thresholds, ix

# Run the test
preds, labels, paths, cm, fpr, tpr, thresholds, ix = test(model, test_loader, device)

# Function to calculate metrics
def calculate_metrics(cm):
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0  # Sensitivity, True positive rate
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0  # True negative rate
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0  # Positive predictive value
    npv = tn / (tn + fn) if (tn + fn) != 0 else 0  # Negative predictive value
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0  # F1 score
    return {
        'Recall (Sensitivity, TPR)': recall,
        'Specificity (TNR)': specificity,
        'Precision (PPV)': precision,
        'Negative Predictive Value (NPV)': npv,
        'Accuracy': accuracy,
        'F1 Score': f1
    }

# Calculate metrics
metrics = calculate_metrics(cm)

# Save metrics to a file
with open(f'metrics/metrics_{expname}_{fold_number}_{change}.txt', 'w') as f:
    for metric, value in metrics.items():
        f.write(f"{metric}: {value:.2f}\n")

# Plot and save confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Non-Tumor', 'Tumor'], yticklabels=['Non-Tumor', 'Tumor'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig(f'confusion/confusion_matrix_{expname}_{fold_number}_{change}.png')

# Plot ROC curve with the optimal threshold
plt.figure(figsize=(10, 8))
plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
plt.plot(fpr, tpr, marker='.', label='Model')
plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best Threshold')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with Optimal Threshold')
plt.legend()
plt.savefig(f'roc_curve_{expname}_{fold_number}_{change}.png')
plt.show()