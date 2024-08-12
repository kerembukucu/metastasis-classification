import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import timm
import matplotlib.pyplot as plt
from torchvision import datasets
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import numpy as np
from comet_ml import Experiment
import shutil
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Initialize Comet ML experiment
experiment = Experiment(
    api_key="dmkx3QlTCup6H6pTiKmLMKXf4",
    project_name="tumor_classifier",
    workspace="krmbkc"
)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def load_checkpoint(model, optimizer, filename='checkpoint.pth.tar'):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        val_accuracies = checkpoint['val_accuracies']
        confusion_matrices = checkpoint['confusion_matrices']
        sensitivities = checkpoint['sensitivities']
        specificities = checkpoint['specificities']
        precisions = checkpoint['precisions']
        npvs = checkpoint['npvs']
        f1_scores = checkpoint['f1_scores']
        print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))
        start_epoch = 0
        train_losses = []
        val_losses = []
        val_accuracies = []
        confusion_matrices = []
        sensitivities = []
        specificities = []
        precisions = []
        npvs = []
        f1_scores = []
    
    return start_epoch, train_losses, val_losses, val_accuracies, confusion_matrices, sensitivities, specificities, precisions, npvs, f1_scores

def train(model, train_loader, criterion, optimizer, device, epoch, fold_idx):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Training]"):
        inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

        desc = {'train_loss': loss}
        experiment.log_metrics(desc, epoch=epoch, prefix=f'fold_{fold_idx}')

    return running_loss / len(train_loader.dataset)

def validate(model, val_loader, criterion, device, epoch, fold_idx):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [Validation]"):
            inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = torch.sigmoid(outputs).cpu().numpy() > 0.5
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
    
    val_loss = running_loss / len(val_loader.dataset)
    val_accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    sensitivity = recall_score(all_labels, all_preds)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    precision = precision_score(all_labels, all_preds)
    npv = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    f1 = f1_score(all_labels, all_preds)

    desc = {
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "f1_score": f1
    }

    experiment.log_metrics(desc, epoch=epoch, prefix=f'fold_{fold_idx}')

    return val_loss, val_accuracy, cm, sensitivity, specificity, precision, npv, f1

def main():
    exp_name = 13
    num_epochs = 100
    num_classes = 1

    # Define transformations
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Cross-validation
    cv_path = 'D:/KEREM/data/cv'
    fold_dirs = [f.path for f in os.scandir(cv_path) if f.is_dir() and f.name.startswith('fold_')]
    
    for fold_id, fold_dir in enumerate(fold_dirs):
        print(f"Training on fold: {fold_dir}")
        
        # Load datasets
        train_dataset = datasets.ImageFolder(root=os.path.join(fold_dir, 'train'), transform=transform_train)
        val_dataset = datasets.ImageFolder(root=os.path.join(fold_dir, 'val'), transform=transform_val)

        # class weight hesapla
        num_samples = len(train_dataset)
        class_counts = np.bincount(train_dataset.targets)
        class_weights = num_samples / (class_counts * num_classes)
        weights = torch.FloatTensor(class_weights).to(device)

        # Create data loaders
        train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)

        # Load a pre-trained ResNet model from timm and modify the final layer
        model = timm.create_model('resnet34', pretrained=True)
        num_features = model.get_classifier().in_features
        model.fc = nn.Linear(num_features, num_classes)  # Replace the final layer with the number of classes

        # Move the model to the GPU if available

        model.to(device)

        # Define loss function with class weights
        criterion = nn.BCEWithLogitsLoss(pos_weight=weights[1])

        # Define optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.000003)

        # Define the ReduceLROnPlateau scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        # Load checkpoint if exists
        checkpoint_name = f"checkpoint_{exp_name}.pth.tar"
        start_epoch, train_losses, val_losses, val_accuracies, confusion_matrices, sensitivities, specificities, precisions, npvs, f1_scores = load_checkpoint(model, optimizer, filename=os.path.join(fold_dir, checkpoint_name))

        # Log hyperparameters to Comet ML
        hyperparameters = {
            "batch_size": 32,
            "num_epochs": num_epochs,
            "learning_rate": 0.001,
            "num_classes": 2,
            "optimizer": "Adam",
            "loss_function": "BCEWithLogitsLoss",
            "scheduler": "ReduceLROnPlateau",
            "scheduler_factor": 0.1,
            "scheduler_patience": 5
        }
        experiment.log_parameters(hyperparameters)

        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(start_epoch, num_epochs):
            train_loss = train(model, train_loader, criterion, optimizer, device, epoch, fold_id)
            val_loss, val_accuracy, cm, sensitivity, specificity, precision, npv, f1 = validate(model, val_loader, criterion, device, epoch, fold_id)

            scheduler.step(val_loss)

            # Update checkpoint only if the validation loss has improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'val_accuracies': val_accuracies,
                    'confusion_matrices': confusion_matrices,
                    'sensitivities': sensitivities,
                    'specificities': specificities,
                    'precisions': precisions,
                    'npvs': npvs,
                    'f1_scores': f1_scores
                }, filename=os.path.join(fold_dir, f"checkpoint_{exp_name}.pth.tar"))
                torch.save(model.state_dict(), os.path.join(fold_dir, f'resnet34_final_model_{exp_name}.pth'))
                print(f'Model saved with {exp_name}')
            else:
                epochs_without_improvement += 1

            # Append metrics to the respective lists
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            confusion_matrices.append(cm)

            sensitivities.append(sensitivity)
            specificities.append(specificity)
            precisions.append(precision)
            npvs.append(npv)
            f1_scores.append(f1)

            # Early stopping
            if epochs_without_improvement >= 10:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            result = f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}"
            print(result)

        # Plot the training and validation loss
        epochs = range(start_epoch + 1, len(train_losses) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_losses[start_epoch:], label='Train Loss')
        plt.plot(epochs, val_losses[start_epoch:], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Train and Validation Loss')
        plt.grid(True)
        plt.savefig(os.path.join(fold_dir, f"loss_plot_{exp_name}.png"))
        plt.close()
        
        # Log the loss plot to Comet ML
        experiment.log_image(os.path.join(fold_dir, f"loss_plot_{exp_name}.png"))

        # Calculate and log average metrics
        avg_val_accuracy = np.mean(val_accuracies)
        avg_sensitivity = np.mean(sensitivities)
        avg_specificity = np.mean(specificities)
        avg_precision = np.mean(precisions)
        avg_npv = np.mean(npvs)
        avg_f1 = np.mean(f1_scores)

        metrics = {
            "avg_val_accuracy": avg_val_accuracy,
            "avg_sensitivity": avg_sensitivity,
            "avg_specificity": avg_specificity,
            "avg_precision": avg_precision,
            "avg_npv": avg_npv,
            "avg_f1_score": avg_f1
        }

        experiment.log_metrics(metrics)
        
        # Log confusion matrix
        experiment.log_confusion_matrix(matrix=cm.tolist(), labels=["non-tumor", "tumor"])

if __name__ == '__main__':
    main()
