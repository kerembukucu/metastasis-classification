import os
import shutil
from sklearn.model_selection import KFold
from glob import glob

# Path configuration
dataset_path = r'D:\KEREM\data'
cv_path = os.path.join(dataset_path, 'cv')
tumor_path = os.path.join(dataset_path, 'cv/tumor')
non_tumor_path = os.path.join(dataset_path, 'cv/normal_from_tumor_slide')

# Create cv directory if it doesn't exist
os.makedirs(cv_path, exist_ok=True)

# Get list of patients
patients = list(set([x.split('\\')[-1][:11] for x in glob(os.path.join(non_tumor_path, 'p*'))]))

# Create K-Folds
kf = KFold(n_splits=10, shuffle=True, random_state=42)

fold_idx = 0
for train_index, test_index in kf.split(patients):
    fold_path = os.path.join(cv_path, f'fold_{fold_idx}')
    os.makedirs(fold_path, exist_ok=True)
    
    # Create directories for this fold
    train_tumor_path = os.path.join(fold_path, 'train', 'tumor')
    train_non_tumor_path = os.path.join(fold_path, 'train', 'non-tumor')
    val_tumor_path = os.path.join(fold_path, 'val', 'tumor')
    val_non_tumor_path = os.path.join(fold_path, 'val', 'non-tumor')
    os.makedirs(train_tumor_path, exist_ok=True)
    os.makedirs(train_non_tumor_path, exist_ok=True)
    os.makedirs(val_tumor_path, exist_ok=True)
    os.makedirs(val_non_tumor_path, exist_ok=True)
    
    train_patients, val_patients = [patients[i] for i in train_index], [patients[i] for i in test_index]
    
    for patient in train_patients:
        # Copy tumor data
        for src in glob(os.path.join(tumor_path, patient) + '*'):
            shutil.copytree(src, os.path.join(train_tumor_path, os.path.basename(src)))
        
        # Copy non-tumor data
        for src in glob(os.path.join(non_tumor_path, patient) + '*'):
            shutil.copytree(src, os.path.join(train_non_tumor_path, os.path.basename(src)))
    
    for patient in val_patients:
        # Copy tumor data
        for src in glob(os.path.join(tumor_path, patient) + '*'):
            shutil.copytree(src, os.path.join(val_tumor_path, os.path.basename(src)))
        
        # Copy non-tumor data
        for src in glob(os.path.join(non_tumor_path, patient) + '*'):
            shutil.copytree(src, os.path.join(val_non_tumor_path, os.path.basename(src)))

    fold_idx += 1
