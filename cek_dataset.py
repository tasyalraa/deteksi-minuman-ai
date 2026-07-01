import os

DATASET_DIR = "dataset_clean"

for folder in os.listdir(DATASET_DIR):
    path = os.path.join(DATASET_DIR, folder)
    
    if os.path.isdir(path):
        jumlah = len(os.listdir(path))
        print(f"{folder}: {jumlah} gambar")