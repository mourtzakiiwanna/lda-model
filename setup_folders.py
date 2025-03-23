import os

folders = [
    "data/raw",
    "data/processed",
    "outputs/lda_model",
    "outputs/visuals",
    "notebooks",
    "scripts"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("Project folders created successfully!")
