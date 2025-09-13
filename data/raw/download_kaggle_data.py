import kagglehub

# Download latest version
path = kagglehub.dataset_download("keonim/nfl-game-scores-dataset-2017-2023")

print("Path to dataset files:", path)