from VOXCELEB1_dataset import VoxCeleb1Identification

# root directory where the dataset will be stored or is already stored
root_dir = 'dataset/voxceleb1_dataset'

# Create an instance of the VoxCeleb1Identification dataset
dataset = VoxCeleb1Identification(root=root_dir, download=True)
