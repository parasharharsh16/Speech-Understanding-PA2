from VOXCELEB1_dataset import VoxCeleb1

from torchaudio.datasets import voxceleb1
voxceleb1_dataset = voxceleb1.VoxCeleb1(root="dataset", download=True)


print("test")