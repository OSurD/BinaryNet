import torchaudio
import os

DATA_DIR = os.path.abspath('./data')
SPEECHCOMMANDS_DATASET_PATH = os.path.join(DATA_DIR, "SPEECHCOMMANDS")

os.makedirs(SPEECHCOMMANDS_DATASET_PATH, exist_ok=True)
dataset = torchaudio.datasets.SPEECHCOMMANDS(SPEECHCOMMANDS_DATASET_PATH , download = True)
