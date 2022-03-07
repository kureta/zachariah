from pathlib import Path

from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    def __getitem__(self, index):
        print(index)

    def __len__(self):
        pass
