import json
from torch.utils.data import Dataset

class ParallelSFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_samples=None):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.samples = []
        self.load_data()
        if max_samples:
            self.samples = self.samples[:max_samples]

    def load_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    self.samples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # This is where the magic will happen.
        # For now, just return the raw sample.
        return self.samples[idx]
