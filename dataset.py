import os
from PIL import Image
from torch.utils.data import Dataset
import torch

class OCRDataset(Dataset):
    def __init__(self, dataset_dir, df, processor, tokenizer, max_target_length=78):
        self.dataset_dir = dataset_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text
        file_name = self.df[0][idx]
        text = self.df[1][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(os.path.join(self.dataset_dir,file_name)).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.tokenizer(text,
                                padding="max_length",
                                truncation=True,
                                max_length=self.max_target_length).input_ids

        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding