import os
from PIL import Image
from torch.utils.data import Dataset
import torch

class trocrDataset(Dataset):
    def __init__(self, root_dir,df ,processor, max_target_length=256):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
            # 圖片名
            file_name = self.df[0][idx]
            #對應標籤
            text = self.df[1][idx]
            if file_name.endswith('jp'):
                file_name = file_name + 'g'

            image = Image.open(os.path.join(self.root_dir,file_name)).convert("RGB")
            pixel_values = self.processor(image, return_tensors="pt").pixel_values

            labels = self.processor.tokenizer(text,
                                              padding="max_length",
                                              truncation=True,
                                              max_length=self.max_target_length).input_ids

            labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

            encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
            return encoding


def encode_text(text, max_target_length=128, vocab=None):
    """
    # list: ['<td>',"3","3",'</td>',....]
    {'input_ids': [0, 1092, 2, 1, 1],
    'attention_mask': [1, 1, 1, 0, 0]}
    """
    if type(text) is not list:
       text = list(text)

    text = text[:max_target_length - 2]
    tokens = [vocab.get('<s>')] #-> [0]
    unk = vocab.get('<unk>') # -> 3
    pad = vocab.get('<pad>') # -> 1
    mask = []
    for tk in text:
        token = vocab.get(tk, unk)  #-> 沒看過的分詞會被定義unk
        tokens.append(token)
        mask.append(1)

    tokens.append(vocab.get('</s>')) #-> 2
    mask.append(1)

    if len(tokens) < max_target_length:
        for i in range(max_target_length - len(tokens)):
            tokens.append(pad)
            mask.append(0)

    return tokens
    #return {"input_ids": tokens, 'attention_mask': mask}

def decode_text(tokens, vocab, vocab_inp):
    ##decode trocr
    s_start = vocab.get('<s>') #-> [0]
    s_end = vocab.get('</s>') # -> 2
    unk = vocab.get('<unk>') # -> 3
    pad = vocab.get('<pad>') # -> 1
    text = ''
    for tk in tokens:
        if tk not in [s_end, s_start , pad, unk]:
           text += vocab_inp[tk]
    return text
