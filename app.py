from PIL import Image
import torch
import argparse
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from dataset import decode_text,trocrDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='trocr fine-tune訓練')
    parser.add_argument('--model_path', default='./checkpoint/trocr/last', type=str,
                        help="訓練好的模型路徑")
    parser.add_argument('--test_img', default='test_data/en/en_img', type=str, help="img path")
    parser.add_argument('--train_label_text_path', default='test_data/en/gt_test.txt', type=str, help="訓練標籤位置")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    tokenizer = TrOCRProcessor.from_pretrained(args.model_path)

    test_df = pd.read_csv(args.train_label_text_path,sep='.jpg',header=None,engine='python')
    test_df[0] =test_df[0]+'.jpg'

    test_dataset = trocrDataset(root_dir=args.test_img,
                            df=test_df,
                            processor=tokenizer)


    test_dataloader = DataLoader(test_dataset, batch_size=16)

    batch = next(iter(test_dataloader))

    model = VisionEncoderDecoderModel.from_pretrained(args.model_path)
    model.to(device)

    total_preds = []
    total_labels = []

    for batch in tqdm(test_dataloader):

        pixel_values = batch["pixel_values"].to(device)
        outputs = model.generate(pixel_values)

        pred_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        total_preds.extend(pred_str)
        labels = batch["labels"]
        labels[labels == -100] = tokenizer.tokenizer.pad_token_id
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        total_labels.extend(labels)

    with open('ground.txt','w') as f:
        for line in total_labels:
            f.write(line+'\n')

    with open('predict.txt','w') as f:
        for line in total_preds:
            f.write(line+'\n')

    with open('gt.txt','w') as f:
        for line in total_labels:
            for word in line:
                f.write(word+' ')
            f.write('\n')


    with open('pred.txt','w') as f:
        for line in total_preds:
            for word in line:
                f.write(word+' ')
            f.write('\n')
    os.system('wer -c gt.txt pred.txt > cer.txt')
    os.system('rm -rf gt.txt pred.txt')