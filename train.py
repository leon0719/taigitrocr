import os
import argparse
import pandas as pd
from dataset import trocrDataset, decode_text
from transformers import TrOCRProcessor
from transformers import VisionEncoderDecoderModel
from transformers import default_data_collator
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_metric

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='trocr fine-tune訓練')
    # parser.add_argument('--cust_data_init_weights_path', default='./weights', type=str,
    #                     help="初始化訓練權重，用於自己數據集上fine-tune權重")
    parser.add_argument('--checkpoint_path', default='./checkpoint/trocr', type=str, help="訓練模型保存地址")
    parser.add_argument('--dataset_path', default='tools/train_data/image', type=str, help="數據集位置")
    parser.add_argument('--train_label_text_path', default='tools/train_data/train_label.txt', type=str, help="訓練標籤位置")
    parser.add_argument('--valid_label_text_path', default='tools/train_data/valid_label.txt', type=str, help="驗證標籤位置")
    parser.add_argument('--per_device_train_batch_size', default=16, type=int, help="train batch size")
    parser.add_argument('--per_device_eval_batch_size', default=8, type=int, help="eval batch size")
    parser.add_argument('--max_target_length', default=128, type=int, help="訓練文字字符數")

    parser.add_argument('--num_train_epochs', default=10, type=int, help="訓練epoch數")
    parser.add_argument('--eval_steps', default=500, type=int, help="模型評估間隔數")
    parser.add_argument('--save_steps', default=1000, type=int, help="模型保存間隔步數")


    args = parser.parse_args()
    print("train param")
    print(args)
    print("loading data .................")

    train_df = pd.read_csv(args.train_label_text_path,sep='\t',header=None)
    test_df = pd.read_csv(args.valid_label_text_path,sep='\t',header=None)
    print("train num:", len(train_df), "test num:", len(test_df))

    #圖像預處理
    tokenizer = TrOCRProcessor.from_pretrained("TrOCRProcessor/trocr-base-handwritten")
    vocab = tokenizer.tokenizer.get_vocab()
    vocab_inp = {vocab[key]: key for key in vocab}


    train_dataset = trocrDataset(root_dir=args.dataset_path,df=train_df ,Tokenizer=tokenizer, max_target_length=args.max_target_length)

    eval_dataset = trocrDataset(root_dir=args.dataset_path,df=test_df ,Tokenizer=tokenizer, max_target_length=args.max_target_length)



    model = VisionEncoderDecoderModel.from_pretrained("VisionEncoderDecoderModel/trocr-base-stage1")
    model.config.decoder_start_token_id = tokenizer.tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = tokenizer.tokenizer.sep_token_id
    model.config.max_length = 128
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    cer_metric = load_metric("cer")

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        fp16=True,
        output_dir=args.checkpoint_path,
        logging_steps=10,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=5
    )

    # seq2seq trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer.image_processor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )
    trainer.train()
    trainer.save_model(os.path.join(args.checkpoint_path, 'last'))
    tokenizer.save_pretrained(os.path.join(args.checkpoint_path, 'last'))
