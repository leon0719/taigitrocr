import os
import json
import argparse
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import AutoConfig


def read_vocab(vocab_path):
    """
    讀取自定義訓練字符集
    vocab_path format:
    1\n
    2\n
    ...
    我\n
    """
    other = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    vocab = {}
    for ot in other:
        vocab[ot] = len(vocab)
    #{'<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3, '<mask>': 4}

    with open(vocab_path) as f:
        for line in f:
            line = line.strip('\n')
            if line not in vocab:
                vocab[line] = len(vocab)
    return vocab


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='trocr fine-tune訓練')

    parser.add_argument('--cust_vocab', default="dicts/total_dic.txt", type=str, help="自定義訓練數字符集")
    parser.add_argument('--cust_data_init_weights_path', default='./weights', type=str,
                        help="初始化訓練權重，用於自己數據集上fine-tune權重")
    args = parser.parse_args()

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
    pre_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
    pre_vocab = processor.tokenizer.get_vocab()
    cust_vocab = read_vocab(args.cust_vocab)

    keep_tokens = []
    unk_index = pre_vocab.get('<unk>') #-> 3
    for key in cust_vocab:
        keep_tokens.append(pre_vocab.get(key, unk_index))

    processor.save_pretrained(args.cust_data_init_weights_path)

    pre_model.save_pretrained(args.cust_data_init_weights_path)

    #替換詞庫
    with open(os.path.join(args.cust_data_init_weights_path, "vocab.json"), "w") as f:
        f.write(json.dumps(cust_vocab, ensure_ascii=False))
    #替換模型參數
    with open(os.path.join(args.cust_data_init_weights_path, "config.json")) as f:
        model_config = json.load(f)

    #替換roberta embedding層詞庫
    model_config["decoder"]['vocab_size'] = len(cust_vocab)

    #替換 attetion 字庫
    model_config['vocab_size'] = len(cust_vocab)

    with open(os.path.join(args.cust_data_init_weights_path, "config.json"), "w") as f:
        f.write(json.dumps(model_config, ensure_ascii=False))

    cust_config = AutoConfig.from_pretrained(args.cust_data_init_weights_path)
    cust_model = VisionEncoderDecoderModel(cust_config)

    pre_model_weigths = pre_model.state_dict()
    cust_model_weigths = cust_model.state_dict()

    #權重初始化
    print("loading init weights..................")
    for key in pre_model_weigths:
        print("name:", key)
        if pre_model_weigths[key].shape != cust_model_weigths[key].shape:
            wt = pre_model_weigths[key][keep_tokens, :]
            cust_model_weigths[key] = wt
        else:
            cust_model_weigths[key] = pre_model_weigths[key]

    cust_model.load_state_dict(cust_model_weigths)
    cust_model.save_pretrained(args.cust_data_init_weights_path)