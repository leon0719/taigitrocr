
from typing import List
import time
def load_dict(path: str) -> List[str]:
    word_dict = []
    with open(path,"r",encoding="utf-8") as d:
        word_dict = [l for l in d.read().splitlines() if len(l) > 0]

    return word_dict

if __name__ == "__main__":
    start =time.time()

    train_data = '/nfs/TS-1635AX/WorkSpace/leon/GPU5/train_data/train_label.txt'

    line_list = load_dict(train_data)

    characters = sorted(list(set(''.join(line_list))))

    with open('/home/leon/trocr/dicts/vocab.txt','w',encoding='utf-8') as f:
        for word in characters:
            if not word =='	':
                f.write(word+'\n')
    end =time.time()
    print('total cost :',round(end-start,2))
