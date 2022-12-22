import os
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
from utiils import *

text_gerner = '/home/leon/trocr/trdg'
# 設定訓練資料圖片保存的資料夾的"絕對路徑"
save_img_path = '/home/leon/trocr/tools'

# --------------------------------------------------
train_data_path = os.path.join(save_img_path, 'train_data')
train_path = os.path.join(train_data_path, 'image')

os.makedirs(train_data_path,exist_ok=True)
os.makedirs(train_path,exist_ok=True)

def Gerner_en_img():
    data_en_path = os.path.join(save_img_path, 'en')
    os.makedirs(data_en_path,exist_ok=True)

    os.chdir(text_gerner)

    print('生成en_img_data')
    os.system(
        f'python run.py  \
            -c 10000 \
            -d 3 \
            -f 48 \
            -i dicts/en/en_article.txt \
            -fd fonts/en/ \
            -t $(cat /proc/cpuinfo | grep "processor" |  wc -l)\
            -id images/en --output_dir {data_en_path}'
    )
    # -----------------------------------------------------------
    os.system(
        f'python run.py  \
            -c 10000 \
            -d 3 \
            -f 48 \
            -i dicts/en/en_article2.txt \
            -fd fonts/en/ \
            -t $(cat /proc/cpuinfo | grep "processor" |  wc -l)\
            -id images/en --output_dir {data_en_path}'
    )
    # -----------------------------------------------------------
    os.system(
        f'python run.py  \
            -c 10000 \
            -f 48 \
            -i dicts/en/en_article3.txt \
            -fd fonts/en/ \
            -t $(cat /proc/cpuinfo | grep "processor" |  wc -l)\
            -id images/en --output_dir {data_en_path}'
    )
    # -----------------------------------------------------------
    os.system(
        f'python run.py  \
            -c 10000 \
            -f 48 \
            -i dicts/en/en_article4.txt \
            -fd fonts/en/ \
            -t $(cat /proc/cpuinfo | grep "processor" |  wc -l)\
            -id images/en --output_dir {data_en_path}'
    )
    # -----------------------------------------------------------
    for file in os.listdir(data_en_path):
        if file.startswith(' '):
            os.rename(
                os.path.join(data_en_path, file),
                os.path.join(data_en_path, file[1:].replace(' ', '_'))
                    )
        else:
            os.rename(
                os.path.join(data_en_path, file),
                os.path.join(data_en_path, file.replace(' ', '_'))
                    )

    data_en = os.listdir(data_en_path)
    train_en, valid_en = train_test_split(
        data_en, test_size=0.1, random_state=0)

    train_img_label(train_data_path,train_en,valid_en)
    print('移動資料夾...')
    for file in tqdm(data_en):
        shutil.move(os.path.join(data_en_path, file),
                    os.path.join(train_path, file))

    os.rmdir(data_en_path)
def Gerner_ch_img():
    data_ch_path = os.path.join(save_img_path, 'ch')
    os.makedirs(data_ch_path,exist_ok=True)

    os.chdir(text_gerner)

    print('生成ch_img_data')
    os.system(
        f'python run.py  \
            -c 100 \
            -f 48 \
            -d 3 \
            -i dicts/ch/ch_article.txt \
            -fd fonts/ch/ \
            -t $(cat /proc/cpuinfo | grep "processor" |  wc -l)\
            -id images/Hanji --output_dir {data_ch_path}'
    )
    # -----------------------------------------------------------
    os.system(
        f'python run.py  \
            -c 100 \
            -f 48 \
            -d 3 \
            -i dicts/ch/ch_article2.txt \
            -fd fonts/ch/ \
            -t $(cat /proc/cpuinfo | grep "processor" |  wc -l)\
            -id images/Hanji --output_dir {data_ch_path}'
    )
    # -----------------------------------------------------------
    os.system(
        f'python run.py  \
            -c 100 \
            -f 48 \
            -i dicts/ch/ch_article3.txt \
            -fd fonts/ch/ \
            -t $(cat /proc/cpuinfo | grep "processor" |  wc -l)\
            -id images/Hanji --output_dir {data_ch_path}'
    )
    # -----------------------------------------------------------
    os.system(
        f'python run.py  \
            -c 100 \
            -f 48 \
            -i dicts/ch/ch_article4.txt \
            -fd fonts/ch/ \
            -t $(cat /proc/cpuinfo | grep "processor" |  wc -l)\
            -id images/Hanji --output_dir {data_ch_path}'
    )
    # -----------------------------------------------------------
    for file in os.listdir(data_ch_path):
        if file.startswith(' '):
            os.rename(
                os.path.join(data_ch_path, file),
                os.path.join(data_ch_path, file[1:].replace(' ', '_'))
                    )
        else:
            os.rename(
                os.path.join(data_ch_path, file),
                os.path.join(data_ch_path, file.replace(' ', '_'))
                    )

    data_ch = os.listdir(data_ch_path)
    train_ch, valid_ch = train_test_split(
        data_ch, test_size=0.1, random_state=0)

    train_img_label(train_data_path,train_ch,valid_ch)
    print('移動資料夾...')
    for file in tqdm(data_ch):
        shutil.move(os.path.join(data_ch_path, file),
                    os.path.join(train_path, file))

    os.rmdir(data_ch_path)
def Gerner_POJ_img():
    data_POJ_path = os.path.join(save_img_path, 'POJ')
    os.makedirs(data_POJ_path,exist_ok=True)

    os.chdir(text_gerner)

    print('生成POJ_img_data')
    os.system(
        f'python run.py  \
            -c 100 \
            -f 48 \
            -d 3 \
            -i dicts/POJ/POJ_corpus.txt \
            -fd fonts/POJ/ \
            -t $(cat /proc/cpuinfo | grep "processor" |  wc -l)\
            -id images/POJ --output_dir {data_POJ_path}'
    )
    # -----------------------------------------------------------
    save_img_path2 = f'{save_img_path}/POJ2'
    os.chdir(text_gerner)
    os.system(
        f'python run.py  \
            -c 100 \
            -f 48 \
            -d 3 \
            -i dicts/POJ/POJ_corpus2.txt \
            -fd fonts/POJ/ \
            -t $(cat /proc/cpuinfo | grep "processor" |  wc -l)\
            -id images/POJ --output_dir {save_img_path2}'
    )

    dilate(save_img_path2, data_POJ_path)

    shutil.rmtree(save_img_path2)
    # -----------------------------------------------------------
    os.chdir(text_gerner)
    save_img_path3 = f'{save_img_path}/POJ3'
    os.system(
        f'python run.py  \
            -c 100 \
            -f 48 \
            -i dicts/POJ/POJ_corpus3.txt \
            -fd fonts/POJ/ \
            -t $(cat /proc/cpuinfo | grep "processor" |  wc -l)\
            -id images/POJ --output_dir {save_img_path}/POJ3'
    )
    closing(save_img_path3, data_POJ_path)

    shutil.rmtree(save_img_path3)
    # -----------------------------------------------------------
    os.chdir(text_gerner)

    os.system(
        f'python run.py  \
            -c 100 \
            -f 48 \
            -i dicts/POJ/POJ_corpus4.txt \
            -fd fonts/POJ/ \
            -t $(cat /proc/cpuinfo | grep "processor" |  wc -l)\
            -id images/POJ --output_dir {data_POJ_path}'
    )
    # -----------------------------------------------------------
    for file in os.listdir(data_POJ_path):
        if file.startswith(' '):
            os.rename(
                os.path.join(data_POJ_path, file),
                os.path.join(data_POJ_path, file[1:].replace(' ', '_'))
                    )
        else:
            os.rename(
                os.path.join(data_POJ_path, file),
                os.path.join(data_POJ_path, file.replace(' ', '_'))
                    )

    data_POJ = os.listdir(data_POJ_path)
    train_POJ, valid_POJ = train_test_split(
        data_POJ, test_size=0.1, random_state=0)

    train_img_label(train_data_path,train_POJ,valid_POJ)
    print('移動資料夾...')
    for file in tqdm(data_POJ):
        shutil.move(os.path.join(data_POJ_path, file),
                    os.path.join(train_path, file))

    os.rmdir(data_POJ_path)
def Gerner_TAI_LO_img():
    data_TAI_LO_path = os.path.join(save_img_path, 'TAI_LO')
    os.makedirs(data_TAI_LO_path,exist_ok=True)

    os.chdir(text_gerner)
    print('生成TAI_LO_img_data')
    os.system(
        f'python run.py  \
            -c 100 \
            -f 48 \
            -i dicts/TAI_LO/TAI_LO_corpus.txt \
            -fd fonts/TAI_LO/ \
            -t $(cat /proc/cpuinfo | grep "processor" |  wc -l)\
            -id images/POJ --output_dir {data_TAI_LO_path}'
    )
    # -----------------------------------------------------------
    os.system(
        f'python run.py  \
            -c 100 \
            -f 48 \
            -i dicts/TAI_LO/TAI_LO_corpus2.txt \
            -fd fonts/TAI_LO/ \
            -t $(cat /proc/cpuinfo | grep "processor" |  wc -l)\
            -id images/POJ --output_dir {data_TAI_LO_path}'
    )
    # -----------------------------------------------------------
    os.system(
        f'python run.py  \
            -c 100 \
            -f 48 \
            -i dicts/TAI_LO/TAI_LO_corpus3.txt \
            -fd fonts/TAI_LO/ \
            -t $(cat /proc/cpuinfo | grep "processor" |  wc -l)\
            -id images/POJ --output_dir {data_TAI_LO_path}'
    )
    # -----------------------------------------------------------
    os.system(
        f'python run.py  \
            -c 100 \
            -f 48 \
            -i dicts/TAI_LO/TAI_LO_corpus4.txt \
            -fd fonts/TAI_LO/ \
            -t $(cat /proc/cpuinfo | grep "processor" |  wc -l)\
            -id images/POJ --output_dir {data_TAI_LO_path}'
    )
    # -----------------------------------------------------------
    for file in os.listdir(data_TAI_LO_path):
        if file.startswith(' '):
            os.rename(
                os.path.join(data_TAI_LO_path, file),
                os.path.join(data_TAI_LO_path, file[1:].replace(' ', '_'))
                    )
        else:
            os.rename(
                os.path.join(data_TAI_LO_path, file),
                os.path.join(data_TAI_LO_path, file.replace(' ', '_'))
                    )

    data_TAI_LO = os.listdir(data_TAI_LO_path)
    train_TAI_LO, valid_TAI_LO = train_test_split(
        data_TAI_LO, test_size=0.1, random_state=0)

    train_img_label(train_data_path,train_TAI_LO,valid_TAI_LO)
    print('移動資料夾...')
    for file in tqdm(data_TAI_LO):
        shutil.move(os.path.join(data_TAI_LO_path, file),
                    os.path.join(train_path, file))

    os.rmdir(data_TAI_LO_path)
def Gerner_HAN_LO_img():
    data_HAN_LO_path = os.path.join(save_img_path, 'HAN_LO')
    os.makedirs(data_HAN_LO_path,exist_ok=True)

    os.chdir(text_gerner)
    print('生成HAN_LO_img_data')
    os.system(
        f'python run.py  \
            -c 100 \
            -f 48 \
            -d 3 \
            -i dicts/HAN_LO/HAN_LO_corpus.txt \
            -fd fonts/HAN_LO/ \
            -t $(cat /proc/cpuinfo | grep "processor" |  wc -l)\
            -id images/HAN_LO --output_dir {data_HAN_LO_path}'
    )
    # -----------------------------------------------------------
    os.system(
        f'python run.py  \
            -c 100 \
            -f 48 \
            -d 3 \
            -i dicts/HAN_LO/HAN_LO_corpus2.txt \
            -fd fonts/HAN_LO/ \
            -t $(cat /proc/cpuinfo | grep "processor" |  wc -l)\
            -id images/HAN_LO --output_dir {data_HAN_LO_path}'
    )

    # -----------------------------------------------------------
    os.system(
        f'python run.py  \
            -c 100 \
            -f 48 \
            -i dicts/HAN_LO/HAN_LO_corpus3.txt \
            -fd fonts/HAN_LO/ \
            -t $(cat /proc/cpuinfo | grep "processor" |  wc -l)\
            -id images/HAN_LO --output_dir {data_HAN_LO_path}'
    )

    # -----------------------------------------------------------
    os.system(
        f'python run.py  \
            -c 100 \
            -f 48 \
            -i dicts/HAN_LO/HAN_LO_corpus4.txt \
            -fd fonts/HAN_LO/ \
            -t $(cat /proc/cpuinfo | grep "processor" |  wc -l)\
            -id images/HAN_LO --output_dir {data_HAN_LO_path}'
    )
    # -----------------------------------------------------------
    for file in os.listdir(data_HAN_LO_path):
        if file.startswith(' '):
            os.rename(
                os.path.join(data_HAN_LO_path, file),
                os.path.join(data_HAN_LO_path, file[1:].replace(' ', '_'))
                    )
        else:
            os.rename(
                os.path.join(data_HAN_LO_path, file),
                os.path.join(data_HAN_LO_path, file.replace(' ', '_'))
                    )

    data_HAN_LO = os.listdir(data_HAN_LO_path)
    train_HAN_LO, valid_HAN_LO = train_test_split(
        data_HAN_LO, test_size=0.1, random_state=0)

    train_img_label(train_data_path,train_HAN_LO,valid_HAN_LO)
    print('移動資料夾...')
    for file in tqdm(data_HAN_LO):
        shutil.move(os.path.join(data_HAN_LO_path, file),
                    os.path.join(train_path, file))

    os.rmdir(data_HAN_LO_path)
def Rename_ImageName():
    train_label =load_dict(os.path.join(train_data_path,'train_label.txt'))
    valid_label =load_dict(os.path.join(train_data_path,'valid_label.txt'))
    count = 1
    for img in train_label:
        os.rename(os.path.join(train_path,img[0]),
        os.path.join(os.path.join(train_path,f'train_img_{count}.jpg'))
        )
        count += 1
    count = 1
    for img in valid_label:
        os.rename(os.path.join(train_path,img[0]),
        os.path.join(os.path.join(train_path,f'valid_img_{count}.jpg'))
        )
        count += 1
    with open(os.path.join(train_data_path,'train_label.txt'),'w') as f:
        count = 1
        for line in train_label:
            f.write(f'train_img_{count}.jpg' + '\t' +line[1] + '\n' )
            count += 1
    with open(os.path.join(train_data_path,'valid_label.txt'),'w') as f:
        count = 1
        for line in valid_label:
            f.write(f'valid_img_{count}.jpg' + '\t' +line[1] + '\n' )
            count += 1




def main():
    Gerner_en_img()
    #Gerner_ch_img()
    #Gerner_POJ_img()
    #Gerner_TAI_LO_img()
    #Gerner_HAN_LO_img()

    Rename_ImageName()




if __name__ == '__main__':
    main()
