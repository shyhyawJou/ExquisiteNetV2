import pickle
from pathlib import Path as p
import shutil
from PIL import Image
from os.path import join as pj



def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def ds(data_dir, new_dir):
    if p(new_dir).exists():
        shutil.rmtree(new_dir)
    p(new_dir).mkdir()

    label_name = unpickle(p(data_dir)/"batches.meta")[b"label_names"]
    img_No = 0
    for batch_file in p(data_dir).glob("*"):
        if len(batch_file.name.split('_')) == 1:
            continue
        if batch_file.name.split('_')[1] == "batch":
            if batch_file.name == "test_batch":
                dir_name = "test"
            else:
                dir_name = "train"
            data_dict = unpickle(batch_file)
            for label, img, img_name in zip(*list(data_dict.values())[1:]):
                img = Image.fromarray(img.reshape(3,32,32).transpose(1,2,0))
                class_name = label_name[label]
                class_name, img_name = class_name.decode("utf-8"), img_name.decode("utf-8")
                dst = p(pj(new_dir, dir_name, class_name, img_name)).with_suffix(".bmp")
                dst.parent.mkdir(parents=True, exist_ok=True)
                img.save(dst)
                img_No += 1
                print(img_No, end='\r')



if __name__ == "__main__":
    ds("cifar-10-batches-py", "cifar10")
