import cv2
import argparse
import shutil
from tqdm import tqdm
from glob import glob
import os.path as osp
import numpy as np
import tempfile
import zipfile
from mmengine.utils import mkdir_or_exist


"""
(1) 按照论文实验设置拆分数据集: 8:1:1
(2) mask图像0-255 ---> 0-1 (mmsegmentation format)

dataset-path: PolypDB.zip
out_dir:      data/PolypDB
"""

def pares_args():
    parser = argparse.ArgumentParser(
        description='Convert polypdb dataset to mmsegmentation format')
    parser.add_argument(
        '--dataset-path', type=str, default="PolypDB.zip", help='polypdb dataset path.')
    parser.add_argument(
        '--out_dir',
        default='data/PolypDB',
        type=str,
        help='save path of the dataset.')
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    args = parser.parse_args()
    return args

def get_data(path):
    samples = []

    images = sorted(glob(osp.join(path, "images", "*.jpg")))
    image_names = [osp.splitext(osp.basename(file))[0] for file in images]

    for image_name in image_names:
        image = osp.join(path, "images", f"{image_name}.jpg")

        mask_jpg = osp.join(path, "masks", f"{image_name}.jpg")
        mask_png = osp.join(path, "masks", f"{image_name}.png")
        # 判断 .jpg 掩码文件是否存在，否则使用 .png
        if osp.exists(mask_png):
            mask = mask_png
        elif osp.exists(mask_jpg):
            mask = mask_jpg
        else:
            # 如果掩码文件不存在，跳过该样本
            continue

        samples.append((image, mask))

    return samples

def load_polypdb_wli_data(path):
    """ Training data """
    modality_data = get_data(path)
    modality_len = len(modality_data)
    modality_train_len = int(0.8 * modality_len)
    modality_val_len = int(0.1 * modality_len)
    train_samples = modality_data[:modality_train_len]
    valid_samples = modality_data[modality_train_len:modality_train_len + modality_val_len]
    test_samples = modality_data[modality_train_len + modality_val_len:]

    return [train_samples, valid_samples, test_samples]

def mv_binary_mask(image, mask, img_save, ann_save, threshold=127):
    mask_ = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取
    if mask_ is None:
        print(f"警告: 无法读取文件 {mask}，跳过")
        return
    
    binary_mask = np.where(mask_ > threshold, 1, 0).astype(np.uint8)
    mask_name = osp.basename(mask).replace('.jpg', '.png').replace('.jpeg', '.png')
    cv2.imwrite(osp.join(ann_save, mask_name), binary_mask)
    shutil.move(image, img_save)
    # print(f"已处理: {image}")


def main():
    args = pares_args()
    dataset_path = args.dataset_path
    out_dir = args.out_dir

    if not osp.exists(dataset_path):
        raise ValueError('The dataset path does not exist. '
                         'Please enter a correct dataset path.')
    
    print('Making directories...')
    mkdir_or_exist(out_dir)

    modality_train_img = osp.join(out_dir, 'modality_wise', 'train', 'images')
    modality_train_mask = osp.join(out_dir, 'modality_wise', 'train', 'masks')
    modality_val_img = osp.join(out_dir, 'modality_wise', 'val', 'images')
    modality_val_mask = osp.join(out_dir, 'modality_wise', 'val', 'masks')
    modality_test_img = osp.join(out_dir, 'modality_wise', 'test', 'WLI', 'images')
    modality_test_mask = osp.join(out_dir, 'modality_wise', 'test', 'WLI', 'masks')

    center_train_img = osp.join(out_dir, 'center_wise', 'train', 'images')
    center_train_mask = osp.join(out_dir, 'center_wise', 'train', 'masks')
    center_val_img = osp.join(out_dir, 'center_wise', 'val', 'images')
    center_val_mask = osp.join(out_dir, 'center_wise', 'val', 'masks')
    center_test_img = osp.join(out_dir, 'center_wise', 'test', 'Simula', 'WLI', 'images')
    center_test_mask = osp.join(out_dir, 'center_wise', 'test', 'Simula', 'WLI', 'masks')

    mkdir_or_exist(modality_train_img)
    mkdir_or_exist(modality_train_mask)
    mkdir_or_exist(modality_val_img)
    mkdir_or_exist(modality_val_mask)
    mkdir_or_exist(modality_test_img)
    mkdir_or_exist(modality_test_mask)
    mkdir_or_exist(center_train_img)
    mkdir_or_exist(center_train_mask)
    mkdir_or_exist(center_val_img)
    mkdir_or_exist(center_val_mask)
    mkdir_or_exist(center_test_img)
    mkdir_or_exist(center_test_mask)

    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
        print('Extracting PolypDB.zip...')
        zip_file = zipfile.ZipFile(dataset_path)
        zip_file.extractall(tmp_dir)
        
        print("Process PolypDB_modality_wise data...")
        mpath = osp.join(tmp_dir, 'PolypDB', 'PolypDB_modality_wise')
        train_data, val_data, test_data = load_polypdb_wli_data(osp.join(mpath, 'WLI'))

        for img, mask in tqdm(train_data):
            mv_binary_mask(img, mask, modality_train_img, modality_train_mask)

        for img, mask in tqdm(val_data):
            mv_binary_mask(img, mask, modality_val_img, modality_val_mask)
        
        for img, mask in tqdm(test_data):
            mv_binary_mask(img, mask, modality_test_img, modality_test_mask)

        for i in ['BLI', 'FICE', 'LCI', 'NBI']:
            path = osp.join(mpath, i)
            samples = get_data(path)
            test_img = osp.join(out_dir, 'modality_wise', 'test', i, 'images')
            test_mask = osp.join(out_dir, 'modality_wise', 'test', i, 'masks')
            mkdir_or_exist(test_img)
            mkdir_or_exist(test_mask)
            for img, mask in tqdm(samples):
                mv_binary_mask(img, mask, test_img, test_mask)


        print("Process PolypDB_center_wise data...")
        cpath = osp.join(tmp_dir, 'PolypDB', 'PolypDB_center_wise')
        train_data, val_data, test_data = load_polypdb_wli_data(osp.join(cpath, 'Simula', 'WLI'))
        
        for img, mask in tqdm(train_data):
            mv_binary_mask(img, mask, center_train_img, center_train_mask)

        for img, mask in tqdm(val_data):
            mv_binary_mask(img, mask, center_val_img, center_val_mask)
        
        for img, mask in tqdm(test_data):
            mv_binary_mask(img, mask, center_test_img, center_test_mask)
        
        for i in ['BKAI', 'Karolinska']:
            path = osp.join(cpath, i, 'WLI')
            samples = get_data(path)
            test_img = osp.join(out_dir, 'center_wise', 'test', i, 'WLI', 'images')
            test_mask = osp.join(out_dir, 'center_wise', 'test', i, 'WLI', 'masks')
            mkdir_or_exist(test_img)
            mkdir_or_exist(test_mask)
            for img, mask in tqdm(samples):
                mv_binary_mask(img, mask, test_img, test_mask)

        print('Removing the temporary files...')

    print('Done!')


if __name__ == "__main__":
    main()
