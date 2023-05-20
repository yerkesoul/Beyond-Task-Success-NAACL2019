import argparse
import json
import csv
import os.path
from time import time

import h5py
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models.CNN import ResNet
from utils.wrap_var import to_var


def extract_features(img_dir, model, img_list, my_cpu=False):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    if my_cpu:
        avg_img_features = np.zeros((5, 2048))
    else:
        avg_img_features = np.zeros((len(img_list), 2048))

    name2id = dict()
    print('creating features ....')
    l = len(img_list)
    for i in range(l):
        print("\rProcessing image [{}/{}]".format(i, l), end="")
        if i >= 5 and my_cpu:
            break
        ImgTensor = transform(Image.open(os.path.join(img_dir, img_list[i])).convert('RGB'))
        ImgTensor = to_var(ImgTensor.view(1, 3, 224, 224))
        conv_features, feat = model(ImgTensor)
        avg_img_features[i] = feat.cpu().data.numpy()
        name2id[img_list[i]] = i

    return avg_img_features, name2id

def img2id(img_dir, img_list, my_cpu=False):
    name2id = dict()
    print('creating features ....')
    l = len(img_list)
    for i in range(l):
        print("\rProcessing image [{}/{}]".format(i, l), end="")
        name2id[img_list[i]] = i
    return name2id


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-image_dir", type=str, default="data/img/raw/train2014",
                        help="this directory should contain both train and val images")
    parser.add_argument("-n2n_train_set", type=str, default="data/n2n_train_successful_data.json")
    parser.add_argument("-n2n_val_set", type=str, default="data/n2n_val_all_gameplay_data.json")
    parser.add_argument("-n2n_test_set", type=str, default="data/n2n_test_all_gameplay_data.json")
    parser.add_argument("-image_features_json_path", type=str, default="data/ResNet_avg_image_features2id.json")
    parser.add_argument("-storage_path", type=str, default="data")
    parser.add_argument("-image_features_path", type=str, default="data/ResNet_avg_image_features.h5")
    parser.add_argument("-only_json", action='store_true', help='only compute the img2id.json dictionary again')
    args = parser.parse_args()
    start = time()
    print('Start')
    splits = ['train', 'val', 'test']
    # splits = ['val']

    my_cpu = False

    with open(args.n2n_train_set, 'r') as file_v:
        n2n_data = json.load(file_v)
    images = {'train': [], 'val': [], 'test': []}
    for k, v in n2n_data.items():
        images['train'].append(v['image_file'])

    with open(args.n2n_val_set, 'r') as file_v:
        n2n_data = json.load(file_v)
    for k, v in n2n_data.items():
        images['val'].append(v['image_file'])

    with open(args.n2n_test_set, 'r') as file_v:
        n2n_data = json.load(file_v)
    for k, v in n2n_data.items():
        images['test'].append(v['image_file'])

    model = ResNet()
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    
    # feat_h5_file = h5py.File(args.image_features_path, 'a')    

    mapping_data = dict()

    if args.only_json:
        for split in splits:
            name2id = img2id(args.image_dir, img_list=images[split], my_cpu=my_cpu)
            mapping_data = name2id
            with open(os.path.join(args.storage_path, f'ResNet_{split}_image_features2id.csv'), 'w') as f:
                writer = csv.writer(f)
                for key, val in mapping_data.items():
                    writer.writerow([key, val])
        print('new dictionary created.')
        print('Time taken: ', time() - start)

    else:
        if os.path.isfile(args.image_features_path):
            feat_h5_file = h5py.File(args.image_features_path, 'a')
        else:
            feat_h5_file = h5py.File(args.image_features_path, 'w')
        for split in splits:
            print(split)
            avg_img_features, name2id = extract_features(args.image_dir, model, img_list=images[split], my_cpu=my_cpu)
            try:
                del feat_h5_file[split + '_img_features']
                print('\noverwriting ' + split + ' dataset...')
            except:
                print('\ncreating ' + split + ' dataset...')
            feat_h5_file.create_dataset(name=split + '_img_features', dtype='float32', data=avg_img_features)
            mapping_data = name2id
            with open(os.path.join(args.storage_path, f'ResNet_{split}_image_features2id.csv'), 'w') as f:
                writer = csv.writer(f)
                for key, val in mapping_data.items():
                    writer.writerow([key, val])
        feat_h5_file.close()

        

        print('Image Features extracted.')
        print('Time taken: ', time() - start)