import argparse
import gzip
import json
import os
import csv

import h5py
import numpy as np
from PIL import Image
from torchvision import transforms

from models.CNN import ResNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-image_dir", type=str, default="./data/img/raw/train2014")
    parser.add_argument("-training_set", type=str, default="./data/guesswhat.train.jsonl.gz")
    parser.add_argument("-validation_set", type=str, default="./data/guesswhat.valid.jsonl.gz")
    parser.add_argument("-test_set", type=str, default="./data/guesswhat.test.jsonl.gz")
    parser.add_argument("-objects_features_index_path", type=str, default="./data/objects_features_index_example.json")
    parser.add_argument("-storage_path", type=str, default="data")
    parser.add_argument("-objects_features_path", type=str, default="./data/objects_features_example.h5")
    parser.add_argument("-only_mapping", action='store_true', help='only compute the mapping files (json/csv)')
    args = parser.parse_args()

    games = []

    print("Loading file: {}".format(args.training_set))
    with gzip.open(args.training_set) as file:
        for json_game in file:
            games.append(json.loads(json_game.decode("utf-8")))

    print("Loading file: {}".format(args.validation_set))
    with gzip.open(args.validation_set) as file:
        for json_game in file:
            games.append(json.loads(json_game.decode("utf-8")))

    print("Loading file: {}".format(args.test_set))
    with gzip.open(args.test_set) as file:
        for json_game in file:
            games.append(json.loads(json_game.decode("utf-8")))

    model = ResNet()
    model.eval()
    model.cuda()

    batch_size = 20

    game_id2pos = {}

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )

    if args.only_mapping:
        for i, game in enumerate(games):
            print("\rProcessing image [{}/{}]".format(i, len(games)), end="")
            game_id2pos[str(game["dialogue_id"])] = i
        with open(os.path.join(args.storage_path, f'objects_features_index_example.csv'), 'w') as f:
            writer = csv.writer(f)
            for key, val in game_id2pos.items():
                writer.writerow([key, val])
        exit('new mapping created...')

    for i in range(0, len(games), batch_size):
        game_batch = games[i: i + batch_size]
        avg_img_features = np.zeros((len(game_batch), 20, 2048))
        

        for game_index, game in enumerate(game_batch):
            print("\rProcessing image [{}/{}]".format(game_index+i, len(games)), end="")

            image = Image.open(os.path.join(args.image_dir, game["picture"]["file_name"])).convert("RGB")
            game_id2pos[str(game["dialogue_id"])] = game_index

            for object_index, object_id in enumerate(game["objects"]):
                object = game["objects"][object_id]
                bbox = object["bbox"]
                cropped_image = image.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
                cropped_image_tensor = transform(cropped_image)
                cropped_image_tensor = cropped_image_tensor.view(1, 3, 224, 224)
                conv_features, feat = model(cropped_image_tensor.cuda())
                avg_img_features[game_index][object_index] = feat.cpu().data.numpy()

        # print("Saving file: {}".format(args.objects_features_path))
        if i == 0:
            objects_features_h5 = h5py.File(args.objects_features_path, "w")
            objects_features_h5.create_dataset(
                                name="objects_features", 
                                dtype="float32", 
                                data=avg_img_features, 
                                maxshape = (None,20,2048))
        else:
            objects_features_h5 = h5py.File(args.objects_features_path, "a")
            objects_features_h5["objects_features"].resize((
                objects_features_h5["objects_features"].shape[0] + avg_img_features.shape[0]), 
                axis = 0)
            objects_features_h5["objects_features"][-avg_img_features.shape[0]:] = avg_img_features
            
            # objects_features_h5.create_dataset(name="objects_features", dtype="float32", data=avg_img_features)
        
        objects_features_h5.close()

        # print("Saving file: {}".format(args.objects_features_index_path))

    with open(os.path.join(args.storage_path, f'objects_features_index_example.csv'), 'w') as f:
        writer = csv.writer(f)
        for key, val in game_id2pos.items():
            writer.writerow([key, val])
