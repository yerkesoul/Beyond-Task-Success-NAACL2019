import os
import json


def get_unsuccessful(split):
    filename = split + '_results.json'
    path = 'logs/GamePlay/results/' + filename
    with open(path, 'r') as f:
        data = json.load(f)
    unsuccessful = []
    for key in data:
        if data[key]['guess_id'] != data[key]['target_id']:
            unsuccessful.append({
                'dialogue_id': key,
                'generated_dialogue': data[key]['gen_dialogue'],
                'image': data[key]['image'],
                'target': data[key]['target_cat_str'],
                'guess': data[key]['guess_cat_str']
            })
    return unsuccessful
    

if __name__ == '__main__':
    '''
    Extarcts the unsuccessful dialogues from the experimental results. 
    For now it only looks for file in the logs/GamePlay/results folder. The files should be named
    val_results.json and test_results.json. The resulting files will be named val_unsuccessful_dialogue.json and 
    test_unsuccessful_dialogue.json.

    If you want to look for files somewhere else or want to specify files by command line argument you will have to
    alter the code.

    '''

    splits = ['val', 'test']
    
    for split in splits:
        data = get_unsuccessful(args, split)

        with open('logs/GamePlay/results/' + split + '_unsuccessful_dialogue.json', 'w') as f:
            json.dump(data, f)