import json
import os
import argparse
from glob import glob
import torch
import ipdb
import random

## define arguments, including the json file path, txt file path, and output path
parser = argparse.ArgumentParser(description='Generate pair index')
parser.add_argument('--data_dir', type=str, default='data.json', help='path to the data .torch')
parser.add_argument('--txt_file', type=str, default='data.txt', help='path to the txt file')
parser.add_argument('--output_path', type=str, default='output.json', help='path to the output file')
## add arguments for min matched num and max matched num
parser.add_argument('--min_num_matches', type=int, default=30, help='minimum number of matches')
parser.add_argument('--max_num_matches', type=int, default=1000000, help='maximum number of matches')
parser.add_argument('--filter_mode', type=str, choices=['max_match', 'each_context'], help='filter the data, only remains the max matched pairs for each context[0]')

args = parser.parse_args()

## read the json file
data_list = sorted(glob(os.path.join(args.data_dir, "*.torch")))
name_id_dict = {}
# ipdb.set_trace()
for data_file in data_list:
    load_data = torch.load(data_file)
    assert len(load_data) == 1
    data = load_data[0]
    for i, name in enumerate(data['image_names']):
        name_id_dict[name] = i
        
    
# with open(args.json_file, 'r') as f:
#     data = json.load(f)['frames']

# name_id_dict = {}
# for i, _data in enumerate(data):
#     image_name = _data['file_path'][7:]
#     ## NOTE colmap image id is different from the index in the json file, some images may deleted
#     name_id_dict[image_name] = i
    

output_dict = {}


counter = 0
with open(args.txt_file, "r") as file:
    for line in file:
        image_paths = line.strip().split(" ")
        if len(image_paths) == 3:
            num_matches, image_path1, image_path2 = image_paths
            num_matches = int(num_matches)
            if num_matches < args.min_num_matches or num_matches > args.max_num_matches:
                continue
            # Do something with the image paths
            print(f"Image path 1: {image_path1}, Image path 2: {image_path2}")
            if image_path1 not in name_id_dict or image_path2 not in name_id_dict:
                print("Invalid image path")
                continue
            image_id1 = name_id_dict[image_path1]
            image_id2 = name_id_dict[image_path2]
            file_dir_name = image_path1.split("/")[0]
            output_dict[f"{file_dir_name}_match_{num_matches}_{counter}"] = {
                "context": [image_id1, image_id2],
                "target": [image_id1, image_id2],
            }
            counter += 1
            
            
        else:
            print("Invalid line format")

print("Before filter Total number of inlier image pairs: ", counter)
context_dict = {}
save_dict = {}
if args.filter_mode == 'each_context':
    for k, v in output_dict.items():
        # num_matches = k.split("_")[-2]
        contextkeys = v["context"]
        file_dir_name = k.split("_match_")[0]
        for ck in contextkeys:
            contextkey = f"{file_dir_name}_context_{ck}"
            # ipdb.set_trace()
            if contextkey not in context_dict:
                context_dict[contextkey] = [k]
            else:
                context_dict[contextkey].append(k)
    # ipdb.set_trace()
    for k, v in context_dict.items():
        if len(v) > 1:
            ## random choose 2 items
            v_select = random.sample(v, 2)
            # max_match = max([int(_v.split("_")[-2]) for _v in v])
            for _v in v_select:
                save_dict[_v] = output_dict[_v]
    
                    
# ipdb.set_trace()

elif args.filter_mode == 'max_match':
    for k, v in output_dict.items():
        # num_matches = k.split("_")[-2]
        contextkey = v["context"][0]
        file_dir_name = k.split("_match_")[0]
        contextkey = f"{file_dir_name}_context_{contextkey}"
        # ipdb.set_trace()
        if contextkey not in context_dict:
            context_dict[contextkey] = [k]
        else:
            context_dict[contextkey].append(k)
    # ipdb.set_trace()
    for k, v in context_dict.items():
        if len(v) > 1:
            max_match = max([int(_v.split("_")[-2]) for _v in v])
            for _v in v:
                if int(_v.split("_")[-2]) != max_match:
                    output_dict.pop(_v)
                    counter -= 1
    
    save_dict = output_dict

save_counter = len(save_dict)
        
## save output dict to output file
with open(args.output_path, 'w') as f:
    json.dump(save_dict, f, indent=4)
    
    print(f"Exported {save_counter} inlier image pairs to {args.output_path}.")
    