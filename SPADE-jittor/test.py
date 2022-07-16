import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str)
parser.add_argument('--output_path', type=str)

args = parser.parse_args()

subprocess.call(f'python spade_test.py --name bs4vae --dataset_mode custom --label_dir {args.input_path} --label_nc 29 --no_instance --use_vae --which_epoch 60', shell=True)

import os
from shutil import copy, rmtree

file_path = './results/bs4vae/test_60/images/synthesized_image/'
save_dir = args.output_path

files = os.listdir(file_path)
for file in files:
    from_path = os.path.join(file_path, file)
    to_path = save_dir

    if not os.path.isdir(to_path):
        os.makedirs(to_path)

    copy(from_path, to_path)

rmtree('./results/bs4vae')
