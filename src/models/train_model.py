import subprocess
import os 
from pathlib import Path

class Trainer:
    """class to run shell script to train GAN models
    """
    def __init__(self, train_script = None, args_file=None):
        self.train_script = train_script
        self.args_file = args_file

    def get_args_from_file(self):
        args = []
        if self.args_file:
            with open(self.args_file, 'r') as f:
                args = f.read().splitlines()
        return args

    def train(self):
        args = self.get_args_from_file()
        argument = f"python {self.train_script} {' '.join(args)}"
        subprocess.run(argument, shell = True)

if __name__ == '__main__':
    train_script = "F:/road_shoulder_gan/src/models/pytorch-CycleGAN-and-pix2pix/train.py"
    # set args folder path
    args_folder_path = "F:/road_shoulder_gan/configs"
    # make a list of files that start with "train_" and end with ".txt" with pathlib
    args_file_list = [file for file in Path(args_folder_path).iterdir() if file.name.startswith("train_") and file.name.endswith(".txt")]
    # args_file_list = ["F:/road_shoulder_gan/configs/default_sidewalk_pix2pix.txt",
    #                   "F:/road_shoulder_gan/configs/default_sidewalk_cyclegan.txt",
    #                   "F:/road_shoulder_gan/configs/sidewalk_pix2pix_flip.txt",
    #                   "F:/road_shoulder_gan/configs/sidewalk_cyclegan_flip.txt"]
    for args_file in args_file_list:# Set the model directory path
        print(f"Working on {args_file}")
        model_directory_path = Path(f"F:/road_shoulder_gan/models/{Path(args_file).stem}")
        if not model_directory_path.exists():
            trainer = Trainer(train_script, args_file)
            trainer.train()
