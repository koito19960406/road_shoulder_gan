import subprocess
import os 
from pathlib import Path
import shutil

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
    # Make a list of drives
    drives = ['E:/', 'F:/']

    # Set relative paths
    script_path_relative = Path("road_shoulder_gan/src/models/pytorch-CycleGAN-and-pix2pix/train.py")
    args_folder_path_relative = Path("road_shoulder_gan/configs")
    models_folder_path_relative = Path("road_shoulder_gan/models")

    train_script = None
    args_folder_path = None
    models_folder_path = None

    for drive in drives:
        temp_script_path = Path(drive) / script_path_relative
        temp_args_folder_path = Path(drive) / args_folder_path_relative
        temp_models_folder_path = Path(drive) / models_folder_path_relative
        if temp_script_path.exists():
            train_script = temp_script_path
        if temp_args_folder_path.exists():
            args_folder_path = temp_args_folder_path
        if temp_models_folder_path.exists():
            models_folder_path = temp_models_folder_path

    if not train_script or not args_folder_path or not models_folder_path:
        print("Required directories not found in any of the drives.")
        exit()

    # make a list of files that start with "train_" and end with ".txt"
    args_file_list = [file for file in args_folder_path.iterdir() if file.name.startswith("train_") and file.name.endswith(".txt")]

    for args_file in args_file_list:  # Set the model directory path
        print(f"Working on {args_file}")
        # model_directory_path = models_folder_path / Path(args_file.stem.replace("train_", ""))
        # if not model_directory_path.exists():
        
        trainer = Trainer(str(train_script), str(args_file))
        trainer.train()

        # Use the shutil.move() function to move the file
        shutil.move(args_file, args_file.parent / "done" / args_file.name)

    print("Done!")