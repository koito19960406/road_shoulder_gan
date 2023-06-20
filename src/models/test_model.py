import subprocess
import os 
from pathlib import Path

class Tester:
    """class to run shell script to test GAN models
    """
    def __init__(self, test_script = None, args_file=None):
        self.test_script = test_script
        self.args_file = args_file

    def get_args_from_file(self):
        args = []
        if self.args_file:
            with open(self.args_file, 'r') as f:
                args = f.read().splitlines()
        return args

    def test(self):
        args = self.get_args_from_file()
        argument = f"python {self.test_script} {' '.join(args)}"
        subprocess.run(argument, shell = True)

if __name__ == '__main__':
    # Make a list of drives
    drives = ['E:/', 'F:/']

    # Set relative paths
    script_path_relative = Path("road_shoulder_gan/src/models/pytorch-CycleGAN-and-pix2pix/test.py")
    args_folder_path_relative = Path("road_shoulder_gan/configs")
    models_folder_path_relative = Path("road_shoulder_gan/models")

    test_script = None
    args_folder_path = None
    models_folder_path = None

    for drive in drives:
        temp_script_path = Path(drive) / script_path_relative
        temp_args_folder_path = Path(drive) / args_folder_path_relative
        temp_models_folder_path = Path(drive) / models_folder_path_relative
        if temp_script_path.exists():
            test_script = temp_script_path
        if temp_args_folder_path.exists():
            args_folder_path = temp_args_folder_path
        if temp_models_folder_path.exists():
            models_folder_path = temp_models_folder_path

    if not test_script or not args_folder_path or not models_folder_path:
        print("Required directories not found in any of the drives.")
        exit()

    # make a list of files that start with "test_" and end with ".txt"
    args_file_list = [file for file in args_folder_path.iterdir() if file.name.startswith("test_") and file.name.endswith(".txt")]

    for args_file in args_file_list:  # Set the model directory path
        print(f"Working on {args_file}")
        fid_file = models_folder_path / args_file.stem.replace("test_", "") / 'fid/fid.txt'
        if not Path(fid_file).exists():
            tester = Tester(test_script, args_file)
            tester.test()