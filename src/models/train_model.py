import subprocess
import os 

class Trainer:
    """class to run shell script to train GAN models
    """
    def __init__(self, name, model = "cyclegan"):
        self.train_script = "src/models/pytorch-CycleGAN-and-pix2pix/train.py"
        self.name = name
        self.data_root = os.path.join("/Volumes/ExFAT/road_shoulder_gan/data/processed",self.name)
        self.check_point = "/Volumes/exfat/road_shoulder_gan/models/"
        self.model = model
        if model=="cyclegan":
            self.dataset_mode = "unaligned"
        else:
            self.dataset_mode = "aligned"
        pass
    def train(self):
        argument = f"python {self.train_script} --dataroot {self.data_root} --name {self.name} --checkpoints_dir {self.check_point} --model {self.model} --dataset_mode {self.dataset_mode}"
        subprocess.Popen([argument], shell = True)
        pass

if __name__ == '__main__':
    name_list = ["cyclegan_filtered", "pix2pix_filtered"]
    for name in name_list:
        if "cyclegan" in name:
            model = "cyclegan"
        if "pix2pix" in name:
            model = "pix2pix"
        trainer = Trainer(name, model)
        trainer.train()