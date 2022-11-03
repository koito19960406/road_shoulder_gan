import subprocess
import os 

class Predictor:
    """class to run shell script to train GAN models
    """
    def __init__(self, root_dir, name, model = "cyclegan"):
        self.root_dir = root_dir
        self.train_script = "src/models/pytorch-CycleGAN-and-pix2pix/train.py"
        self.name = name
        self.data_root = os.path.join(self.root_dir, "data/processed",self.name)
        self.check_point = os.path.join(self.root_dir,"models")
        self.model = model
        self.results_dir = os.path.join(self.check_point,self.name)
        if model=="cyclegan":
            self.dataset_mode = "unaligned"
        else:
            self.dataset_mode = "aligned"
        pass
    def predict(self):
        argument = f"python {self.train_script} --dataroot {self.data_root} --name {self.name} --checkpoints_dir {self.check_point} --results_dir {self.results_dir} --model {self.model} --dataset_mode {self.dataset_mode}"
        subprocess.Popen([argument], shell = True)
        pass

if __name__ == '__main__':
    root_dir = "/Volumes/ExFAT/road_shoulder_gan"
    name_list = ["cyclegan_filtered", "pix2pix_filtered"]
    for name in name_list:
        if "cyclegan" in name:
            model = "cyclegan"
        if "pix2pix" in name:
            model = "pix2pix"
        predictor = Predictor(root_dir, name, model)
        predictor.predict()