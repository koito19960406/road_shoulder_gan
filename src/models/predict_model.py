import subprocess
import os 
import glob
import torch 

class Predictor:
    """class to run shell script to train GAN models
    """
    def __init__(self, data_root, check_point, results_dir, name, model, num_test):
        self.test_script = "src/models/pytorch-CycleGAN-and-pix2pix/test.py"
        self.name = name
        self.data_root = data_root
        self.check_point = check_point
        self.results_dir = results_dir
        self.num_test = num_test
        self.model = model
        self.gpu_ids = str(torch.cuda.device_count()-1)
        if model=="cycle_gan":
            self.dataset_mode = "unaligned"
        else:
            self.dataset_mode = "aligned"

    def predict(self):
        argument = f"python {self.test_script} --dataroot {self.data_root} --name {self.name} --gpu_ids {self.gpu_ids} --checkpoints_dir {self.check_point} --model {self.model} --results_dir {self.results_dir} --num_test {self.num_test} --dataset_mode {self.dataset_mode}"
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
        predictor = Predictor(os.path.join(root_dir, "data/processed",name), 
                            os.path.join(root_dir,"models"), name, model)
        predictor.predict()