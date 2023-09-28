import subprocess
import os 
import glob
import torch 
import pandas as pd
import joblib
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from torchvision.models import resnet50
import torch

class Predictor:
    """class to run shell script to test GAN models
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
        subprocess.run(argument, shell = True)
        pass

class CustomPredictor:
    def __init__(self, data_root, csv_path, check_point, results_dir, model_name_with_gan, image_path = None):
        self.data_root = data_root
        self.csv_path = csv_path
        self.check_point = check_point
        self.results_dir = results_dir
        self.model_name_with_gan = model_name_with_gan
        self.model_name = self._get_model_name(model_name_with_gan) # either lightgbm or resnet + with_gan or without_gan
        self.with_gan = self._get_with_gan(model_name_with_gan) # either with_gan or without_gan
        self.image_path = image_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def _get_model_name(self, model_name_with_gan):
        if "lightgbm" in model_name_with_gan:
            return "lightgbm"
        elif "resnet" in model_name_with_gan:
            return "resnet"
        else:
            raise ValueError("model name not recognized")
        
    def _get_with_gan(self, model_name_with_gan):
        if "cyclegan" in model_name_with_gan:
            return "cyclegan"
        elif "pix2pix" in model_name_with_gan:
            return "pix2pix"
        elif "without_gan" in model_name_with_gan:
            return "without_gan"
        else:
            raise ValueError("model name not recognized")
    
    def _get_model(self, num_additional_features=None):
        from baseline import MyCustomLayer
        
        model = resnet50(pretrained=True).to(self.device)

        if num_additional_features is not None:

            class CustomResNet(torch.nn.Module):
                def __init__(self, original_model, num_additional_features, device):
                    super(CustomResNet, self).__init__()
                    self.features = torch.nn.Sequential(
                        *list(original_model.children())[:-1]
                    )  # get all layers except the last one
                    self.fc = MyCustomLayer(num_additional_features).to(
                        device
                    )  # your custom layer

                def forward(self, x, additional_features=None):
                    x = self.features(x)
                    x = x.view(x.size(0), -1)
                    x = self.fc(
                        x, additional_features
                    )  # now pass the additional_features here
                    return x

            model = CustomResNet(model, num_additional_features, self.device)
        else:
            model.fc = torch.nn.Linear(2048, 1).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        return model, optimizer
    
    def load_model(self, num_additional_features=None):
        if self.model_name == "lightgbm":
            # load from pickle with joblib
            model = joblib.load(self.check_point)
        elif self.model_name == "resnet":
            model = self._get_model(num_additional_features=num_additional_features)[0]
            # load from torch
            model.load_state_dict(torch.load(self.check_point))
            model = model.to(self.device)
        else:
            raise ValueError("model name not recognized")
        return model
    
    def _predict_lightgbm(self, target):
        # load model and feature names
        model_data = self.load_model()
        model = model_data['model']
        feature_names = model_data['feature_names']

        # load data
        df = pd.read_csv(self.csv_path)

        # Reorder the columns in the test data to match the order used during training
        df = df[feature_names + [target, 'filename_key']]

        # run prediction
        df["predicted_" + target] = model.predict(df[feature_names])

        # save to csv
        df.to_csv(os.path.join(self.results_dir, f"{target}_{self.model_name_with_gan}_predictions.csv"), index=False)

    def _predict_resnet(self, target):
        from baseline import ImageDataset
        # load data
        df = pd.read_csv(self.csv_path)
        # load scaler
        scalers = joblib.load(os.path.join(self.data_root, f"{target}_{self.with_gan}_scaler.pkl"))
        # load predictors
        predictors = joblib.load(os.path.join(self.data_root, f"{target}_{self.with_gan}_predictors.pkl"))
        # load model 
        if len(predictors) == 0:
            num_additional_features = None
        else:
            num_additional_features = len(predictors)
        model = self.load_model(num_additional_features = num_additional_features)
        # Transforming the target variable
        df[target] = scalers[target].transform(df[target].values.reshape(-1, 1)).flatten()
        # Transforming the predictor variables
        for col in predictors:
            col_values = df[col].values.reshape(-1, 1)
            df[col] = scalers[col].transform(col_values).flatten()
        # run prediction
        dataset = ImageDataset(df[['filename_key', target] + predictors], self.image_path, transform=self.transform, with_gan=self.with_gan)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)
        all_outputs = []
        with torch.no_grad():
            for data in dataloader:
                if self.with_gan == "without_gan":
                    images, labels = data
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                else:
                    images, labels, additional_features = data
                    images, labels, additional_features = images.to(self.device), labels.to(self.device), additional_features.to(self.device)
                    outputs = model(images, additional_features)
                all_outputs.append(outputs.cpu().numpy())
        # Flatten the list of results
        flat_outputs = [item for sublist in all_outputs for item in sublist]
        # Inverse scaling the predictions before calculating the metrics
        flat_outputs = scalers[target].inverse_transform(flat_outputs)
                        
        # store in df as a column
        df["predicted_" + target] = flat_outputs
        # save to csv
        df.to_csv(os.path.join(self.results_dir, f"{target}_{self.model_name_with_gan}_predictions.csv"), index=False)

    def predict(self, target):
        if self.model_name == "lightgbm":
            self._predict_lightgbm(target)
        elif self.model_name == "resnet":
            self._predict_resnet(target)
        else:
            raise ValueError("model name not recognized")
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