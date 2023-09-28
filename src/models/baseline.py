from zensvi.cv import Segmenter
import os
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import resnet18, resnet50
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import dotenv
import joblib  # Import joblib to save and load the scaler
import shutil
import glob
from collections import defaultdict
import tqdm

from predict_model import Predictor

import warnings

warnings.filterwarnings("ignore")


class GAN:
    def __init__(self, data_path, model_path, platform=["road_shoulder", "sidewalk"], gan_type=["cyclegan", "pix2pix"]):
        self.data_path = data_path
        self.model_path = model_path
        self.platform = platform
        self.gan_type = gan_type
        pass

    def _move_delete_results(self, src, dst):
        # make sure the dst folder exists
        os.makedirs(dst, exist_ok=True)
        # get a list of files in src that ends with "_fake_B.png"
        fake_B_list = glob.glob(os.path.join(src, "*_fake_B.png"))
        for fake_B in fake_B_list:
            # remove the "_fake_B.png" from the file name
            dst_file = os.path.join(
                dst, os.path.basename(fake_B).replace("_fake_B.png", ".png")
            )
            shutil.move(fake_B, dst_file)
        # delete src folder
        shutil.rmtree(src)

    def generate(self):
        for platform in self.platform:
            platform_data_path = os.path.join(self.data_path, platform)
            for data_type in ["panorama", "perspective"]:
                for gan_type in self.gan_type:
                    print(f"Generating {gan_type} data for {platform} in {data_type}")
                    if gan_type == "cyclegan":
                        path_prefix = (
                            "cyclegan_filtered"
                            if data_type == "panorama"
                            else "cyclegan_filtered_perspective"
                        )
                        
                        # create a new folder called gan_type in platform_data_path
                        os.makedirs(
                            os.path.join(platform_data_path, data_type, gan_type),
                            exist_ok=True,
                        )
                        # create trainA, trainB, testA, testB folders
                        os.makedirs(
                            os.path.join(platform_data_path, data_type, gan_type, "trainA"),
                            exist_ok=True,
                        )
                        os.makedirs(
                            os.path.join(platform_data_path, data_type, gan_type, "trainB"),
                            exist_ok=True,
                        )
                        os.makedirs(
                            os.path.join(platform_data_path, data_type, gan_type, "testA"),
                            exist_ok=True,
                        )
                        os.makedirs(
                            os.path.join(platform_data_path, data_type, gan_type, "testB"),
                            exist_ok=True,
                        )
                        # copy the images from trainA and testA to the new folder testA. do the same for trainB and testB
                        for folder in ["A", "B"]:
                            src_folder = os.path.join(
                                platform_data_path, path_prefix, "train" + folder
                            )
                            dest_folder = os.path.join(
                                platform_data_path, data_type, gan_type, "test" + folder
                            )
                            # add tqdm
                            for file_name in tqdm.tqdm(
                                os.listdir(src_folder),
                                desc=f"Copying {folder} files for {platform} in {data_type}",
                                total=len(os.listdir(src_folder)),
                            ):
                                full_file_name = os.path.join(src_folder, file_name)
                                new_full_file_name = os.path.join(dest_folder, file_name)
                                if os.path.isfile(full_file_name) and not os.path.exists(
                                    new_full_file_name
                                ):
                                    shutil.copy(full_file_name, dest_folder)

                        # Adjust paths for different data types
                        model_name_prefix = (
                            "cyclegan_default"
                            if data_type == "panorama"
                            else "cyclegan_perspective"
                        )
                        model_name = platform + "_" + model_name_prefix

                        # run the prediction for original test
                        test_len = len(
                            os.listdir(os.path.join(platform_data_path, path_prefix, "testA"))
                        )
                        if os.path.exists(
                            os.path.join(platform_data_path, data_type, gan_type, "test_gan_img")
                        ):
                            generated_test_len = len(
                                os.listdir(
                                    os.path.join(platform_data_path, data_type, gan_type, "test_gan_img")
                                )
                            )
                        else:
                            generated_test_len = 0
                        if not test_len * 6 == generated_test_len:
                            predictor = Predictor(
                                os.path.join(platform_data_path, path_prefix),
                                self.model_path,
                                os.path.join(platform_data_path, data_type, gan_type),
                                model_name,
                                "cycle_gan",
                                len(
                                    os.listdir(
                                        os.path.join(platform_data_path, path_prefix, "testA")
                                    )
                                ),
                            )
                            predictor.predict()
                            # move the results
                            self._move_delete_results(
                                os.path.join(
                                    platform_data_path,
                                    data_type,
                                    gan_type,
                                    model_name,
                                    "test_latest/images",
                                ),
                                os.path.join(platform_data_path, data_type, gan_type, "test_gan_img"),
                            )
                        train_len = len(
                            os.listdir(os.path.join(platform_data_path, path_prefix, "trainA"))
                        )
                        if os.path.exists(
                            os.path.join(platform_data_path, data_type, gan_type, "train_gan_img")
                        ):   
                            generated_train_len = len(
                                os.listdir(
                                    os.path.join(platform_data_path, data_type, gan_type, "train_gan_img")
                                )
                            )
                        else:
                            generated_train_len = 0
                        if not train_len * 6 == generated_train_len:
                            # run the prediction for the new test (original train)
                            predictor = Predictor(
                                os.path.join(platform_data_path, data_type, gan_type),
                                self.model_path,
                                os.path.join(platform_data_path, data_type, gan_type),
                                model_name,
                                "cycle_gan",
                                len(
                                    os.listdir(
                                        os.path.join(
                                            platform_data_path, data_type, gan_type, "testA"
                                        )
                                    )
                                ),
                            )
                            predictor.predict()
                            # move the results
                            self._move_delete_results(
                                os.path.join(
                                    platform_data_path,
                                    data_type,
                                    gan_type,
                                    model_name,
                                    "test_latest/images",
                                ),
                                os.path.join(
                                    platform_data_path, data_type, gan_type, "train_gan_img"
                                ),
                            )
                        
                    elif gan_type == "pix2pix":
                        path_prefix = (
                            "pix2pix_filtered"
                            if data_type == "panorama"
                            else "pix2pix_filtered_perspective"
                        )
                        os.makedirs(
                            os.path.join(platform_data_path, data_type, gan_type, "train"),
                            exist_ok=True,
                        )
                        os.makedirs(
                            os.path.join(platform_data_path, data_type, gan_type, "test"),
                            exist_ok=True,
                        )
                        # copy the images from train and test to the new folder test
                        src_folder = os.path.join(
                            platform_data_path, path_prefix, "train"
                        )
                        dest_folder = os.path.join(
                            platform_data_path, data_type, gan_type, "test"
                        )
                        # add tqdm
                        for file_name in tqdm.tqdm(
                            os.listdir(src_folder),
                            desc=f"Copying files for {platform} in {data_type}",
                            total=len(os.listdir(src_folder)),
                        ):
                            full_file_name = os.path.join(src_folder, file_name)
                            new_full_file_name = os.path.join(dest_folder, file_name)
                            if os.path.isfile(full_file_name) and not os.path.exists(
                                new_full_file_name
                            ):
                                shutil.copy(full_file_name, dest_folder)
                        # Adjust paths for different data types
                        model_name_prefix = (
                            "pix2pix_default"
                            if data_type == "panorama"
                            else "pix2pix_perspective"
                        )
                        model_name = platform + "_" + model_name_prefix
                        
                        # run the prediction for original test
                        test_len = len(
                            os.listdir(os.path.join(platform_data_path, path_prefix, "test"))
                        )
                        if os.path.exists(
                            os.path.join(platform_data_path, data_type, gan_type, "test_gan_img")
                        ): 
                            generated_test_len = len(
                                os.listdir(
                                    os.path.join(platform_data_path, data_type, gan_type, "test_gan_img")
                                )
                            )
                        else:
                            generated_test_len = 0
                        if not test_len * 3 == generated_test_len:
                            predictor = Predictor(
                                os.path.join(platform_data_path, path_prefix),
                                self.model_path,
                                os.path.join(platform_data_path, data_type, gan_type),
                                model_name,
                                "pix2pix",
                                len(
                                    os.listdir(
                                        os.path.join(platform_data_path, path_prefix, "test")
                                    )
                                ),
                            )
                            predictor.predict()
                            # move the results
                            self._move_delete_results(
                                os.path.join(
                                    platform_data_path,
                                    data_type,
                                    gan_type,
                                    model_name,
                                    "test_latest/images",
                                ),
                                os.path.join(platform_data_path, data_type, gan_type, "test_gan_img"),
                            )
                        train_len = len(
                            os.listdir(os.path.join(platform_data_path, path_prefix, "train"))
                        )
                        if os.path.exists(
                            os.path.join(platform_data_path, data_type, gan_type, "train_gan_img")
                        ): 
                            generated_train_len = len(
                                os.listdir(
                                    os.path.join(platform_data_path, data_type, gan_type, "train_gan_img")
                                )
                            )
                        else:
                            generated_train_len = 0
                        if not train_len * 3 == generated_train_len:
                            predictor = Predictor(
                                os.path.join(platform_data_path, data_type, gan_type),
                                self.model_path,
                                os.path.join(platform_data_path, data_type, gan_type),
                                model_name,
                                "pix2pix",
                                len(
                                    os.listdir(
                                        os.path.join(
                                            platform_data_path, data_type, gan_type, "test"
                                        )
                                    )
                                ),
                            )
                            predictor.predict()
                            # move the results
                            self._move_delete_results(
                                os.path.join(
                                    platform_data_path,
                                    data_type,
                                    gan_type,
                                    model_name,
                                    "test_latest/images",
                                ),
                                os.path.join(platform_data_path, data_type, gan_type, "train_gan_img"),
                            )


class Segmentation:
    def __init__(self, data_path, platform=["road_shoulder", "sidewalk"], gan_type=["cyclegan", "pix2pix"]):
        self.data_path = data_path
        self.platform = platform
        self.segmenter = Segmenter()
        self.gan_type = gan_type
        pass

    def segment(self):
        for platform in self.platform:
            platform_data_path = os.path.join(self.data_path, platform)
            for data_type in ["panorama", "perspective"]:
                for gan_type in self.gan_type:
                    print(f"Segmenting {gan_type} data for {platform} in {data_type}")
                    if gan_type == "cyclegan":
                        # Adjust paths for different data types
                        path_prefix = (
                            "cyclegan_filtered"
                            if data_type == "panorama"
                            else "cyclegan_filtered_perspective"
                        )
                        output_path = os.path.join(platform_data_path, data_type)
                        os.makedirs(
                            output_path, exist_ok=True
                        )  # Create output directory if not exists
                        path_dict = {
                            "train": [
                                os.path.join(platform_data_path, f"{path_prefix}/trainA"),
                                os.path.join(platform_data_path, f"{path_prefix}/trainB"),
                                os.path.join(
                                    platform_data_path, data_type, gan_type, "train_gan_img"
                                ),
                            ],
                            "test": [
                                os.path.join(platform_data_path, f"{path_prefix}/testA"),
                                os.path.join(platform_data_path, f"{path_prefix}/testB"),
                                os.path.join(platform_data_path, data_type, gan_type, "test_gan_img"),
                            ],
                        }
                    elif gan_type == "pix2pix":
                        # Adjust paths for different data types
                        path_prefix = (
                            "pix2pix_filtered_init"
                            if data_type == "panorama"
                            else "pix2pix_filtered_perspective_init"
                        )
                        output_path = os.path.join(platform_data_path, data_type)
                        os.makedirs(
                            output_path, exist_ok=True
                        )
                        path_dict = {
                            "train": [
                                os.path.join(platform_data_path, f"{path_prefix}/A/train"),
                                os.path.join(platform_data_path, f"{path_prefix}/B/train"),
                                os.path.join(
                                    platform_data_path, data_type, gan_type, "train_gan_img"
                                ),
                            ],
                            "test": [
                                os.path.join(platform_data_path, f"{path_prefix}/A/test"),
                                os.path.join(platform_data_path, f"{path_prefix}/B/test"),
                                os.path.join(
                                    platform_data_path, data_type, gan_type, "test_gan_img"
                                ),
                            ],
                        }
                    for key, value in path_dict.items():
                        path_A, path_B, path_GAN = value
                        csv_path_A = os.path.join(
                            platform_data_path, data_type, key, "A_segmentation"
                        )
                        csv_path_B = os.path.join(
                            platform_data_path, data_type, key, "B_segmentation"
                        )
                        csv_path_GAN = os.path.join(
                            platform_data_path, data_type, key, f"{gan_type}_segmentation"
                        )
                        # run if the csv files are not there
                        if not os.path.exists(csv_path_A):
                            self.segmenter.segment(
                                path_A,
                                dir_segmentation_summary_output=csv_path_A,
                                pixel_ratio_save_format=["csv"],
                                csv_format="wide",
                            )
                        if not os.path.exists(csv_path_B):
                            self.segmenter.segment(
                                path_B,
                                dir_segmentation_summary_output=csv_path_B,
                                pixel_ratio_save_format=["csv"],
                                csv_format="wide",
                            )
                        if not os.path.exists(csv_path_GAN):
                            self.segmenter.segment(
                                path_GAN,
                                dir_segmentation_summary_output=csv_path_GAN,
                                pixel_ratio_save_format=["csv"],
                                csv_format="wide",
                            )
                        # merge them
                        if not os.path.exists(
                            os.path.join(platform_data_path, data_type, f"{key}.csv")
                        ):
                            df_A = pd.read_csv(os.path.join(csv_path_A, "pixel_ratios.csv"))
                            df_B = pd.read_csv(os.path.join(csv_path_B, "pixel_ratios.csv"))
                            df_GAN = pd.read_csv(
                                os.path.join(csv_path_GAN, "pixel_ratios.csv")
                            )
                            # add f"_{gan_type}" to all the columns except filename_key
                            df_GAN.columns = [
                                col + f"_{gan_type}" if not col == "filename_key" else col
                                for col in df_GAN.columns
                            ]
                            joined_df = pd.merge(
                                df_A,
                                df_B,
                                on="filename_key",
                                how="left",
                                suffixes=("_gsv", "_mly"),
                            )
                            joined_df = pd.merge(
                                joined_df, df_GAN, on="filename_key", how="left"
                            )

                            # save it
                            joined_df.to_csv(
                                os.path.join(platform_data_path, data_type, f"{key}.csv"),
                                index=False,
                            )
                        else:
                            # check if the csv file has the gan_type columns
                            joined_df = pd.read_csv(
                                os.path.join(platform_data_path, data_type, f"{key}.csv")
                            )
                            if not f"building_{gan_type}" in joined_df.columns:
                                # join df_GAN to the existing csv
                                df_GAN = pd.read_csv(
                                    os.path.join(csv_path_GAN, "pixel_ratios.csv")
                                )
                                # add f"_{gan_type}" to all the columns except filename_key
                                df_GAN.columns = [
                                    col + f"_{gan_type}" if not col == "filename_key" else col
                                    for col in df_GAN.columns
                                ]
                                joined_df = pd.merge(
                                    joined_df, df_GAN, on="filename_key", how="left"
                                )
                                # save it
                                joined_df.to_csv(
                                    os.path.join(
                                        platform_data_path, data_type, f"{key}.csv"
                                    ),
                                    index=False,
                                )
                            else:
                                print("Already segmented")


class LightGBMModel:
    def __init__(self, data_path, platform=["road_shoulder", "sidewalk"], with_gan=["without_gan", "cyclegan", "pix2pix"]):
        self.data_path = data_path
        self.platform = platform
        self.with_gan = with_gan
        self.kf = KFold(n_splits=5, random_state=42, shuffle=True)
        self.model = lgb.LGBMRegressor(
            objective="regression",
            num_leaves=31,  # Lower bound of the grid
            learning_rate=0.1,  # Single value in the grid
            n_estimators=500,  # Lower bound of the grid
            max_depth=-1,  # Lower bound of the grid
            reg_alpha=0.0,  # Lower bound of the grid
            reg_lambda=0.0,  # Lower bound of the grid
            min_child_samples=10,  # Lower bound of the grid
            subsample=0.8,  # Single value in the grid
            colsample_bytree=0.8,  # Single value in the grid
        )

    def train_and_test(self):
        target_vars = [
            "building",
            "vegetation",
            "sky",
        ]  # Limit to these target variables
        for platform in self.platform:
            platform_data_path = os.path.join(self.data_path, platform)
            for data_type in ["panorama", "perspective"]:
                for with_gan in self.with_gan:
                    data_path = os.path.join(platform_data_path, data_type)
                    data_train = pd.read_csv(os.path.join(data_path, "train.csv"))
                    data_test = pd.read_csv(os.path.join(data_path, "test.csv"))
                    if with_gan != "without_gan":
                        predictors = [
                            col
                            for col in data_train.columns
                            if (f"_{with_gan}" in col) or ("_gsv" in col)
                        ]
                    else:
                        predictors = [
                            col for col in data_train.columns if "_gsv" in col
                        ]
                    print(f"Predictor Variables: {predictors}")
                    targets = [
                        col
                        for col in data_train.columns
                        if col.replace("_mly", "") in target_vars
                    ]
                    for target in targets:
                        print(
                            f"Training {target} on {platform} with {data_type} data {with_gan}"
                        )
                        # skip if the output file exists
                        if os.path.exists(
                            os.path.join(
                                data_path, f"{target}_{with_gan}_test_metrics.csv"
                            )
                        ) and os.path.exists(
                            os.path.join(
                                data_path, f"{target}_{with_gan}_lightgbm_model.pkl"
                            )
                        ):
                            continue
                        # Training part
                        X_train = data_train[predictors]
                        y_train = data_train[target]

                        # Feature selection
                        select = SelectKBest(score_func=f_regression, k="all")

                        # Making a pipeline with feature selection and your model
                        pipeline = make_pipeline(select, self.model)

                        param_grid = {
                            "selectkbest__k": [
                                "all"
                            ],  # Keeping 'all' as it consistently appears as the top performer
                            "lgbmregressor__num_leaves": [
                                31,
                                50,
                                100,
                            ],  # Adding an intermediate value for exploration
                            "lgbmregressor__max_depth": [
                                -1,
                                20,
                            ],  # Adding 20 to allow for some exploration around the unrestricted depth
                            "lgbmregressor__learning_rate": [
                                0.1
                            ],  # Adding a lower value to explore a possibly smoother convergence
                            "lgbmregressor__n_estimators": [
                                500,
                                700,
                            ],  # Adding values around 500 to explore potential improvements
                            "lgbmregressor__reg_alpha": [
                                0.0,
                                0.5,
                                1.0,
                            ],  # Adding a higher value to explore increased regularization
                            "lgbmregressor__reg_lambda": [
                                0.0,
                                0.5,
                                1.0,
                            ],  # Adding 0.0 and 1.0 to explore a range of regularization strengths
                            "lgbmregressor__min_child_samples": [
                                10,
                                20,
                                30,
                            ],  # Adding values around 20 to explore potential benefits of different regularization
                            "lgbmregressor__subsample": [
                                0.8
                            ],  # Adding values around 0.8 to explore potential improvements
                            "lgbmregressor__colsample_bytree": [
                                0.8
                            ],  # Adding values around 0.8 to explore potential improvements
                        }

                        # Grid search with cross-validation
                        grid_search = GridSearchCV(
                            pipeline,
                            param_grid,
                            cv=self.kf,
                            scoring="neg_mean_squared_error",
                        )
                        grid_search.fit(X_train, y_train)

                        # Get the best model
                        best_model = grid_search.best_estimator_

                        # Saving the cross-validation results
                        cv_results = pd.DataFrame(grid_search.cv_results_)
                        cv_results["target"] = target
                        cv_results.to_csv(
                            os.path.join(
                                data_path, f"{target}_{with_gan}_train_metrics.csv"
                            ),
                            index=False,
                        )

                        # Testing part
                        X_test = data_test[predictors]
                        y_test = data_test[target]

                        y_pred_test = best_model.predict(X_test)
                        mse_test = mean_squared_error(y_test, y_pred_test)
                        mae_test = mean_absolute_error(y_test, y_pred_test)
                        r2_test = r2_score(y_test, y_pred_test)

                        result_test = pd.DataFrame(
                            {
                                "target": [target],
                                "mse_mean": [mse_test],
                                "mae_mean": [mae_test],
                                "r2_mean": [r2_test],
                            }
                        )
                        result_test.to_csv(
                            os.path.join(
                                data_path, f"{target}_{with_gan}_test_metrics.csv"
                            ),
                            index=False,
                        )
                        # Saving the feature names used for training
                        feature_names = X_train.columns.tolist()

                        # Save the model and feature names together
                        joblib.dump(
                            {"model": best_model, "feature_names": feature_names},
                            os.path.join(
                                data_path, f"{target}_{with_gan}_lightgbm_model.pkl"
                            ),
                        )


class ImageDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, with_gan="without_gan"):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.with_gan = with_gan

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.df.iloc[idx, 0]))
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(
            self.df.iloc[idx, 1:2].values.astype(np.float32)
        )  # Assuming the label is in the second column
        if self.with_gan != "without_gan":
            additional_features = torch.tensor(
                self.df.iloc[idx, 2:].values.astype(np.float32)
            )
            return image, label, additional_features
        else:
            return image, label


class MyCustomLayer(torch.nn.Module):
    def __init__(self, num_additional_features):
        super(MyCustomLayer, self).__init__()
        
        # Single linear layer
        self.fc = torch.nn.Linear(2048 + num_additional_features, 1)

    def forward(self, x, additional_features=None):
        # Concatenate the additional features to the input x if available
        if additional_features is not None:
            x = torch.cat((x, additional_features), dim=1)
        
        # Apply the linear layer
        x = self.fc(x)
        return x

class ResnetModel:
    def __init__(self, data_path, platform=["road_shoulder", "sidewalk"], with_gan = ["without_gan", "cyclegan", "pix2pix"]):
        self.data_path = data_path
        self.platform = platform
        self.with_gan = with_gan
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet50(pretrained=True).to(
            self.device
        )  # Use a deeper architecture
        self.model.fc = torch.nn.Linear(2048, 1).to(
            self.device
        )  # Adjust the fully connected layer accordingly
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.001
        )  # Use a different optimizer
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),  # Add more data augmentations
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.scalers = defaultdict(lambda: defaultdict(StandardScaler))

    def get_model(self, num_additional_features=None):
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

    def train(self, epochs=30):
        target_vars = [
            "building",
            "vegetation",
            "sky",
        ]  # Limit to these target variables
        for platform in self.platform:
            platform_data_path = os.path.join(self.data_path, platform)
            for data_type in ["panorama", "perspective"]:
                for with_gan in self.with_gan:
                    path_prefix = (
                        "cyclegan_filtered"
                        if data_type == "panorama"
                        else "cyclegan_filtered_perspective"
                    )
                    data_path = os.path.join(platform_data_path, data_type)
                    image_path = os.path.join(platform_data_path, path_prefix, "trainA")
                    df = pd.read_csv(os.path.join(data_path, "train.csv"))
                    df["filename_key"] = df["filename_key"].apply(
                        lambda x: str(x) + ".jpg"
                    )
                    if with_gan != "without_gan":
                        self.predictors = [
                            col
                            for col in df.columns
                            if (f"_gsv" in col) or (f"_{with_gan}" in col)
                        ]
                    else:
                        self.predictors = []
                    print(f"Predictor Variables: {self.predictors}")
                    targets = [
                        col
                        for col in df.columns
                        if col.replace("_mly", "") in target_vars
                    ]
                    for target in targets:
                        print(
                            f"Training {target} on {platform} with {data_type} data {with_gan}"
                        )
                        if os.path.exists(
                            os.path.join(
                                data_path, f"{target}_{with_gan}_train_model.pth"
                            )
                        ) and os.path.exists(
                            os.path.join(data_path, f"{target}_{with_gan}_scaler.pkl")
                        ) and os.path.exists(
                            os.path.join(
                                data_path, f"{target}_{with_gan}_predictors.pkl"
                            )):
                            continue
                        # Initializing the scaler for this target
                        self.scalers[target][target] = StandardScaler()

                        # Getting the targets and scaling them
                        target_values = df[target].values.reshape(-1, 1)
                        scaled_targets = self.scalers[target][target].fit_transform(
                            target_values
                        )
                        df[target] = scaled_targets.flatten()

                        # Getting the targets and scaling them
                        if len(self.predictors) != 0:
                            for col in self.predictors:
                                self.scalers[target][col] = StandardScaler()
                                col_values = df[col].values.reshape(-1, 1)
                                scaled_col_values = self.scalers[target][
                                    col
                                ].fit_transform(col_values)
                                df[col] = scaled_col_values.flatten()

                        dataset = ImageDataset(
                            df[["filename_key", target] + self.predictors],
                            image_path,
                            transform=self.transform,
                            with_gan=with_gan,
                        )
                        dataloader = DataLoader(
                            dataset, batch_size=32, shuffle=True, num_workers=8
                        )

                        if with_gan != "without_gan":
                            num_additional_features = len(self.predictors)
                        else:
                            num_additional_features = None
                        self.model, self.optimizer = self.get_model(
                            num_additional_features
                        )
                        train_loss = []
                        test_loss = []  # Initialize a list to hold the test losses
                        for epoch in range(
                            epochs
                        ):  # loop over the dataset multiple times
                            running_loss = 0.0
                            for i, data in enumerate(dataloader, 0):
                                # zero the parameter gradients
                                self.optimizer.zero_grad()
                                if with_gan != "without_gan":
                                    inputs, labels, additional_features = data
                                    inputs, labels, additional_features = (
                                        inputs.to(self.device),
                                        labels.to(self.device),
                                        additional_features.to(self.device),
                                    )
                                    # forward + backward + optimize
                                    outputs = self.model(inputs, additional_features)
                                else:
                                    inputs, labels = data
                                    inputs, labels = inputs.to(self.device), labels.to(
                                        self.device
                                    )
                                    outputs = self.model(inputs)
                                loss = self.criterion(outputs, labels)
                                loss.backward()
                                self.optimizer.step()
                                running_loss += loss.item()
                            epoch_loss = running_loss / len(dataloader)
                            train_loss.append(epoch_loss)
                            # Get the test loss at the end of each epoch
                            epoch_test_loss = self._test_for_train(
                                target,
                                platform,
                                data_type,
                                with_gan,
                                data_path,
                                image_path,
                            )
                            test_loss.append(
                                epoch_test_loss
                            )  # Append the test loss to the list
                            print(
                                f"Epoch {epoch + 1}, train loss: {epoch_loss}, test loss: {epoch_test_loss}"
                            )

                        print(
                            f"Finished Training {target} on {platform} with {data_type} data {with_gan}"
                        )
                        # save model
                        torch.save(
                            self.model.state_dict(),
                            os.path.join(
                                data_path, f"{target}_{with_gan}_train_model.pth"
                            ),
                        )
                        # Saving the scaler
                        joblib.dump(
                            self.scalers[target],
                            os.path.join(data_path, f"{target}_{with_gan}_scaler.pkl"),
                        )
                        # save column names as a list in pickle file
                        joblib.dump(
                            self.predictors,
                            os.path.join(
                                data_path, f"{target}_{with_gan}_predictors.pkl"
                            ),
                        )
                        # Save both train and test losses to the same CSV
                        loss_df = pd.DataFrame(
                            {"train_loss": train_loss, "test_loss": test_loss}
                        )
                        loss_df.to_csv(
                            os.path.join(data_path, f"{target}_{with_gan}_loss.csv"),
                            index=False,
                        )

    def _test_for_train(
        self, target, platform, data_type, with_gan, data_path, image_path
    ):
        platform_data_path = os.path.join(self.data_path, platform)
        path_prefix = (
            "cyclegan_filtered"
            if data_type == "panorama"
            else "cyclegan_filtered_perspective"
        )
        data_path = os.path.join(platform_data_path, data_type)
        image_path = os.path.join(platform_data_path, path_prefix, "testA")
        df = pd.read_csv(os.path.join(data_path, "test.csv"))
        df["filename_key"] = df["filename_key"].apply(lambda x: str(x) + ".jpg")
        target_values = df[target].values.reshape(-1, 1)
        df[target] = self.scalers[target][target].transform(target_values).flatten()

        # Transforming the predictor variables
        for col in self.predictors:
            col_values = df[col].values.reshape(-1, 1)
            df[col] = self.scalers[target][col].transform(col_values).flatten()

        dataset = ImageDataset(
            df[["filename_key", target] + self.predictors],
            image_path,
            transform=self.transform,
            with_gan=with_gan,
        )
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)

        total_loss = 0.0

        with torch.no_grad():
            for data in dataloader:
                if with_gan == "without_gan":
                    images, labels = data
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                else:
                    images, labels, additional_features = data
                    images, labels, additional_features = (
                        images.to(self.device),
                        labels.to(self.device),
                        additional_features.to(self.device),
                    )
                    outputs = self.model(images, additional_features)

                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

        mean_loss = total_loss / len(dataloader)
        return mean_loss

    def test(self):
        target_vars = [
            "building",
            "vegetation",
            "sky",
        ]  # Limit to these target variables
        for platform in self.platform:
            platform_data_path = os.path.join(self.data_path, platform)
            for data_type in ["panorama", "perspective"]:
                for with_gan in self.with_gan:
                    path_prefix = (
                        "cyclegan_filtered"
                        if data_type == "panorama"
                        else "cyclegan_filtered_perspective"
                    )
                    data_path = os.path.join(platform_data_path, data_type)
                    image_path = os.path.join(platform_data_path, path_prefix, "testA")
                    df = pd.read_csv(os.path.join(data_path, "test.csv"))
                    df["filename_key"] = df["filename_key"].apply(
                        lambda x: str(x) + ".jpg"
                    )

                    targets = [
                        col
                        for col in df.columns
                        if col.replace("_mly", "") in target_vars
                    ]
                    for target in targets:
                        print(
                            f"Testing {target} on {platform} with {data_type} data {with_gan}"
                        )
                        if os.path.exists(
                            os.path.join(
                                data_path, f"{target}_{with_gan}_test_loss.csv"
                            )
                        ):
                            continue
                        # load the predictors
                        predictors = joblib.load(
                            os.path.join(
                                data_path, f"{target}_{with_gan}_predictors.pkl"
                            )
                        )
                        # Loading the scaler
                        self.scalers[target] = joblib.load(
                            os.path.join(data_path, f"{target}_{with_gan}_scaler.pkl")
                        )

                        # Transforming the target variable
                        target_values = df[target].values.reshape(-1, 1)
                        df[target] = (
                            self.scalers[target][target]
                            .transform(target_values)
                            .flatten()
                        )

                        # Transforming the predictor variables
                        for col in predictors:
                            col_values = df[col].values.reshape(-1, 1)
                            df[col] = (
                                self.scalers[target][col]
                                .transform(col_values)
                                .flatten()
                            )

                        dataset = ImageDataset(
                            df[["filename_key", target] + predictors],
                            image_path,
                            transform=self.transform,
                            with_gan=with_gan,
                        )
                        dataloader = DataLoader(
                            dataset, batch_size=32, shuffle=False, num_workers=8
                        )
                        if with_gan != "without_gan":
                            num_additional_features = len(predictors)
                        else:
                            num_additional_features = None
                        self.model, self.optimizer = self.get_model(
                            num_additional_features
                        )
                        # Load the model
                        self.model.load_state_dict(
                            torch.load(
                                os.path.join(
                                    data_path, f"{target}_{with_gan}_train_model.pth"
                                )
                            )
                        )

                        total_loss = 0.0
                        total_mae = 0.0
                        total_r2_score = 0.0
                        total = 0
                        all_labels = []
                        all_outputs = []

                        with torch.no_grad():
                            for data in dataloader:
                                if with_gan == "without_gan":
                                    images, labels = data
                                    images, labels = images.to(self.device), labels.to(
                                        self.device
                                    )
                                    outputs = self.model(images)
                                else:
                                    images, labels, additional_features = data
                                    images, labels, additional_features = (
                                        images.to(self.device),
                                        labels.to(self.device),
                                        additional_features.to(self.device),
                                    )
                                    outputs = self.model(images, additional_features)

                                loss = self.criterion(outputs, labels)
                                total_loss += loss.item()
                                total += labels.size(0)

                                # Accumulate all labels and outputs for MAE and R2 Score calculation
                                all_labels.append(labels.cpu().numpy())
                                all_outputs.append(outputs.cpu().numpy())

                        # Flatten the list of results and compute MAE and R2 Score
                        flat_labels = [
                            item for sublist in all_labels for item in sublist
                        ]
                        flat_outputs = [
                            item for sublist in all_outputs for item in sublist
                        ]
                        # Inverse scaling the predictions before calculating the metrics
                        flat_outputs = self.scalers[target][target].inverse_transform(
                            flat_outputs
                        )

                        total_mae = mean_absolute_error(flat_labels, flat_outputs)
                        total_r2_score = r2_score(flat_labels, flat_outputs)

                        mean_loss = total_loss / total
                        print(
                            f"Mean loss of the network on the {total} test images: {mean_loss}"
                        )
                        print(f"Mean absolute error on the test images: {total_mae}")
                        print(f"R2 score on the test images: {total_r2_score}")

                        result = pd.DataFrame(
                            {
                                "target": [target],
                                "mean_loss": [mean_loss],
                                "mae": [total_mae],
                                "r2_score": [total_r2_score],
                            }
                        )
                        result.to_csv(
                            os.path.join(
                                data_path, f"{target}_{with_gan}_test_loss.csv"
                            ),
                            index=False,
                        )


if __name__ == "__main__":
    dotenv.load_dotenv(dotenv.find_dotenv())
    root_dir = os.getenv("ROOT_DIR")
    if not os.path.exists(root_dir):
        # list of drives from D to Z
        drives = [f"{chr(x)}:/" for x in range(ord("D"), ord("Z") + 1)] + [
            "/Volumes/ExFAT/"
        ]
        for drive in drives:
            # check if the root_dir exists
            if os.path.exists(os.path.join(drive, "road_shoulder_gan")):
                root_dir = os.path.join(drive, "road_shoulder_gan")
                break

    data_path = os.path.join(root_dir, "data/processed")
    # GAN
    gan = GAN(data_path, os.path.join(root_dir,"models"))
    gan.generate()
    # segmentation
    segmenter = Segmentation(data_path)
    segmenter.segment()
    # lightgbm
    lightgbm = LightGBMModel(data_path)
    lightgbm.train_and_test()
    # resnet
    resnet = ResnetModel(data_path)
    resnet.train()
    # resnet.test()
