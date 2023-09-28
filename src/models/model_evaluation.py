import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import dotenv
import tqdm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import re
import numpy as np

from src.models.predict_model import CustomPredictor

class EvaluateModels:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.platforms = ["road_shoulder", "sidewalk"]
        self.targets = ["building", "sky", "vegetation"]
        self.models = ["gsv", "cyclegan", "pix2pix", "lightgbm", "resnet"]
        self.gan_modes = ["without_gan", "cyclegan", "pix2pix"]
        self.views = ["panorama", "perspective"]

    def _calc_metrics(self, y_true, y_pred, model, view, results, gan_mode = ""):
        # Calculate the metrics
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        # Pearson's correlation coefficient
        pearson = pd.Series(y_true).corr(pd.Series(y_pred), method="pearson")

        # Store the results
        results.append({
            "Model": f"{model.capitalize()} {view.capitalize()} {gan_mode.replace('_', ' ').capitalize()}",
            "MSE": mse,
            "MAE": mae,
            "R-squared": r2,
            "Pearson's r": pearson
        })
        return results

    def _load_rename(self, file_path, suffix):
        """load csv as df and rename col names

        Args:
            file_path (str): file path to the segmentation result
            prefix (str): prefix to add to col names
        """
        result = pd.read_csv(file_path)
        # rename all the cols excepts for the first column, which is pid
        result.columns = [colname if "file" in colname else colname + suffix for i, colname in enumerate(result.columns)]
        return result
    
    def evaluate(self):
        for platform in tqdm.tqdm(self.platforms, desc="platform"):
            for target in tqdm.tqdm(self.targets, desc="target", leave=False):
                if (os.path.exists(os.path.join(self.root_dir, "data/processed", platform, f"metrics_{target}.csv"))):
                    results_df = pd.read_csv(os.path.join(self.root_dir, "data/processed", platform, f"metrics_{target}.csv"))
                else:
                    results = []
                    for view in tqdm.tqdm(self.views, desc="view", leave=False):
                        for model in tqdm.tqdm(self.models, desc="model", leave=False):
                            # get the ground truth
                            data_path = os.path.join(self.root_dir, "data/processed", platform, view)
                            gt_df = pd.read_csv(os.path.join(data_path, f"test.csv"))
                            # convert filename_key to string
                            gt_df["filename_key"] = gt_df["filename_key"].apply(lambda x: str(x))
                            y_true = gt_df[target + "_mly"].values
                            pred_df = pd.read_csv(os.path.join(data_path, f"test.csv"))
                            if model == "gsv":
                                y_pred = pred_df[target + "_gsv"].values
                                results = self._calc_metrics(y_true, y_pred, model, view, results)
                            elif (model == "cyclegan") or (model == "pix2pix"):
                                y_pred = pred_df[target + f"_{model}"].values
                                results = self._calc_metrics(y_true, y_pred, model, view, results)
                            else:
                                for gan_mode in tqdm.tqdm(self.gan_modes, desc="gan_mode", leave=False):
                                    # first, we need to combine the segmentation results
                                    csv_folder = os.path.join(self.root_dir, "data/processed", platform, view, "test")
                                    csv_path = os.path.join(csv_folder,"segmentation_result.csv")
                                    # load
                                    gsv_result = self._load_rename(os.path.join(csv_folder, "A_segmentation/pixel_ratios.csv"), "_gsv") 
                                    mly_result = self._load_rename(os.path.join(csv_folder, "B_segmentation/pixel_ratios.csv"), "_mly")
                                    if gan_mode != "without_gan":
                                        gan_result = self._load_rename(os.path.join(csv_folder, f"{gan_mode}_segmentation/pixel_ratios.csv"), f"_{gan_mode}")
                                        # merge
                                        merged_result = (gsv_result.merge(mly_result,on=["filename_key"],how="left").
                                            merge(gan_result,on=["filename_key"],how="left")
                                            )
                                    else:
                                        merged_result = (gsv_result.merge(mly_result,on=["filename_key"],how="left"))
                                    
                                    if model == "resnet":
                                        merged_result["filename_key"] = merged_result["filename_key"].apply(
                                            lambda x: str(x) + ".jpg"
                                        )
                                    # save as csv
                                    merged_result.to_csv(csv_path)
                                    # Predict the target variable: data_root, csv_path, check_point, results_dir, model_name_with_gan
                                    if model == "lightgbm":
                                        model_ext = "lightgbm_model.pkl"
                                        image_path = None
                                    elif model == "resnet":
                                        model_ext = "train_model.pth"
                                        if view == "panorama":
                                            image_path = os.path.join(self.root_dir, "data", "processed", platform, "cyclegan_filtered", "testA")
                                        elif view == "perspective":
                                            image_path = os.path.join(self.root_dir, "data", "processed", platform, "cyclegan_filtered_perspective", "testA")
                                        else:
                                            raise ValueError("view name not recognized")
                                    else:
                                        raise ValueError("model name not recognized")
                                    check_point = os.path.join(self.root_dir, "data/processed",platform, view, f"{target}_mly_{gan_mode}_{model_ext}")
                                    results_dir = os.path.join(self.root_dir, "data/processed",platform, view)
                                    model_name_with_gan = f"{model}_{gan_mode}"
                                    predictor = CustomPredictor(data_path,
                                                                csv_path,
                                                                check_point,
                                                                results_dir,
                                                                model_name_with_gan,
                                                                image_path = image_path)
                                    predictor.predict(target + "_mly")
                                    pred_df = pd.read_csv(os.path.join(results_dir, f"{target}_mly_{model_name_with_gan}_predictions.csv"))
                                    # make sure to remove ".jpg" from the filename_key
                                    pred_df["filename_key"] = pred_df["filename_key"].apply(lambda x: str(x).replace(".jpg",""))
                                    # merge with the ground truth
                                    joined_df = pred_df[["filename_key",f"predicted_{target}_mly"]].merge(gt_df[["filename_key", f"{target}_mly"]], on="filename_key", how="left")
                                    y_true = joined_df[f"{target}_mly"].values
                                    y_pred = joined_df[f"predicted_{target}_mly"].values
                                    results = self._calc_metrics(y_true, y_pred, model, view, results, gan_mode = gan_mode)
                            
                    # Convert the results to a DataFrame
                    results_df = pd.DataFrame(results)
                    
                    # Save the results to a separate file
                    results_df.to_csv(os.path.join(self.root_dir, "data/processed", platform, f"metrics_{target}.csv"), index=False)
                    
                # Function to round to n significant digits
                def round_to_n_significant_digits(x, n=3):
                    if x == 0:
                        return "0"
                    else:
                        # Round to significant digits
                        rounded = round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1))
                        
                        # Convert to string and remove trailing zeros
                        str_rounded = format(rounded, f".{n}g")
                        if "." in str_rounded:
                            str_rounded = str_rounded.rstrip('0').rstrip('.')
                        return str_rounded
                # Estimate the width based on the longest column name
                max_col_name_length = max(len(col) for col in results_df.columns)
                # Assume each character takes up approximately 10 pixels
                width = f"{max_col_name_length * 6}px"

                # Round the numerical columns of the DataFrame to 3 significant digits
                numerical_columns = ['MSE', 'MAE', 'R-squared', "Pearson's r"]
                for col in numerical_columns:
                    results_df[col] = results_df[col].apply(round_to_n_significant_digits)

                # Create a reversed colormap for MSE and MAE
                reversed_blues_cmap = plt.cm.Blues.reversed()

                styled_df = results_df.style.background_gradient(cmap='Blues', subset=['R-squared', "Pearson's r"]) \
                                    .background_gradient(cmap=reversed_blues_cmap, subset=['MSE', 'MAE']) \
                                    .set_properties(**{'text-align': 'right', 'min-width': width})

                # Save the styled DataFrame to a separate HTML file
                html_path = os.path.join(self.root_dir, "data/processed", platform, f"styled_metrics_{target}.html")
                styled_df.to_html(html_path,
                                index=False)

                # Convert the styled DataFrame to a LaTeX table and save to a separate file
                latex_table = styled_df.hide(axis="index").to_latex()
                
                # Transform the LaTeX table to the desired format
                def replace_format(match):
                    bg = match.group("bg")
                    color = match.group("color")
                    value = match.group("value")
                    return r"\cellcolor[HTML]{" + bg + r"}\textcolor[HTML]{" + color + r"}{" + value + r"}"
                
                pattern = r"\\background-color#(?P<bg>[a-fA-F0-9]{6}) \\color#(?P<color>[a-fA-F0-9]{6}) (?P<value>[^&\\\\]+)"
                transformed_latex = re.sub(pattern, replace_format, latex_table)
                latex_packages = r"""
                \documentclass{article}

                \usepackage{colortbl}
                \usepackage{xcolor}
                \begin{document}
                """
                end_document = r"""
                \end{document}"""
                transformed_latex = latex_packages + transformed_latex + end_document
                # Save the transformed LaTeX table
                with open(os.path.join(self.root_dir, "data/processed", platform, f"latex_table_metrics_{target}.tex"), "w") as f:
                    f.write(transformed_latex)

                # # Render the LaTeX string to SVG using matplotlib
                # fig, ax = plt.subplots(figsize=(12, 8))
                # ax.text(0.5, 0.5, f'${transformed_latex}$', size=12, va='center', ha='center', usetex=True)
                # ax.axis('off')

                # # Save the figure as an SVG file
                # svg_path = os.path.join(self.root_dir, "data/processed", platform, f"latex_table_metrics_{target}.svg")
                # fig.savefig(svg_path, format='svg')
                # plt.close(fig)

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
    model_evaluator = EvaluateModels(root_dir = root_dir)
    model_evaluator.evaluate()