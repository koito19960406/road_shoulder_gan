import subprocess
import os 
from pathlib import Path
import os
import pandas as pd
from tabulate import tabulate
import glob
import shutil
import tqdm
from zensvi.cv import Segmenter

from src.models.utils import move_results, CorrelationAnalysis

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

def create_fid_csv(root_folder):
    data = {'model_name': [], 'FID': []}
    root = Path(root_folder)

    for dir in root.rglob('fid'):
        fid_file = dir / 'fid.txt'
        if fid_file.exists():
            with fid_file.open('r') as f:
                fid_value = f.read().strip()
                data['model_name'].append(str(dir.parent.name))
                data['FID'].append(fid_value)

    df = pd.DataFrame(data)
    # Assign Category, Method and Parameters
    df['Category'] = df['model_name'].apply(lambda x: 'road_shoulder' if 'road_shoulder' in x else 'sidewalk')
    df['Method'] = df['model_name'].apply(lambda x: 'pix2pix' if 'pix2pix' in x else 'cyclegan')
    df['Parameters'] = df['model_name'].apply(lambda x: x.split('pix2pix_')[1] if 'pix2pix' in x else x.split('cyclegan_')[1])
    df["FID"] = pd.to_numeric(df["FID"])
    
    # Sort dataframe by the columns you want
    df = df.sort_values(['Category', 'FID', 'Method', 'Parameters'])

    # Drop the 'model_name' column
    df = df.drop(columns=['model_name'])
    
    # Move 'FID' to the end
    fid = round(df['FID'])
    df = df.drop(columns=['FID'])
    df['FID'] = fid

    # save as csv 
    df.to_csv(root_folder / 'fid.csv', index=False)
    
    # Convert dataframe to LaTeX table
    latex_table = tabulate(df, tablefmt='latex', headers='keys', showindex=False)

    # Write LaTeX table to file
    with open(root_folder / "fid.tex", 'w') as f:
        f.write(latex_table)

def pool_images(data_path, model_path, output_path):
    """Pool images of real gsv panorama, real gsv perspective, real mapillary, fake mapillary (by models) by image id

    Args:
        data_path (_type_): _description_
        model_path (_type_): _description_
        output_path (_type_): _description_
    """    
    for platform in ["sidewalk", "road_shoulder"]:
        # get a list of ids in one of the folders
        processed_path = data_path / 'processed' / platform
        # list directories in processed_path and check if the name contains 'cyclegan' (because they have an easier structure)
        model_data_folder = [x for x in processed_path.iterdir() if x.is_dir() and "cyclegan_filtered" in x.name][0]
        # get a list of ids in the folder testA
        img_id_list = [str(x.stem).replace(".jpg", "") for x in (model_data_folder / 'testA').iterdir()]
        # loop through img_id_list and find all the files with the same id in all the folders
        for img_id in tqdm.tqdm(img_id_list):
            # make output_path if it doesn't exist
            if (output_path / platform / img_id).exists():
                continue 
            (output_path / platform / img_id).mkdir(parents=True, exist_ok=True)
            # get real image of GSV panorama, GSV perspective and Mapillary
            real_gsv_pano = processed_path / "cyclegan_filtered" / "testA" / f"{img_id}.jpg"
            real_gsv_persepctive = processed_path / "cyclegan_filtered_perspective" / "testA" / f"{img_id}.jpg"
            real_mly = processed_path / "cyclegan_filtered" / "testB" / f"{img_id}.jpg"
            # find all the fake_B images with the same id in model_path that start with platform name and in test_latest\images with glob
            # for example, E:\road_shoulder_gan\models\road_shoulder_cyclegan_perspective\test_latest\images\0000000000_fake_B.png
            fake_mly_list = glob.glob(str(model_path / f"{platform}_*" / "test_latest" / "images" / f"{img_id}_fake_B.png"))
            # save real_gsv_pano, real_gsv_persepctive, real_mly and fake_mly_list to output_path in each id's unique folder
            shutil.copy2(real_gsv_pano, output_path / img_id / "real_gsv_pano.jpg")
            shutil.copy2(real_gsv_persepctive, output_path / img_id / "real_gsv_persepctive.jpg")
            shutil.copy2(real_mly, output_path / img_id / "real_mly.jpg")
            for fake_mly in fake_mly_list:
                # use the model name to name the fake mly image 
                # for example, E:\road_shoulder_gan\models\road_shoulder_cyclegan_perspective\test_latest\images\0000000000_fake_B.png should be named cycle_gan_perspective_fake_mly.jpg
                model_name = fake_mly.split("\\")[-4]
                shutil.copy2(fake_mly, output_path / img_id / f"{model_name}_fake_mly.jpg")
                
def segment(data_path, model_path):
    processed_path = data_path / 'processed'
    # list all the folders in model_path
    names_list = [x.name for x in model_path.iterdir() if x.is_dir()]
    for name in names_list:
        result_folder = os.path.join(model_path, f"{name}/test_latest/images")
        new_folder = os.path.join(model_path, f"{name}/gan_results")
        if not os.path.exists(new_folder):
            move_results(result_folder, new_folder)
        # segment input (mly and gsv) and output (fake) 
        if "sidewalk" in name:
            platform = "sidewalk"
        else:
            platform = "road_shoulder"
        if "pix2pix" in name:
            if "perspective" in name:
                mly_input_folder = os.path.join(processed_path,platform,"pix2pix_filtered_perspective_init/B/test") 
                gsv_input_folder = os.path.join(processed_path,platform,"pix2pix_filtered_perspective_init/A/test")
            else:
                mly_input_folder = os.path.join(processed_path,platform,"pix2pix_filtered_init/B/test") 
                gsv_input_folder = os.path.join(processed_path,platform,"pix2pix_filtered_init/A/test")
        if "cyclegan" in name:
            if "perspective" in name:
                mly_input_folder = os.path.join(processed_path,platform,"cyclegan_filtered_perspective/testB")
                gsv_input_folder = os.path.join(processed_path,platform,"cyclegan_filtered_perspective/testA")
            else:
                mly_input_folder = os.path.join(processed_path,platform, "cyclegan_filtered/testB")
                gsv_input_folder = os.path.join(processed_path,platform,"cyclegan_filtered/testA")
        for input_folder in [new_folder, mly_input_folder, gsv_input_folder]:
            # we need to use this gimmick here to set img_type
            #TODO create a separate test dataset with standardized folder names
            basename = os.path.basename(input_folder)
            if "pix2pix" in name and basename!="gan_results":
                img_type = "test" + os.path.split(input_folder)[0][-1]
            else:
                img_type = os.path.basename(input_folder)
            #TODO update folder names
            img_output_folder = os.path.join(model_path, name, "segmented",img_type)
            csv_output_folder = os.path.join(model_path, name, "segmentation_result",img_type)
            # create output folders
            os.makedirs(img_output_folder, exist_ok=True)
            os.makedirs(csv_output_folder, exist_ok=True)
            # initialize the segmenter
            segmenter  = Segmenter()
            pixel_ratio_save_format = ["csv"]
            if not os.path.exists(os.path.join(csv_output_folder, "pixel_ratios.csv")):
                segmenter.segment(input_folder,
                                dir_image_output = img_output_folder,
                                dir_segmentation_summary_output = csv_output_folder,
                                pixel_ratio_save_format = pixel_ratio_save_format)
            # img_seg = segmentation.ImageSegmentationSimple(input_folder, img_output_folder, csv_output_folder)
            # img_seg.segment_svi(batch_size_store=100)
            # img_seg.calculate_ratio()
        gsv_result_csv = os.path.join(model_path, name, "segmentation_result/testA/pixel_ratios.csv")
        mly_result_csv = os.path.join(model_path, name, "segmentation_result/testB/pixel_ratios.csv")
        gan_result_csv = os.path.join(model_path, name, "segmentation_result/gan_results/pixel_ratios.csv")
        output_folder = os.path.join(model_path, name, "segmentation_result")
        correlation_analysis = CorrelationAnalysis(gsv_result_csv, mly_result_csv, gan_result_csv, output_folder)
        correlation_analysis.merge_save()

if __name__ == "__main__":
    # Set relative paths
    script_path_relative = Path("road_shoulder_gan/src/models/pytorch-CycleGAN-and-pix2pix/test.py")
    args_folder_path_relative = Path("road_shoulder_gan/configs")
    models_folder_path_relative = Path("road_shoulder_gan/models")
    data_folder_path_relative = Path("road_shoulder_gan/data")
    
    test_script = None
    args_folder_path = None
    models_folder_path = None
    # list of drives from D to Z
    drives = [f"{chr(x)}:/" for x in range(ord('D'), ord('Z') + 1)] + ["/Volumes/ExFAT/"] 
    for drive in drives:
        temp_script_path = Path(drive) / script_path_relative
        temp_args_folder_path = Path(drive) / args_folder_path_relative
        temp_models_folder_path = Path(drive) / models_folder_path_relative
        temp_data_folder_path = Path(drive) / data_folder_path_relative
        if (Path(drive) / "road_shoulder_gan").exists():
            root_dir = Path(drive) / "road_shoulder_gan"
        if temp_script_path.exists():
            test_script = temp_script_path
        if temp_args_folder_path.exists():
            args_folder_path = temp_args_folder_path
        if temp_models_folder_path.exists():
            models_folder_path = temp_models_folder_path
        if temp_data_folder_path.exists():
            data_folder_path = temp_data_folder_path

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
    # create fid csv
    print("Creating fid tables")
    create_fid_csv(models_folder_path)
    
    # pool images
    print("Pooling images")
    pool_images(data_folder_path, models_folder_path, root_dir / "reports/images")

    # segment images
    print("Segmenting images")
    segment(data_folder_path, models_folder_path)