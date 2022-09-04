import glob
import tqdm
import os
import numpy as np
import pandas as pd
from torchvision.io.image import read_image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image
import ssl
from fastseg import MobileV3Small
from PIL import Image
import torch
from sklearn.model_selection import train_test_split
import shutil
import cv2

class FilterImage:
    """class for filtering out unusable SVI
        - Mapillary
            - Too much occlusion 
            - Too close to each other
        - GSV
            - Highway
    """
    def __init__(self, gsv_folder, mly_folder):
        self.gsv_folder = gsv_folder
        self.mly_folder = mly_folder
        
    def load_model(self):
        if torch.cuda.is_available():
            model = MobileV3Small.from_pretrained().cuda()
        else:
            model = MobileV3Small.from_pretrained()
        model.eval()
        self.model = model
        
    def segment_svi(self):
        filtered_list = []
        for image_file in tqdm.tqdm(glob.glob(os.path.join(self.mly_folder,"image/*.jpg"))):
            image = Image.open(image_file)
            labels = self.model.predict_one(image)
            labels_size = labels.size
            _, labels_count = np.unique(labels, return_counts=True)
            # refer to the labels: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
            occlusion_total = np.sum(labels_count[10:])
            if occlusion_total/labels_size > 0.2:
                mly_id = os.path.split(image_file)[1].replace(".jpg","")
                filtered_list.append(mly_id)
        filtered_df = pd.DataFrame(filtered_list, columns =["id"])
        filtered_df.to_csv(os.path.join(self.mly_folder,"filtered_images.csv"))

        # # Step 1: Initialize model with the best available weights
        # weights = FCN_ResNet50_Weights.DEFAULT
        # model = fcn_resnet50(weights=weights)
        # model.eval()
        
        # # Step 2: Initialize the inference transforms
        # preprocess = weights.transforms()
        
        # filtered_list = []
        # for image_file in tqdm.tqdm(glob.glob(os.path.join(self.mly_folder,"image/*.jpg"))):
        #     img = read_image(image_file)
        #     # Step 3: Apply inference preprocessing transforms
        #     batch = preprocess(img).unsqueeze(0)   
        #     # Step 4: Use the model and visualize the prediction
        #     prediction = model(batch)["out"]
        #     normalized_masks = prediction.softmax(dim=1)
        #     # class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
        #     # mask = normalized_masks[0, class_to_idx["dog"]]
        #     to_pil_image(normalized_masks).show()
        #     break
        #     # labels_size = labels.size
        #     # _, labels_count = np.unique(labels, return_counts=True)
        #     # refer to the labels: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
        # #     occlusion_total = np.sum(labels_count[10:])
        # #     if occlusion_total/labels_size > 0.2:
        # #         mly_id = os.path.split(image_file)[1].replace(".jpg","")
        # #         filtered_list.append(mly_id)
        # # filtered_df = pd.DataFrame(filtered_list, columns =["id"])
        # # filtered_df.to_csv(os.path.join(self.mly_folder,"filtered_images.csv"))
        

class FormatFolder():
    """class for structuring folders to be ready for GAN modelling
    """
    def __init__(self ,gsv_folder, mly_folder, new_folder):
        self.gsv_folder = gsv_folder
        self.mly_folder = mly_folder
        self.new_folder = new_folder
        os.makedirs(self.new_folder, exist_ok = True)
        
    def create_new_folder(self, random_state = 42):
        """create following directories: 
            - trainA: mly
            - trainB: gsv
            - testA: mly
            - testB: gsv
        """
        # get a list of files for mly and gsv
        mly_file_list = glob.glob(os.path.join(self.mly_folder,"image/*.jpg"))
        gsv_file_list = glob.glob(os.path.join(self.gsv_folder,"image/panorama/*.jpg"))
        
        # get size of mly to resize gsv panorama later
        mly_img_size = cv2.imread(mly_file_list[0]).shape

        # test and train split 50:50
        trainA, testA = train_test_split(mly_file_list, test_size=0.5, random_state=random_state)
        trainB, testB = train_test_split(gsv_file_list, test_size=0.5, random_state=random_state)

        # save the file lists to respective folders
        for i, dataset in enumerate([trainA, testA, trainB, testB]):
            if i % 2 == 0:
                train_test = "train"
            else:
                train_test = "test"
            if i < 2:
                A_B = "A"
            else:
                A_B = "B"
            folder_path = os.path.join(self.new_folder,train_test + A_B)
            os.makedirs(folder_path,exist_ok = True)
            for file_path in tqdm.tqdm(dataset):
                file_name = os.path.split(file_path)[1]
                if A_B == "B" and not os.path.exists(os.path.join(folder_path,file_name)):
                    image = cv2.imread(file_path)
                    resized = cv2.resize(image, (mly_img_size[1], mly_img_size[0]))
                    cv2.imwrite(os.path.join(folder_path,file_name), resized)
                if A_B == "A" and not os.path.exists(os.path.join(folder_path,file_name)):
                    shutil.copy2(file_path, os.path.join(folder_path, file_name))
            
            
        
if __name__ == '__main__':
    ssl._create_default_https_context = ssl._create_unverified_context
    root_dir = "/Volumes/exfat/road_shoulder_gan"
    gsv_folder = os.path.join(root_dir, "data/raw/gsv")
    mly_folder = os.path.join(root_dir, "data/raw/mapillary")
    new_folder = os.path.join(root_dir, "data/processed")
    # filter_image = FilterImage(gsv_folder, mly_folder)
    # filter_image.load_model()
    # filter_image.segment_svi()
    format_folder = FormatFolder(gsv_folder, mly_folder, new_folder)
    format_folder.create_new_folder()
    
    