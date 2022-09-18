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
        
    def get_latest_gsv_only(self, threshold=10):
        # load gsv_metadata with distance
        gsv_metadata = pd.read_csv(os.path.join(self.gsv_folder,"metadata/gsv_metadata_dist.csv"))
        # extract a list of unique mly id
        unique_mly_id_list = gsv_metadata["mly_id"].unique()
        # loop through unique mly id
        for unique_mly_id in tqdm.tqdm(unique_mly_id_list):
            if os.path.exists(os.path.join(self.gsv_folder,"metadata/gsv_metadata_filtered.csv")):
                gsv_metadata_filtered = pd.read_csv(os.path.join(self.gsv_folder,"metadata/gsv_metadata_filtered.csv"))
            else:    
                # initialize the final df with latest gsv within [threshold] meters from the mly svi
                gsv_metadata_filtered = pd.DataFrame(columns=gsv_metadata.columns)
            # run the following if mly_id is not in the final data yet
            if not unique_mly_id in gsv_metadata_filtered["mly_id"].tolist():
                # filter gsv_metadata to get rows with unique_mly_id
                gsv_metadata_temp = gsv_metadata[gsv_metadata["mly_id"]==unique_mly_id]
                # sort by year and month
                gsv_metadata_temp = gsv_metadata_temp.sort_values(by=['year', 'month'], ascending=False)
                # loop through each row
                for _, row in gsv_metadata_temp.iterrows():
                    # concatenate to gsv_metadata_filtered if the distance is within the threshold
                    # continute other wise
                    if row["distance"] <= 10:
                        row_df = row.to_frame()
                        gsv_metadata_filtered = pd.concat([gsv_metadata_filtered, row_df.T], ignore_index = True)
                        break
            
                # save gsv_metadata_filtered as csv
                gsv_metadata_filtered.to_csv(os.path.join(self.gsv_folder,"metadata/gsv_metadata_filtered.csv"), index = False)
        
class FormatFolder():
    """class for structuring folders to be ready for GAN modelling
    """
    def __init__(self ,gsv_folder, mly_folder, new_folder):
        self.gsv_folder = gsv_folder
        self.mly_folder = mly_folder
        self.new_folder = new_folder
        os.makedirs(self.new_folder, exist_ok = True)
        
    def create_new_folder(self, random_state = 42, model = "cyclegan"):
        """create following directories: 
            - trainA: mly
            - trainB: gsv
            - testA: mly
            - testB: gsv
        """
        # create folders
        os.makedirs(os.path.join(self.new_folder,model,"trainA"), exist_ok = True)
        os.makedirs(os.path.join(self.new_folder,model,"trainB"), exist_ok = True)
        os.makedirs(os.path.join(self.new_folder,model,"testA"), exist_ok = True)
        os.makedirs(os.path.join(self.new_folder,model,"testB"), exist_ok = True)
        # get a list of files for mly and gsv
        gsv_metadata = pd.read_csv(os.path.join(self.gsv_folder,"metadata/gsv_metadata_filtered.csv"))
        
        # # filter gsv_file_list
        # panoid = gsv_metadata["panoid"].tolist()
        # for gsv_file in gsv_file_list:
        #     panoid_extracted = os.path.split(gsv_file)[1]
        #     panoid_extracted = panoid_extracted.replace(".jpg","")
        #     # remove if not in the panoid list
        #     if not panoid_extracted in panoid:
        #         gsv_file_list.remove(gsv_file)
            
        # # get size of mly to resize gsv panorama later
        # mly_img_size = cv2.imread(mly_file_list[0]).shape

        # test and train split 50:50
        train, test = train_test_split(gsv_metadata, test_size=0.5, random_state=random_state)
        print(train, test)
        
        # loop through train to copy gsv and mly images
        for df, train_test in zip([train,test], ["train","test"]):
            for _, row in tqdm.tqdm(df.iterrows(), total=len(df.index)):
                # get IDs
                panoid = row["panoid"]
                mly_id = row["mly_id"]
                
                if not (os.path.exists(os.path.join(self.new_folder, model, train_test+"A", str(mly_id)+".jpg"))&\
                    os.path.exists(os.path.join(self.new_folder, model, train_test+"B", str(mly_id)+".jpg"))):
                    # load mly image
                    mly_image = cv2.imread(os.path.join(self.mly_folder,"image", str(mly_id) + ".jpg"))
                    if mly_image is not None:
                        # resize GSV image
                        gsv_image = cv2.imread(os.path.join(self.gsv_folder,"image/panorama",panoid+".jpg"))
                        # save images if they are not empty
                        if gsv_image is not None:
                            gsv_resized = cv2.resize(gsv_image, (mly_image.shape[1], mly_image.shape[0]))
                            cv2.imwrite(os.path.join(self.new_folder, model, train_test+"B", str(mly_id)+".jpg"), mly_image) 
                            cv2.imwrite(os.path.join(self.new_folder, model, train_test+"A", str(mly_id)+".jpg"), gsv_resized)
                    
                    
            # trainB, testB = train_test_split(gsv_file_list, test_size=0.5, random_state=random_state)

            # # save the file lists to respective folders
            # for i, dataset in enumerate([trainA, testA, trainB, testB]):
            #     if i % 2 == 0:
            #         train_test = "train"
            #     else:
            #         train_test = "test"
            #     if i < 2:
            #         A_B = "A"
            #     else:
            #         A_B = "B"
            #     folder_path = os.path.join(self.new_folder,model,train_test + A_B)
            #     os.makedirs(folder_path,exist_ok = True)
            #     for file_path in tqdm.tqdm(dataset):
            #         file_name = os.path.split(file_path)[1]
            #         if A_B == "B" and not os.path.exists(os.path.join(folder_path,file_name)):
            #             panoid = file_name.replace(".jpg","")
            #             mly_id = gsv_metadata[gsv_metadata["panoid"]==panoid]["mly"]
            #             image = cv2.imread(file_path)
            #             resized = cv2.resize(image, (mly_img_size[1], mly_img_size[0]))
            #             cv2.imwrite(os.path.join(folder_path,file_name), resized)
            #         if A_B == "A" and not os.path.exists(os.path.join(folder_path,file_name)):
            #             shutil.copy2(file_path, os.path.join(folder_path, file_name))
            
            
        
if __name__ == '__main__':
    ssl._create_default_https_context = ssl._create_unverified_context
    root_dir = "/Volumes/exfat/road_shoulder_gan"
    gsv_folder = os.path.join(root_dir, "data/raw/gsv")
    mly_folder = os.path.join(root_dir, "data/raw/mapillary")
    new_folder = os.path.join(root_dir, "data/processed")
    # filter_image = FilterImage(gsv_folder, mly_folder)
    # filter_image.get_latest_gsv_only()
    # filter_image.load_model()
    # filter_image.segment_svi()
    format_folder = FormatFolder(gsv_folder, mly_folder, new_folder)
    format_folder.create_new_folder()
    
    