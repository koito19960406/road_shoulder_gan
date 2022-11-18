import subprocess
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
from torchvision.models import resnet50
from torchvision import transforms
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torch.nn import functional as F

class FilterImage:
    """class for filtering out unusable SVI
        - Mapillary
            - Too much occlusion 
            - Too close to each other *optional for now
        - GSV
            - Highway
    """
    def __init__(self, pretrained_model_folder, gsv_folder, mly_folder):
        self.gsv_folder = gsv_folder
        self.mly_folder = mly_folder
        self.pretrained_model_folder = pretrained_model_folder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load models
        if torch.cuda.is_available():
            segmentation_model = MobileV3Small.from_pretrained().cuda()
        else:
            segmentation_model = MobileV3Small.from_pretrained()
        segmentation_model.eval()
        self.segmentation_model = segmentation_model
        
        # Model class must be defined somewhere
        arch = 'resnet50'
        model = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(os.path.join(self.pretrained_model_folder, "resnet50_places365.pth.tar"), map_location=torch.device(self.device))
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        model.eval()
        self.classification_model = model
        
    def segment_svi(self, update = False):
        if not update and os.path.exists(os.path.join(self.mly_folder,"metadata/filtered_images.csv")):
            print("The output file already exists, please set update to True if you want to update it")
        else:
            filtered_list = []
            for image_file in tqdm.tqdm(glob.glob(os.path.join(self.mly_folder,"image/*.jpg"))):
                try:
                    image = Image.open(image_file)
                    centre_crop = transforms.Compose([
                            transforms.Resize((512,512))
                    ])
                    image = centre_crop(image)
                    labels = self.segmentation_model.predict_one(image)
                    labels_size = labels.size
                    unique_labels, labels_count = np.unique(labels, return_counts=True)
                    # refer to the labels: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
                    occlusion_total = np.sum(labels_count[unique_labels>10])
                    if occlusion_total/labels_size > 0.1:
                        mly_id = os.path.split(image_file)[1].replace(".jpg","")
                        filtered_list.append(mly_id)
                except:
                    print("Error with " + image_file)
            filtered_df = pd.DataFrame(filtered_list, columns =["id"])
            filtered_df.to_csv(os.path.join(self.mly_folder,"metadata/filtered_images.csv"))

    def classify_svi(self, update = False):
        if not update and os.path.exists(os.path.join(self.gsv_folder,"metadata/filtered_images.csv")):
            print("The output file already exists, please set update to True if you want to update it")
        else:
            # load the image transformer
            centre_crop = transforms.Compose([
                    transforms.Resize((256,256)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            # load the class label
            file_name = os.path.join(self.pretrained_model_folder, 'categories_places365.txt')
            classes = list()
            with open(file_name) as class_file:
                for line in class_file:
                    classes.append(line.strip().split(' ')[0][3:])
            classes = tuple(classes)
            
            filtered_list = []
            for image_file in tqdm.tqdm(glob.glob(os.path.join(self.gsv_folder,"image/panorama/*.jpg"))):
                img = Image.open(image_file)
                input_img = V(centre_crop(img).unsqueeze(0))

                # forward pass
                logit = self.classification_model.forward(input_img)
                h_x = F.softmax(logit, 1).data.squeeze()
                probs, idx = h_x.sort(0, True)
                print(image_file)
                # output the prediction
                for i in range(0, 5):
                    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
                
                # refer to the labels: https://github.com/zhoubolei/places_devkit/blob/master/categories_places365.txt
                if (classes[idx[0]] == "highway") & (float(probs[0]) > 0.5):
                    mly_id = os.path.split(image_file)[1].replace(".jpg","")
                    filtered_list.append(mly_id)
            filtered_df = pd.DataFrame(filtered_list, columns =["id"])
            filtered_df.to_csv(os.path.join(self.gsv_folder,"metadata/filtered_images.csv"))
        
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
                    if row["distance"] <= threshold:
                        row_df = row.to_frame()
                        gsv_metadata_filtered = pd.concat([gsv_metadata_filtered, row_df.T], ignore_index = True)
                        break
            
                # save gsv_metadata_filtered as csv
                gsv_metadata_filtered.to_csv(os.path.join(self.gsv_folder,"metadata/gsv_metadata_filtered.csv"), index = False)
        
    def filter_with_cv_result(self):
        """function to filter gsv and mly images further based on the results of segmentation and classification
        make sure to run this after get_latest_gsv_only()
        """
        # run get_latest_gsv_only() if gsv_metadata_filtered.csv doesn't exist yet
        if not os.path.exists(os.path.join(self.gsv_folder,"metadata/gsv_metadata_filtered.csv")):
            self.get_latest_gsv_only()
        
        # import gsv metadata
        gsv_metadata = pd.read_csv(os.path.join(self.gsv_folder,"metadata/gsv_metadata_filtered.csv"))
        
        # import the results of cv filtering
        segment_filtered = set(pd.read_csv(os.path.join(self.mly_folder,"metadata/filtered_images.csv"))["id"].unique())
        classification_filtered = set(pd.read_csv(os.path.join(self.gsv_folder,"metadata/filtered_images.csv"))["id"].unique())
        
        # remove gsv and mly ids that match with segment_filtered and classification_filtered
        def extract_good_img(df):
            if (not df["mly_id"] in segment_filtered) and (not df["panoid"] in classification_filtered):
                return df
        # apply extract_good_img to get only good img
        tqdm.tqdm.pandas()
        gsv_metadata_filtered = gsv_metadata.progress_apply(extract_good_img, axis=1).dropna()
        
        # save the result
        gsv_metadata_filtered.to_csv(os.path.join(self.gsv_folder,"metadata/gsv_metadata_cv_filtered.csv"))
    
    def run_all(self):
        self.segment_svi()
        self.classify_svi()
        self.get_latest_gsv_only()
        self.filter_with_cv_result()
        
class FormatFolder():
    """class for structuring folders to be ready for GAN modelling
    """
    def __init__(self, gsv_folder, mly_folder, new_folder):
        self.gsv_folder = gsv_folder
        self.mly_folder = mly_folder
        self.new_folder = new_folder
        os.makedirs(self.new_folder, exist_ok = True)
        
    def stitch_gsv(self, gsv_id):
        """function to stitch image based on the goven gsv id

        Args:
            gsv_id (str): panoid

        Returns:
            numpy array: stitched img
        """
        img_list = glob.glob(os.path.join(self.gsv_folder, f"image/perspective/{gsv_id}*.png"))
        if len(img_list) > 0:
            img_agg = None
            for i in range(0,360,90):
                # get img that match with the direction in each loop
                img_file_match = list(filter(lambda x: f"Direction_{str(i)}" in x, img_list))[0]
                img_temp = cv2.imread(img_file_match)
                if img_agg is None:
                    img_agg = img_temp
                else:
                    img_agg = cv2.hconcat([img_agg, img_temp])
            return img_agg
        
    # define a function to check the file validity and save to the new folders
    def check_and_save(self, df, train_test, model):
        # get IDs
            panoid = df["panoid"]
            mly_id = df["mly_id"]
            # run the following if the image doesn't still exist
            if "cyclegan" in model:
                condition = not (os.path.exists(os.path.join(self.new_folder, model, train_test+"A", str(int(mly_id))+".jpg"))&\
                    os.path.exists(os.path.join(self.new_folder, model, train_test+"B", str(int(mly_id))+".jpg")))
            if "pix2pix" in model:
                condition = not (os.path.exists(os.path.join(self.new_folder, model+"_init", "A", train_test, str(int(mly_id))+".jpg"))&\
                    os.path.exists(os.path.join(self.new_folder, model+"_init", "B", train_test, str(int(mly_id))+".jpg")))
            if condition:
                # load mly image
                mly_image = cv2.imread(os.path.join(self.mly_folder,"image", str(int(mly_id)) + ".jpg"))
                if mly_image is not None:
                    # stitch gsv perspective images
                    gsv_image = self.stitch_gsv(panoid)
                    # save images if they are not empty
                    if gsv_image is not None:
                        # resize images to their average sizes
                        width = int((mly_image.shape[1]+gsv_image.shape[1])/2)
                        height = int((mly_image.shape[0]+gsv_image.shape[0])/2)
                        mly_resized = cv2.resize(mly_image, (width, height))
                        gsv_resized = cv2.resize(gsv_image, (width, height))
                        if "cyclegan" in model:
                            cv2.imwrite(os.path.join(self.new_folder, model, train_test+"B", str(int(mly_id))+".jpg"), mly_resized) 
                            cv2.imwrite(os.path.join(self.new_folder, model, train_test+"A", str(int(mly_id))+".jpg"), gsv_resized)
                        if "pix2pix" in model:
                            cv2.imwrite(os.path.join(self.new_folder, model+"_init", "B", train_test, str(int(mly_id))+".jpg"), mly_resized) 
                            cv2.imwrite(os.path.join(self.new_folder, model+"_init", "A", train_test, str(int(mly_id))+".jpg"), gsv_resized)
                            
    def create_new_folder(self, random_state = 42, model = "cyclegan", test_size = 0.1):
        """create following directories: 
            - trainA: mly
            - trainB: gsv
            - testA: mly
            - testB: gsv
        """
        # create folders
        if "cyclegan" in model:
            os.makedirs(os.path.join(self.new_folder,model,"trainA"), exist_ok = True)
            os.makedirs(os.path.join(self.new_folder,model,"trainB"), exist_ok = True)
            os.makedirs(os.path.join(self.new_folder,model,"testA"), exist_ok = True)
            os.makedirs(os.path.join(self.new_folder,model,"testB"), exist_ok = True)
        if "pix2pix" in model:
            os.makedirs(os.path.join(self.new_folder,model), exist_ok = True)
            os.makedirs(os.path.join(self.new_folder,model+"_init","A","train"), exist_ok = True)
            os.makedirs(os.path.join(self.new_folder,model+"_init","B","train"), exist_ok = True)
            os.makedirs(os.path.join(self.new_folder,model+"_init","A","test"), exist_ok = True)
            os.makedirs(os.path.join(self.new_folder,model+"_init","B","test"), exist_ok = True)
        # get a list of files for mly and gsv
        gsv_metadata = pd.read_csv(os.path.join(self.gsv_folder,"metadata/gsv_metadata_cv_filtered.csv"))

        # test and train split
        if test_size != 1:
            train, test = train_test_split(gsv_metadata, test_size=test_size, random_state=random_state)
        else:
            train, test = None, gsv_metadata
        print(train, test)
        
        # loop through train to copy gsv and mly images
        for df, train_test in zip([train,test], ["train","test"]):
            # apply check_and_save to df
            tqdm.tqdm.pandas()
            if df is not None:
                df.progress_apply(self.check_and_save, args=(train_test, model), axis=1)
            
    def prepare_pix2pix(self,model):
        fold_A = os.path.join(self.new_folder,model+"_init","A")
        fold_B = os.path.join(self.new_folder,model+"_init","B")
        subprocess.Popen([f"python src/models/pytorch-CycleGAN-and-pix2pix/datasets/combine_A_and_B.py --fold_A {fold_A} --fold_B {fold_B} --fold_AB {os.path.join(self.new_folder,model)} --no_multiprocessing"],
                        shell=True)
        pass
if __name__ == '__main__':
    ssl._create_default_https_context = ssl._create_unverified_context
    root_dir = "/Volumes/ExFAT/road_shoulder_gan"
    model_folder = os.path.join(root_dir, "data/external")
    gsv_folder = os.path.join(root_dir, "data/raw/gsv")
    mly_folder = os.path.join(root_dir, "data/raw/mapillary")
    new_folder = os.path.join(root_dir, "data/processed")
    # filter_image = FilterImage(model_folder, gsv_folder, mly_folder)
    # filter_image.get_latest_gsv_only()
    # filter_image.segment_svi()
    # filter_image.classify_svi()
    # filter_image.filter_with_cv_result()
    format_folder = FormatFolder(gsv_folder, mly_folder, new_folder)
    # format_folder.create_new_folder(model = "cyclegan_filtered")
    format_folder.create_new_folder(model = "pix2pix_filtered")
    format_folder.prepare_pix2pix("pix2pix_filtered")