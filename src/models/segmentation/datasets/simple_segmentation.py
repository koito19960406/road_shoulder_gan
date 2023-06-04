from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image
import cv2 
import torch
import tqdm
from multiprocessing.pool import ThreadPool
import numpy as np
import itertools
import glob
from pyarrow import feather
import dask
import dask.dataframe as dd

class SimpleSegmentationInferDataset(Dataset):
    def __init__(self, img_dir, feature_extractor, csv_output_folder):
        super().__init__()
        global segmented_image_file_set
        # set variables
        self.img_dir = img_dir
        self.feature_extractor = feature_extractor
        feather_list = glob.glob(os.path.join(csv_output_folder, "*.feather"))
        if len(feather_list) > 0:
            dfs = [dask.delayed(feather.read_feather)(f) for f in feather_list]
            df = dd.from_delayed(dfs)
            segmented_image_file_set = set(df.drop_duplicates().compute()["file_name"].unique())
        else:
            segmented_image_file_set = set()
        self.cpu_num = os.cpu_count()
        # # load images and pass to feature extractor
        # image_list = []
        # for file_name in tqdm.tqdm(os.listdir(self.img_dir)):
        #     file_name_key = file_name[:-4]
        #     if not gsv_invalid_file["file_name"].str.contains(file_name_key).any():
        #         # run the following only if the file_name doesn't exist in the file yet
        #         if len(segmented_image_file_list) > 0:
        #             # skip if it's already been sgemented or invalids
        #             if not segmented_image_file_list.str.contains(file_name_key).any():
        #                 image_list.append(os.path.join(self.img_dir,file_name))
        #             else:
        #                 continue
        #         else:
        #             image_list.append(os.path.join(self.img_dir,file_name))
        #     else:
        #         continue
        # self.images = image_list
        
        
        def check_file(image_file):
            file_name_key = image_file[:-4]
            print(file_name_key)
            # run the following only if the file_name doesn't exist in the file yet
            if len(segmented_image_file_set) > 0:
                # skip if it's already been sgemented or invalids
                if file_name_key not in segmented_image_file_set:
                    return os.path.join(self.img_dir,image_file)
                else:
                    return
            else:
                return os.path.join(self.img_dir,image_file)
            
        # apply the function to each row
        def parallelize_function(image_file_list):
            output_list = []
            for image_file in tqdm.tqdm(image_file_list):
                output_temp = check_file(image_file)
                if output_temp is not None:
                    output_list.append(output_temp)
            return output_list
        
        # split the input df and map the input function
        def parallelize_list(cpu_num, input_list, outer_func):
            num_processes = cpu_num
            pool = ThreadPool(processes=num_processes)
            input_list_split = np.array_split(input_list, num_processes)
            output_list = list(itertools.chain.from_iterable(pool.map(outer_func, input_list_split)))
            return output_list
        
        output_list = parallelize_list(self.cpu_num, os.listdir(self.img_dir), parallelize_function)
        self.images = output_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = torch.squeeze(self.feature_extractor(images=image, return_tensors="pt").pixel_values)
        # file base name
        file_name = os.path.basename(self.images[idx])[:-4]
        return inputs, file_name