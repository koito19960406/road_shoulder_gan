import mapillary as mly
import requests
import os
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
import streetview
import pandas as pd
from get_img import my_task
from perspective.tool import Equirectangular
import threading
import cv2

class Downloader:
    """class for downloading images from mapillary
    """
    def __init__(self, output_folder, MLY_ACCESS_TOKEN):
        # set access token for mapillary
        mly.interface.set_access_token(MLY_ACCESS_TOKEN)
        # set output folders
        self.output_folder = output_folder
        # mapillary
        self.mly_output_folder = os.path.join(self.output_folder,"mapillary")
        self.mly_image_output_folder = os.path.join(self.mly_output_folder,"image")
        os.makedirs(self.mly_image_output_folder, exist_ok = True)
        self.mly_metadata_output_folder = os.path.join(self.mly_output_folder,"metadata")
        os.makedirs(self.mly_metadata_output_folder, exist_ok = True)
        # gsv
        self.gsv_output_folder = os.os.path.join(self.output_folder,"gsv")
        self.gsv_image_output_folder = os.path.join(self.gsv_output_folder,"image")
        os.makedirs(self.gsv_image_output_folder, exist_ok = True)
        self.gsv_metadata_output_folder = os.path.join(self.gsv_output_folder,"metadata")
        os.makedirs(self.gsv_metadata_output_folder, exist_ok = True)
    
        # set the number of cpus
        self._cpus = cpu_count()
        pass
    
    def get_mly_image_id(self, bbox, ORGANIZATION_ID):
        """function to retrieve image id from bbox and organization id and store
        as response and save as geojson

        Args:
            bbox (dict): bbox
            ORGANIZATION_ID (str): organization id
        """
        response = mly.interface.images_in_bbox(bbox, organization_id = ORGANIZATION_ID)
        # store response as property
        self.response = response
        # save response in the output_folder as geojson
        response.to_file(os.path.join(self.mly_metadata_output_folder, "response.geojson"), driver='GeoJSON')
        
    def get_mly_image_id_fake(self):
        """function to retrieve image id from bbox and organization id

        Args:
            bbox (dict): bbox
            ORGANIZATION_ID (str): organization id

        Returns:
            response: geojson containing image info
        """
        bbox = {'south': 1.295949,
            'north': 1.296,
            'west': 103.769733,
            'east': 103.77}
        response = mly.interface.images_in_bbox(bbox)
        return response
    
    def get_mly_url(self):
        """function to retrieve list of url from mapillary
        and store the url list as property
        
        """
        id_list = self.response["id"].tolist()
        url_list = []
        for id in id_list:
            url = mly.interface.image_thumbnail(image_id=id, resolution=2048)
            url_list.append(url)
        self.url_list = url_list
    
    def download_mly_image(self):            
        # create a simple function to download files from url
        def download_files(args):
            url, fn = args[0], args[1]
            try:
                r = requests.get(url)
                with open(fn, 'wb') as f:
                    f.write(r.content)
            except Exception as e:
                print('Exception in download_url():', e)
        
        # create args
        file_name_list = [os.path.join(self.mly_image_output_folder,str(id)+".jpg") for id in self.id_list]
        args = zip(self.url_list, file_name_list)
        results = ThreadPool(self._cpus - 1).imap_unordered(download_files, args)
        for result in results:
            print('url:', result[0], 'time (s):', result[1]) 
            
    def download_gsv(self):
        # create output folders
        dir_save = os.path.join(self.gsv_image_output_folder,"panorama")
        os.makedirs(dir_save, exist_ok = True)
        # set path to the pid csv file
        path_pid = os.path.join(self.gsv_metadata_output_folder,"gsv_metadata.csv")
        # set path to user agent info csv file
        ua_path = "src/data/get_img/utils/UserAgent.csv"
        # set path to the 1st error log csv file
        log_path = os.path.join(self.gsv_metadata_output_folder,"gsv_metadata_error_1.csv")
        # Number of threads: num of cpus
        nthreads = self.cpu_num
        # set user agent to avoid gettting banned
        UA = my_task.get_ua(path=ua_path)
        # run the main function to download gsv as the 1st try
        my_task.main(UA, path_pid, dir_save, log_path, nthreads)
        # some good pids will be missed when 1 bad pid is found in multithreading
        # so run again the main function with error log file and only 1 thread
        log_path_2 = os.path.join(self.gsv_metadata_output_folder,"gsv_metadata_error_2.csv")
        my_task.main(UA, log_path, dir_save, log_path_2, 1)
            
            
    def transform_pano_to_perspective(self):
        # define function to run in the threading
        def run(path_input_raw, path_output_c,show_size):
            # get perspective at each 90 degree
            thetas = [0, 90, 180, 270]
            FOV = 90

            # set aspect as 9 to 16
            aspects_v = (2.25, 4)
            aspects = (9, 16)

            img_raw = cv2.imread(path_input_raw, cv2.IMREAD_COLOR)
            equ_raw = Equirectangular(img_raw)

            for theta in thetas:
                height = int(aspects_v[0] * show_size)
                width = int(aspects_v[1] * show_size)
                aspect_name = '%s--%s'%(aspects[0], aspects[1])
                img_raw = equ_raw.GetPerspective(FOV, theta, 0, height, width)
                path_output = path_output_c[:]
                path_output_raw = path_output.replace('.png', '_Direction_%s_FOV_%s_aspect_%s_raw.png'%(theta, FOV, aspect_name))
                cv2.imwrite(path_output_raw, img_raw)
        
        # set and create directories       
        dir_input_raw = os.path.join(self.gsv_image_output_folder,"panorama")
        dir_out_show = os.path.join(self.gsv_image_output_folder,"perspective")
        os.makedirs(dir_out_show, exist_ok = True)

        # set parameters
        index = 0
        show_size = 50  # 像素大小 * 4 或者 3
        threads = []
        num_thread = self.cpu_num

        for name in os.listdir(dir_input_raw):
            index += 1

            path_input_raw = os.path.join(dir_input_raw, name)
            path_output = os.path.join(dir_out_show, name.replace('jpg','png'))

            if index % num_thread == 0:
                print('Now:', index)
                t = threading.Thread(target=run, args=(path_input_raw, path_output, show_size,))
                threads.append(t)
                for t in threads:
                    t.setDaemon(True)
                    t.start()
                t.join()
                threads = []
            else:
                t = threading.Thread(target=run, args=(path_input_raw, path_output, show_size,))
                threads.append(t)