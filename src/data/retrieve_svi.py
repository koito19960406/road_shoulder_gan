import mapillary as mly
import requests
import os
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
import streetview
import pandas as pd
from src.data.get_img import my_task
from src.data.perspective.tool import Equirectangular
import threading
import cv2
import numpy as np
import geopandas as gpd
from geopy import distance
from shapely.geometry import box, Polygon, MultiPolygon, GeometryCollection
import logging
import queue
import glob
import tqdm

class Downloader:
    """class for downloading images from mapillary
    """
    def __init__(self, input_folder, output_folder, MLY_ACCESS_TOKEN):
        # set access token for mapillary
        mly.interface.set_access_token(MLY_ACCESS_TOKEN)
        # set input folders
        self.input_folder = input_folder
        # set output folders
        self.output_folder = output_folder
        # mapillary
        self.mly_output_folder = os.path.join(self.output_folder,"mapillary")
        self.mly_image_output_folder = os.path.join(self.mly_output_folder,"image")
        os.makedirs(self.mly_image_output_folder, exist_ok = True)
        self.mly_metadata_output_folder = os.path.join(self.mly_output_folder,"metadata")
        os.makedirs(self.mly_metadata_output_folder, exist_ok = True)
        # gsv
        self.gsv_output_folder = os.path.join(self.output_folder,"gsv")
        self.gsv_image_output_folder = os.path.join(self.gsv_output_folder,"image")
        os.makedirs(self.gsv_image_output_folder, exist_ok = True)
        self.gsv_metadata_output_folder = os.path.join(self.gsv_output_folder,"metadata")
        os.makedirs(self.gsv_metadata_output_folder, exist_ok = True)
    
        # set the number of cpus
        self.cpu_num = cpu_count()
        pass
    
    # use a function to create a geomerty from bbox
    def create_geometry(self, bbox):
        return Polygon([[bbox["east"], bbox["south"]],
                        [bbox["west"], bbox["south"]],
                        [bbox["west"], bbox["north"]],
                        [bbox["east"], bbox["north"]]])
    
    def katana(self, geometry, threshold, count=0):
        """Split a Polygon into two parts across it's shortest dimension"""
        bounds = geometry.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        if max(width, height) <= threshold or count == 250:
            # either the polygon is smaller than the threshold, or the maximum
            # number of recursions has been reached
            return [geometry]
        if height >= width:
            # split left to right
            a = box(bounds[0], bounds[1], bounds[2], bounds[1]+height/2)
            b = box(bounds[0], bounds[1]+height/2, bounds[2], bounds[3])
        else:
            # split top to bottom
            a = box(bounds[0], bounds[1], bounds[0]+width/2, bounds[3])
            b = box(bounds[0]+width/2, bounds[1], bounds[2], bounds[3])
        result = []
        for d in (a, b,):
            c = geometry.intersection(d)
            if not isinstance(c, GeometryCollection):
                c = [c]
            for e in c:
                if isinstance(e, (Polygon, MultiPolygon)):
                    result.extend(self.katana(e, threshold, count+1))
        if count > 0:
            return result
        # convert multipart into singlepart
        final_result = []
        for g in result:
            if isinstance(g, MultiPolygon):
                final_result.extend(g)
            else:
                final_result.append(g)
        return final_result
    
    def get_mly_image_id_paralell(self, bbox, ORGANIZATION_ID, update = False):
        """function to retrieve image id from bbox and organization id and store
        as response and save as geojson by parelell computation with multi-threads

        Args:
            bbox (dict): bbox
            ORGANIZATION_ID (str): organization id
        """
        # if users set update false and the output file already exists, then let them know how to run
        if not update and os.path.exists(os.path.join(self.mly_metadata_output_folder, "response.geojson")):
            print("The output file already exists, please set update to True if you want to update it")
        else:
            # split bbox into smaller bboxs
            geometry = self.create_geometry(bbox)
            small_geometry_list = self.katana(geometry, 0.01)
            def geometry_to_bbox(geometry):
                bounds = geometry.bounds
                return {
                    'east': bounds[0],
                    'south': bounds[1],
                    'west': bounds[2],
                    'north': bounds[3]
                    }
            small_bbox_list = list(map(geometry_to_bbox, small_geometry_list))

            #set up the queue to hold all the ids
            q = queue.Queue(maxsize=0)
            
            #Populating Queue with tasks
            response_dict = {"type": "FeatureCollection", "features": [{}]}
            #load up the queue with the bbox to fetch and the index for each job (as a tuple):
            for i in range(len(small_bbox_list)):
                #need the index and the url in each queue item.
                q.put(small_bbox_list[i])
            # Threaded function for queue processing.
            def fetch_id(q, response_dict):
                while not q.empty():
                    small_bbox = q.get()                      #fetch new work from the Queue
                    try:
                        response = mly.interface.images_in_bbox(small_bbox, organization_id = ORGANIZATION_ID)
                        logging.info("Requested..." + small_bbox)
                        response_dict["features"].append(response['features'])
                    except:
                        logging.error('Retrieved no response')
                        response_dict["features"].append({})
                    #signal to the queue that task has been processed
                    q.task_done()
                return True
            
            #Starting worker threads on queue processing
            for i in range(self.cpu_num):
                logging.debug('Starting thread ', i)
                t = threading.Thread(target=fetch_id, args=(q,response_dict,))
                t.setDaemon(True)    #setting threads as "daemon" allows main program to 
                                     #exit eventually even if these dont finish 
                                     #correctly.
                t.start()
            #now we wait until the queue has been processed
            q.join()
            logging.info('All tasks completed.')
            # store response_dict as gdf
            response_df = gpd.read_file(response_dict)
            # save response in the output_folder as geojson
            response_df.to_file(os.path.join(self.mly_metadata_output_folder, "response.geojson"), driver='GeoJSON')
        pass

    def get_mly_image_id(self, boundary, ORGANIZATION_ID, update = False):
        """function to retrieve image id from bbox and organization id and store
        as response and save as geojson
        Args:
            boundary (dict): boundary to get image id from 
            ORGANIZATION_ID (str): organization id
        """
        if not update and os.path.exists(os.path.join(self.mly_metadata_output_folder, "response.geojson")):
            print("The output file already exists, please set update to True if you want to update it")
        else:
            if boundary.get("west") != None:
                response = mly.interface.images_in_bbox(boundary, organization_id = ORGANIZATION_ID)
            else:
                response = mly.interface.images_in_geojson(boundary, organization_id = ORGANIZATION_ID)
            response_df = gpd.read_file(response)
            # save response in the output_folder as geojson
            response_df.to_file(os.path.join(self.mly_metadata_output_folder, "response.geojson"), driver='GeoJSON')
            
    def get_mly_image_id_batch(self, bbox, ORGANIZATION_ID, update = False):
        """function to retrieve image id from bbox and organization id and store
        as response and save as geojson by batch
        Args:
            bbox (dict): bbox
            ORGANIZATION_ID (str): organization id
        """
        if not update and os.path.exists(os.path.join(self.mly_metadata_output_folder, "response.geojson")):
            print("The output file already exists, please set update to True if you want to update it")
        else:
            # split bbox into smaller bboxs
            geometry = self.create_geometry(bbox)
            small_geometry_list = self.katana(geometry, 0.01)
            def geometry_to_bbox(geometry):
                bounds = geometry.bounds
                return {
                    'east': bounds[0],
                    'south': bounds[1],
                    'west': bounds[2],
                    'north': bounds[3]
                    }
            small_bbox_list = list(map(geometry_to_bbox, small_geometry_list))
            
            # loop through small_bbox_list and save the response as separate geojson files
            for index, small_bbox in enumerate(small_geometry_list):
                mly_id_output_folder = os.path.join(self.mly_metadata_output_folder, "id")
                os.makedirs(mly_id_output_folder, exist_ok = True)
                small_bbox_gdf = gpd.GeoDataFrame(index=[0] ,crs='epsg:4326', geometry=[small_bbox])
                small_bbox_gdf.to_file(os.path.join(mly_id_output_folder, "{:03d}".format(index) + "_response.geojson"), driver='GeoJSON')
                
                # response = mly.interface.images_in_bbox(small_bbox, organization_id = ORGANIZATION_ID)
                # response_df = gpd.read_file(response)
                # # save response in the output_folder as geojson
                
                # response_df.to_file(os.path.join(mly_id_output_folder, "{:03d}".format(index) + "_response.geojson"), driver='GeoJSON')
        
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
    
    def get_mly_url(self, update = False):
        """function to retrieve list of url from mapillary
        and save the url list as csv file
        
        """
        if not update and os.path.exists(os.path.join(self.mly_metadata_output_folder, 'id_url_list.csv')):
            print("The output file already exists, please set update to True if you want to update it")
        else:
            response = gpd.read_file(os.path.join(self.mly_metadata_output_folder, "response.geojson"))
            id_list = response["id"].tolist()
            #set up the queue to hold all the ids
            q = queue.Queue(maxsize=0)
            
            #Populating Queue with tasks
            id_url_list = []
            #load up the queue with the bbox to fetch and the index for each job (as a tuple):
            for i in range(len(id_list)):
                #need the index and the url in each queue item.
                q.put(id_list[i])
            
            # Threaded function for queue processing.
            def fetch_url(q, id_url_list):
                while not q.empty():
                    image_id = str(q.get())                      #fetch new work from the Queue
                    try:
                        url = mly.interface.image_thumbnail(image_id=image_id, resolution=2048)
                        logging.info("Requested..." + image_id)
                        id_url_list.append([image_id, url])
                    except:
                        logging.error('Retrieved no response')
                    #signal to the queue that task has been processed
                    q.task_done()
                return True
            
            #Starting worker threads on queue processing
            for i in range(self.cpu_num):
                logging.debug('Starting thread ', i)
                t = threading.Thread(target=fetch_url, args=(q, id_url_list,))
                t.setDaemon(True)    #setting threads as "daemon" allows main program to 
                                     #exit eventually even if these dont finish 
                                     #correctly.
                t.start()
            #now we wait until the queue has been processed
            q.join()
            logging.info('All tasks completed.')
            
            # save to csv file
            url_df = pd.DataFrame(id_url_list, columns=["id","url"])
            url_df.to_csv(os.path.join(self.mly_metadata_output_folder, 'id_url_list.csv'), index=False)    
        pass
        
    def download_mly_image(self):    
        """function to download mapillary svi with multiple threads
        """
        # create a simple function to download files from url
        def download_files(url, fn):
            try:
                if not os.path.exists(fn):
                    r = requests.get(url)
                    with open(fn, 'wb') as f:
                        f.write(r.content)
            except Exception as e:
                print('Exception in download_url():', e)
        
        # load id and url lists
        id_url_list = pd.read_csv(os.path.join(self.mly_metadata_output_folder, 'id_url_list.csv'))
        id_list = id_url_list["id"].tolist()
        url_list = id_url_list["url"].tolist()
        file_name_list = [os.path.join(self.mly_image_output_folder,str(id)+".jpg") for id in id_list]
        # for url, fn in tqdm.tqdm(zip(url_list, file_name_list),total=len(url_list)):
        #     r = requests.get(url)
        #     if not os.path.exists(fn):
        #         with open(fn, 'wb') as f:
        #             f.write(r.content)
        # create args
        # args = zip(url_list, file_name_list)
        # results = ThreadPool(self.cpu_num - 1).imap_unordered(download_files, args)
        # for result in results:
        #     print('url:', result[0], 'time (s):', result[1]) 
        threads = []
        for i, (url, fn) in tqdm.tqdm(enumerate(zip(url_list, file_name_list)),total=len(url_list)):
            if i % self.cpu_num == 0:
                t = threading.Thread(target=download_files, args=(url, fn,))
                threads.append(t)
                for t in threads:
                    t.setDaemon(True)
                    t.start()
                t.join()
                threads = []
            else:
                t = threading.Thread(target=download_files, args=(url, fn,))
                threads.append(t)
        
    
    def get_gsv_metadata_multiprocessing(self, update = False):
        """get GSV metadata (e.g., panoids, years, months, etc) for each location and store the result as self.panoids
        """
        global get_gsv_metadata
        global parallelize_function
        
        # define a function to retrieve GSV metadata based on each row of the input GeoDataFrame
        def get_gsv_metadata(row):
            try:
                panoids = streetview.panoids(lat=row.geometry.y, lon=row.geometry.x)
                panoids = pd.DataFrame.from_dict(panoids)
                panoids["input_lat"] = row.geometry.y
                panoids["input_lon"] = row.geometry.x
                panoids["mly_id"] = row["id"]
                return panoids
            except:
                print(row.geometry.y,row.geometry.x)
                print(panoids)
                return
            
        # apply the function to each row
        def parallelize_function(input_df):
            output_df = pd.DataFrame()
            for _, row in tqdm.tqdm(input_df.iterrows(),total=len(input_df.index)):
                output_df_temp = get_gsv_metadata(row)
                output_df = pd.concat([output_df,output_df_temp], ignore_index = True)
            return output_df
        
        # split the input df and map the input function
        def parallelize_dataframe(input_df, outer_func):
            num_processes = self.cpu_num
            pool = ThreadPool(processes=num_processes)
            input_df_split = np.array_split(input_df, num_processes)
            output_df = pd.concat(pool.map(outer_func, input_df_split), ignore_index = True)
            return output_df
        
        # run the parallelized functions if the metadata doesn't exist yet
        if not update and os.path.exists(os.path.join(self.gsv_metadata_output_folder, "gsv_metadata.csv")):
            print("The output file already exists, please set update to True if you want to update it")
            
        else:
            point_gdf = gpd.read_file(os.path.join(self.mly_metadata_output_folder, "response.geojson"))
            df_output = parallelize_dataframe(point_gdf, parallelize_function)
        
            # save df_output
            df_output.to_csv(os.path.join(self.gsv_metadata_output_folder, "gsv_metadata.csv"), index = False)
            
    # calculate the distance from the original input location
    def calc_dist(self, update = False):
        if not update and os.path.join(self.gsv_metadata_output_folder, "gsv_metadata_dist.csv"):
            print("The output file already exists, please set update to True if you want to update it")
        else:
            # assign gsv_metadata to gsv_metadata
            gsv_metadata = pd.read_csv(os.path.join(self.gsv_metadata_output_folder, "gsv_metadata.csv"))
            # define a function that takes two sets of lat and lon and return distance
            def calc_dist_row(row):
                dist = distance.distance((row["lat"],row["lon"]), (row["input_lat"],row["input_lon"])).meters
                return dist
            
            gsv_metadata["distance"] = gsv_metadata.apply(lambda row: calc_dist_row(row), axis=1)
            
            # save df_output
            gsv_metadata.to_csv(os.path.join(self.gsv_metadata_output_folder, "gsv_metadata_dist.csv"), index = False)
            
    def download_gsv(self):
        """function to download gsv svi with multiple threads
        """
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
                if not os.path.exists(path_output_raw):
                    cv2.imwrite(path_output_raw, img_raw)
        
        # set and create directories       
        dir_input_raw = os.path.join(self.gsv_image_output_folder,"panorama")
        dir_out_show = os.path.join(self.gsv_image_output_folder,"perspective")
        os.makedirs(dir_out_show, exist_ok = True)

        # set parameters
        index = 0
        show_size = 100  # 像素大小 * 4 或者 3
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