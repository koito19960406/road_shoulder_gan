import mapillary as mly
import requests
import os
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from multiprocessing import Lock
import streetview
import pandas as pd

class Downloader:
    """class for downloading images from mapillary
    """
    def __init__(self,MLY_ACCESS_TOKEN) -> None:
        mly.interface.set_access_token(MLY_ACCESS_TOKEN)
        self._cpus = cpu_count()
        pass
    
    def get_mly_image_id(self, bbox, ORGANIZATION_ID):
        """function to retrieve image id from bbox and organization id

        Args:
            bbox (dict): bbox
            ORGANIZATION_ID (str): organization id

        Returns:
            response: geojson containing image info
        """
        response = mly.interface.images_in_bbox(bbox, organization_id = ORGANIZATION_ID)
        return response
    
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
    
    def get_mly_url(self, id_list):
        url_list = []
        for id in id_list:
            url = mly.interface.image_thumbnail(image_id=id, resolution=2048)
            url_list.append(url)
        return url_list
    
    def download_mly_image(self, id_list, url_list, output_folder):
        # create output folder if not existing
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
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
        file_name_list = [os.path.join(output_folder,str(id)+".jpg") for id in id_list]
        args = zip(url_list, file_name_list)
        results = ThreadPool(self._cpus - 1).imap_unordered(download_files, args)
        for result in results:
            print('url:', result[0], 'time (s):', result[1])   
        
    def get_gsv_panoid(self, responses_df):
        panoids_df_agg = None
        for index, row in responses_df.iterrow():
            panoids = streetview.panoids(lat=row.geometry.y, lon=row.geometry.x)
            panoids_df_temp = pd.DataFrame.from_dict(panoids)
            panoids_df_temp["original_lat"]=row.geometry.y
            panoids_df_temp["original_lon"]=row.geometry.x
            if panoids_df_agg==None:
                panoids_df_agg=panoids_df_temp
            else:
                panoids_df_agg = panoids_df_agg.append(panoids_df_temp)
        return panoids_df_agg
    
    def calculate_gsv_dist(self,panoids_df_agg):
        # unify the crs to 3857
        distance = count_station_gdf['geometry'].distance(row.geometry)
        distance = distance.tolist()[0]
        return distance
        return panoids_df_agg_dist
        pass