# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from retrieve_svi import Downloader
import os 
import pandas as pd
import geopandas as gpd

# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def main(input_folder, output_folder, MLY_ACCESS_TOKEN,ORGANIZATION_ID):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    # create Downloader instance to download SVI from mapillary and google street view
    downloader = Downloader(input_folder, output_folder, MLY_ACCESS_TOKEN)
    # set bbox for the entire singapore
    bbox = {'south': 1.1304753,
            'north': 1.4504753,
            'west': 103.6920359,
            'east': 104.0120359}
    
    # bbox = {'south': 1.295949,
    #         'north': 1.296,
    #         'west': 103.769733,
    #         'east': 103.77}
    
    # get json response
    downloader.get_mly_image_id(bbox, ORGANIZATION_ID)

    # # get url list
    downloader.get_mly_url()
    
    # download mapillary images
    downloader.download_mly_image()
    
    # get metadata
    downloader.get_gsv_metadata_multiprocessing(update = True)

    # calculate distance
    downloader.calc_dist(update = True)
    
    # download GSV images
    downloader.download_gsv()

    # transform panorama svi to perspective svi
    downloader.transform_pano_to_perspective()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    MLY_ACCESS_TOKEN = os.getenv('MLY_ACCESS_TOKEN')
    ORGANIZATION_ID = [int(os.getenv('ORGANIZATION_ID'))]
    # ORGANIZATION_ID = [253500583201167]
    # set your own data path
    root_dir = "/Volumes/Data_Store/road_shoulder_gan"
    if os.path.exists(root_dir):
        input_folder= os.path.join(root_dir,"data/external")
        output_folder = os.path.join(root_dir,"data/raw")
        main(input_folder,output_folder,MLY_ACCESS_TOKEN,ORGANIZATION_ID)   
    else:
        input_folder = "./data/external"
        output_folder = "./data/raw"
        main(input_folder,output_folder,MLY_ACCESS_TOKEN,ORGANIZATION_ID) 