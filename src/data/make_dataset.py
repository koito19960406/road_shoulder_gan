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
def main(output_folder, MLY_ACCESS_TOKEN,ORGANIZATION_ID):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    # create Downloader instance to download SVI from mapillary and google street view
    downloader = Downloader(MLY_ACCESS_TOKEN)
    # set bbox for the entire singapore
    bbox = {'south': 1.1304753,
            'north': 1.4504753,
            'west': 103.6920359,
            'east': 104.0120359}
    # get json response
    response = downloader.get_mly_image_id(bbox, ORGANIZATION_ID)
    # response = downloader.get_mly_image_id_fake()
    response_df = gpd.read_file(response)
    
    # save and read response_df
    mly_output_folder = os.path.join(output_folder,"mapillary")
    mly_metadata_output_folder = os.path.join(mly_output_folder,"metadata")
    os.makedirs(mly_metadata_output_folder,exist_ok = True)
    response_df.to_file(os.path.join(mly_metadata_output_folder, "response_df.geojson"), driver='GeoJSON')
    response_df = gpd.read_file(os.path.join(mly_metadata_output_folder, "response_df.geojson"))
    
    # download mapillary images
    id_list = response_df["id"].tolist()
    url_list = downloader.get_mly_url(id_list)
    mly_image_output_folder = os.path.join(mly_output_folder,"image")
    downloader.download_mly_image(id_list,url_list,mly_image_output_folder)
    
    # download google street view
    panoids_df_agg = downloader.get_gsv_panoid(response_df)
    # create output folder for GSV
    gsv_output_folder = os.path.join(output_folder,"gsv")
    gsv_metadata_output_folder = os.path.join(gsv_output_folder,"metadata")
    panoids_df_agg.to_csv(os.path.join(gsv_metadata_output_folder,"panoids_df_agg.csv"))
    panoids_df_agg = pd.read_csv(os.path.join(gsv_metadata_output_folder,"panoids_df_agg.csv"))

    # calculate the distance
    panoids_df_agg_dist = downloader.calculate_gsv_dist(panoids_df_agg)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    MLY_ACCESS_TOKEN = os.getenv('MLY_ACCESS_TOKEN')
    # ORGANIZATION_ID = os.getenv('ORGANIZATION_ID')
    ORGANIZATION_ID = [253500583201167]
    output_folder = "./data/raw/"
    main(output_folder,MLY_ACCESS_TOKEN,ORGANIZATION_ID)
