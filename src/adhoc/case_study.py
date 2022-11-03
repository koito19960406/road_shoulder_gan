import os
import dotenv
import pandas as pd
import geopandas as gpd
from src.data.retrieve_svi import Downloader
from src.features.build_features import FilterImage, FormatFolder
from src.models.segmentation import segmentation
from src.models.predict_model import Predictor

class CaseStudy:
    """class for conducting a case study
    """
    def __init__(self, input_gdf, models = ["cyclegan", "pix2pix"]):
        self.input_gdf = input_gdf
        self.models = models
    
    def check_geom(self):
        """function to check if input_gdf is whether polygon, line, or point
        """
        # get numpy array of geometry
        geom_type = self.input_gdf.geom_type.unique()
        # make sure there's only 1 unique type of geometry
        assert len(geom_type) > 1, "Please make sure your input gdf has only 1 type of geometry."
        # return the geometry type
        return(geom_type[0])
    
    def convert_to_poly(self, buffer = 50):
        """function to convert input_gdf to polygon if needed
        """
        geom_type = self.check_geom()
        # if geom_type isn't polygon, then convert to 3857 and create buffer
        if geom_type != "Polygon":
            self.input_gdf = self.input_gdf.to_crs(3857).geometry.buffer(buffer)
        
    def download_images(self, input_folder, output_folder, MLY_ACCESS_TOKEN, ORGANIZATION_ID):
        # convert to polygon
        self.convert_to_poly()
        # init downloader for images
        downloader = Downloader(input_folder, output_folder, MLY_ACCESS_TOKEN)
        downloader.get_mly_image_id(self.input_gdf, ORGANIZATION_ID)
        downloader.get_mly_url()
        downloader.download_mly_image()
        downloader.get_gsv_metadata_multiprocessing()
        downloader.calc_dist()
        downloader.download_gsv()
        downloader.transform_pano_to_perspective()
    
    def prepare_images(self, pretrained_model_folder, gsv_folder, mly_folder, gan_folder):
        filter_image = FilterImage(pretrained_model_folder, gsv_folder, mly_folder)
        filter_image.run_all()
        format_folder = FormatFolder(gsv_folder, mly_folder, gan_folder)
        for model in self.models:
            format_folder.create_new_folder(model = model, test_size = 1)
            format_folder.create_new_folder(model = model, test_size = 1)
            if model == "pix2pix":
                format_folder.prepare_pix2pix(model)
    
    def predict(self, root_dir):
        for model in self.models:
            predictor = Predictor(root_dir, model, model)
            predictor.predict()
        
    def segment(self):
        result_folder = os.path.join(root_dir, f"models/{name}/{name}/test_latest/images")
        new_folder = os.path.join(root_dir, f"data/processed/{name}/results")
        if not os.path.exists(new_folder):
            segmentation.move_results(result_folder, new_folder)
        segmentation = segmentation.ImageSegmentationSimple(input_folder, img_output_folder, csv_output_folder)
        segmentation.segment_svi()
        segmentation.calculate_ratio()

if __name__ == '__main__':
    dotenv.load_dotenv(dotenv.find_dotenv())
    root_dir = os.getenv('ROOT_DIR')
    test = gpd.read_file(os.path.join(root_dir, "data/external/pasir_panjang_rd.geojson"))