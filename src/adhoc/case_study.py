import os
import dotenv
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import contextily as ctx
import glob
import tqdm

from src.data.retrieve_svi import Downloader
from src.features.build_features import FilterImage, FormatFolder
from src.models.segmentation import segmentation
from src.models.predict_model import Predictor

class CaseStudy:
    """class for conducting a case study. As this class requires trained GAN models, this should be used after training a GAN model. 
    """
    def __init__(self, root_dir, case_study_name, input_gdf, names = ["cyclegan_filtered", "pix2pix_filtered"]):
        self.input_gdf = input_gdf
        self.names = names
        self.root_dir = root_dir
        self.case_study_dir = os.path.join(root_dir,"case_study",case_study_name)
        self.case_study_dir_raw = os.path.join(root_dir,"case_study",case_study_name, "data/raw")
        self.case_study_dir_interim = os.path.join(root_dir,"case_study",case_study_name, "data/interim")
        self.case_study_dir_processed = os.path.join(root_dir,"case_study",case_study_name, "data/processed")
        os.makedirs(self.case_study_dir_raw, exist_ok = True)
        os.makedirs(self.case_study_dir_interim, exist_ok = True)
        os.makedirs(self.case_study_dir_processed, exist_ok = True)
    
    def convert_to_bbox(self):
        input_gdf_dissolved = self.input_gdf.unary_union
        return {
                'west': input_gdf_dissolved.bounds[0],
                'south': input_gdf_dissolved.bounds[1],
                'east': input_gdf_dissolved.bounds[2],
                'north': input_gdf_dissolved.bounds[3]
                }
        
    def download_images(self, MLY_ACCESS_TOKEN, ORGANIZATION_ID):
        # # convert to polygon
        # self.convert_to_poly()
        # init downloader for images
        downloader = Downloader(self.case_study_dir_raw, MLY_ACCESS_TOKEN)
        bbox = self.convert_to_bbox()
        downloader.get_mly_image_id(bbox, ORGANIZATION_ID)
        downloader.get_mly_url()
        downloader.download_mly_image()
        downloader.get_gsv_metadata_multiprocessing()
        downloader.calc_dist()
        downloader.download_gsv()
        downloader.transform_pano_to_perspective()
    
    def prepare_images(self, pretrained_model_folder):
        gsv_folder = os.path.join(self.case_study_dir_raw,"gsv")
        mly_folder = os.path.join(self.case_study_dir_raw,"mapillary")
        filter_image = FilterImage(pretrained_model_folder, gsv_folder, mly_folder)
        filter_image.run_all()
        for name in self.names:
            gan_folder = os.path.join(self.case_study_dir_processed,name)
            format_folder = FormatFolder(gsv_folder, mly_folder, gan_folder)
            format_folder.create_new_folder(model = name, test_size = 1)
            if "pix2pix" in name:
                format_folder.prepare_pix2pix(name)
    
    def predict(self):
        for name in self.names:
            if "cyclegan" in name:
                num_test = len(glob.glob(os.path.join(self.case_study_dir_processed,name, name, "testA/*.jpg")))
                model = "cycle_gan"
            elif "pix2pix" in name:
                num_test = len(glob.glob(os.path.join(self.case_study_dir_processed,name, name, "test/*.jpg")))
                model = "pix2pix"
            else:
                num_test = 50  
            predictor = Predictor(os.path.join(self.case_study_dir_processed,name,name), 
                            os.path.join(self.root_dir,"models"), 
                            os.path.join(self.case_study_dir_processed, f"{name}/results"),
                            name, model, num_test)
            predictor.predict()
        
    def segment(self):
        for name in self.names:
            result_folder = os.path.join(self.case_study_dir_processed, f"{name}/results/{name}/test_latest/images")
            new_folder = os.path.join(self.case_study_dir_processed, f"{name}/gan_results")
            if not os.path.exists(new_folder):
                segmentation.move_results(result_folder, new_folder)
            # segment input (mly and gsv) and output (fake) 
            if "pix2pix" in name:
                mly_input_folder = os.path.join(self.case_study_dir_processed,f"{name}/{name}_init/B/test") 
                gsv_input_folder = os.path.join(self.case_study_dir_processed,f"{name}/{name}_init/A/test") 
            if "cyclegan" in name:
                mly_input_folder = os.path.join(self.case_study_dir_processed,f"{name}/{name}/testB") 
                gsv_input_folder = os.path.join(self.case_study_dir_processed,f"{name}/{name}/testA") 
            for input_folder in [new_folder, mly_input_folder, gsv_input_folder]:
                # we need to use this gimmick here to set img_type
                #TODO create a separate test dataset with standardized folder names
                basename = os.path.basename(input_folder)
                if "pix2pix" in name and basename!="gan_results":
                    img_type = "test" + os.path.split(input_folder)[0][-1]
                else:
                    img_type = os.path.basename(input_folder)
                img_output_folder = os.path.join(self.case_study_dir_interim, "segmented",name,img_type)
                csv_output_folder = os.path.join(self.case_study_dir_processed, name,"segmentation_result",img_type)
                img_seg = segmentation.ImageSegmentationSimple(input_folder, img_output_folder, csv_output_folder)
                img_seg.segment_svi(batch_size_store=100)
                img_seg.calculate_ratio()
            gsv_result_csv = os.path.join(self.case_study_dir_processed,name,"segmentation_result/testA/segmentation_pixel_ratio_wide.csv")
            mly_result_csv = os.path.join(self.case_study_dir_processed,name,"segmentation_result/testB/segmentation_pixel_ratio_wide.csv")
            gan_result_csv = os.path.join(self.case_study_dir_processed,name,"segmentation_result/gan_results/segmentation_pixel_ratio_wide.csv")
            output_folder = os.path.join(self.case_study_dir_processed,name,"segmentation_result")
            correlation_analysis = segmentation.CorrelationAnalysis(gsv_result_csv, mly_result_csv, gan_result_csv, output_folder)
            correlation_analysis.merge_save()

    def spatial_join(self):
        def join_line_point(line,point):
            return gpd.sjoin_nearest(line,point,how="left")
        # load mly points
        response_gdf = gpd.read_file(os.path.join(self.case_study_dir_raw,"mapillary/metadata/response.geojson"))
        for name in self.names:
            # load segmentation results
            segmentation_result = pd.read_csv(os.path.join(self.case_study_dir_processed,f"{name}/segmentation_result/segmentation_result.csv"))
            # get geometry and convert to gdf
            segmentation_result = pd.merge(segmentation_result,response_gdf,left_on="file_name", right_on="id",how="left")
            segmentation_result_gdf = gpd.GeoDataFrame(segmentation_result,geometry='geometry')
            # filter out those outside of 10m buffer
            buffer_filter=gpd.GeoDataFrame(geometry=self.input_gdf.to_crs(3857).buffer(10)).to_crs(4326)
            buffer_filter['filter_flag']=1
            segmentation_result_gdf=gpd.sjoin(segmentation_result_gdf,buffer_filter,how="left")
            segmentation_result_gdf=segmentation_result_gdf[segmentation_result_gdf['filter_flag']==1]
            segmentation_result_gdf=segmentation_result_gdf.drop(["index_right"],axis=1)
            # nearest join
            line_joined = join_line_point(self.input_gdf,segmentation_result_gdf)
            # group by line segment
            line_joined = line_joined.dissolve(by='osm_id', aggfunc='mean')
            # save
            line_joined.to_file(os.path.join(self.case_study_dir_processed,name,"segmentation_result/line_joined.geojson"), driver='GeoJSON')
    
    def plot(self, output_folder):
        for name in self.names:
            line_joined = gpd.read_file(os.path.join(self.case_study_dir_processed,name,"segmentation_result/line_joined.geojson"))
            categories = [col.replace("gsv_", "") for col in line_joined.columns if col.startswith("gsv_")]
            for category in tqdm.tqdm(categories):
                # calculate the difference
                line_joined[f"diff_{category}"] = abs(line_joined[f"mly_{category}"] - line_joined[f"gan_{category}"])
                # visualize
                fig, ax = plt.subplots(1, figsize=(10,10))
                line_joined.plot(column=f"diff_{category}",ax=ax,
                                cmap='plasma',
                                alpha=1,
                                linewidth=3,
                                legend=True
                                )
                ax.set_axis_off()
                # add title
                title=f"{category}"
                ax.set_title(title, fontdict={"fontsize": "25", "fontweight" : "3"})
                # add scalebar
                scalebar = ScaleBar(dx=1) 
                plt.gca().add_artist(scalebar)
                
                ctx.add_basemap(ax,source=ctx.providers.CartoDB.PositronNoLabels,crs=line_joined.crs.to_string())
                fig.savefig(os.path.join(output_folder,name,f'{category}_difference.png'),
                            bbox_inches='tight',
                            dpi=400)

if __name__ == '__main__':
    dotenv.load_dotenv(dotenv.find_dotenv())
    root_dir = os.getenv('ROOT_DIR')
    if not os.path.exists(root_dir):
        root_dir = os.getenv('ROOT_DIR_2')
    case_study_name = "pasir_panjang_road"
    MLY_ACCESS_TOKEN = os.getenv('MLY_ACCESS_TOKEN')
    ORGANIZATION_ID = [int(os.getenv('ORGANIZATION_ID'))]
    test = gpd.read_file(os.path.join(root_dir, "data/external/pasir_panjang_rd_filtered.geojson")).set_crs(epsg=4326)
    case_study = CaseStudy(root_dir, case_study_name, test, names = ["pix2pix_filtered"])
    # case_study.download_images(MLY_ACCESS_TOKEN, ORGANIZATION_ID)
    # pretrained_model_folder = os.path.join(root_dir,"data/external")
    # case_study.prepare_images(pretrained_model_folder)
    # case_study.predict()
    # case_study.segment()
    case_study.spatial_join()
    # case_study.plot("reports/figures")