import os
import dotenv
import pandas as pd
import geopandas as gpd

# import matplotlib.pyplot as plt
# from matplotlib_scalebar.scalebar import ScaleBar
# import contextily as ctx
import glob
import tqdm
from zensvi.cv import Segmenter
from zensvi.download import MLYDownloader, GSVDownloader
import shutil

from src.models.utils import move_results, CorrelationAnalysis
from src.data.retrieve_svi import Downloader
from src.features.build_features import FilterImage, FormatFolder

# from src.models.segmentation import segmentation
from src.models.predict_model import Predictor, CustomPredictor


class CaseStudy:
    """class for conducting a case study. As this class requires trained GAN models, this should be used after training a GAN model."""

    def __init__(
        self,
        root_dir,
        case_study_name,
        input_gdf,
        names=["cyclegan_filtered", "pix2pix_filtered"],
    ):
        self.input_gdf = input_gdf
        self.names = names
        self.root_dir = root_dir
        self.case_study_dir = os.path.join(root_dir, "case_study", case_study_name)
        self.case_study_dir_raw = os.path.join(
            root_dir, "case_study", case_study_name, "data/raw"
        )
        self.case_study_dir_interim = os.path.join(
            root_dir, "case_study", case_study_name, "data/interim"
        )
        self.case_study_dir_processed = os.path.join(
            root_dir, "case_study", case_study_name, "data/processed"
        )
        self.case_study_dir_processed_relative = os.path.join(
            "case_study", case_study_name, "data/processed"
        )
        self.data_dir_processed = os.path.join(root_dir, "data/processed")
        os.makedirs(self.case_study_dir_raw, exist_ok=True)
        os.makedirs(self.case_study_dir_interim, exist_ok=True)
        os.makedirs(self.case_study_dir_processed, exist_ok=True)

    def convert_to_bbox(self):
        input_gdf_dissolved = self.input_gdf.unary_union
        return {
            "west": input_gdf_dissolved.bounds[0],
            "south": input_gdf_dissolved.bounds[1],
            "east": input_gdf_dissolved.bounds[2],
            "north": input_gdf_dissolved.bounds[3],
        }

    def download_images(self, MLY_ACCESS_TOKEN, org_id_dict):
        # # convert to polygon
        # self.convert_to_poly()
        # init downloader for images
        for key, value in org_id_dict.items():
            output_folder = os.path.join(self.case_study_dir_raw, key)
            downloader = Downloader(output_folder, MLY_ACCESS_TOKEN, value)
            downloader.download_mly(self.input_gdf)
            downloader.download_gsv(self.input_gdf)
            downloader.calc_dist()
            downloader.transform_pano_to_perspective()

    def prepare_images(self, pretrained_model_folder):
        for name in self.names:
            if "perspective" in name:
                perspective = "_perspective"
            else:
                perspective = ""
            if "cyclegan" in name:
                model_name = "cyclegan" + perspective
            if "pix2pix" in name:
                model_name = "pix2pix" + perspective
            # get platform from name
            if "road_shoulder" in name:
                platform = "road_shoulder"
            if "sidewalk" in name:
                platform = "sidewalk"

            # filter data
            gsv_folder = os.path.join(self.case_study_dir_raw, platform, "gsv")
            mly_folder = os.path.join(self.case_study_dir_raw, platform, "mapillary")
            filter_image = FilterImage(pretrained_model_folder, gsv_folder, mly_folder)
            filter_image.run_all()

            # move data to clean folders
            gan_folder = os.path.join(self.case_study_dir_processed, "images", platform)
            format_folder = FormatFolder(gsv_folder, mly_folder, gan_folder)
            if "perspective" in name:
                format_folder.create_new_folder(
                    model=model_name, test_size=1, panorama=False
                )
            else:
                format_folder.create_new_folder(model=model_name, test_size=1)
            if "pix2pix" in name:
                format_folder.prepare_pix2pix(model_name)

    def predict(self):
        for name in self.names:
            if "perspective" in name:
                perspective = "_perspective"
            else:
                perspective = ""
            # get platform from name
            if "road_shoulder" in name:
                platform = "road_shoulder"
            if "sidewalk" in name:
                platform = "sidewalk"
            if "cyclegan" in name:
                model = "cycle_gan"
                model_name = "cyclegan" + perspective
                num_test = len(
                    glob.glob(
                        os.path.join(
                            self.case_study_dir_processed,
                            "images",
                            platform,
                            model_name,
                            "testA/*.jpg",
                        )
                    )
                )
            elif "pix2pix" in name:
                model = "pix2pix"
                model_name = "pix2pix" + perspective
                num_test = len(
                    glob.glob(
                        os.path.join(
                            self.case_study_dir_processed,
                            "images",
                            platform,
                            model_name,
                            "test/*.jpg",
                        )
                    )
                )
            else:
                num_test = 50
            predictor = Predictor(
                os.path.join(
                    self.case_study_dir_processed, "images", platform, model_name
                ),
                os.path.join(self.root_dir, "models"),
                os.path.join(self.case_study_dir_processed),
                name,
                model,
                num_test,
            )
            predictor.predict()
        # delete the input images
        shutil.rmtree(
            os.path.join(self.case_study_dir_processed, "images", platform, model_name)
        )

    def segment(self):
        for name in self.names:
            result_folder = os.path.join(
                self.case_study_dir_processed, f"{name}/test_latest/images"
            )
            new_folder = os.path.join(
                self.case_study_dir_processed, f"{name}/gan_results"
            )
            if not os.path.exists(new_folder):
                move_results(result_folder, new_folder)
            # get platform from name
            if "road_shoulder" in name:
                platform = "road_shoulder"
            if "sidewalk" in name:
                platform = "sidewalk"
            # segment input (mly and gsv) and output (fake)
            if "pix2pix" in name:
                if "perspective" in name:
                    mly_input_folder = os.path.join(
                        self.case_study_dir_processed,
                        "images",
                        platform,
                        "pix2pix_perspective_init/B/test",
                    )
                    gsv_input_folder = os.path.join(
                        self.case_study_dir_processed,
                        "images",
                        platform,
                        "pix2pix_perspective_init/A/test",
                    )
                else:
                    mly_input_folder = os.path.join(
                        self.case_study_dir_processed,
                        "images",
                        platform,
                        "pix2pix_init/B/test",
                    )
                    gsv_input_folder = os.path.join(
                        self.case_study_dir_processed,
                        "images",
                        platform,
                        "pix2pix_init/A/test",
                    )
            if "cyclegan" in name:
                if "perspective" in name:
                    mly_input_folder = os.path.join(
                        self.case_study_dir_processed,
                        "images",
                        platform,
                        "cyclegan_perspective/testB",
                    )
                    gsv_input_folder = os.path.join(
                        self.case_study_dir_processed,
                        "images",
                        platform,
                        "cyclegan_perspective/testA",
                    )
                else:
                    mly_input_folder = os.path.join(
                        self.case_study_dir_processed,
                        "images",
                        platform,
                        "cyclegan/testB",
                    )
                    gsv_input_folder = os.path.join(
                        self.case_study_dir_processed,
                        "images",
                        platform,
                        "cyclegan/testA",
                    )
            for input_folder in [new_folder, mly_input_folder, gsv_input_folder]:
                # we need to use this gimmick here to set img_type
                # TODO create a separate test dataset with standardized folder names
                basename = os.path.basename(input_folder)
                if "pix2pix" in name and basename != "gan_results":
                    img_type = "test" + os.path.split(input_folder)[0][-1]
                else:
                    img_type = os.path.basename(input_folder)
                img_output_folder = os.path.join(
                    self.case_study_dir_processed, name, "segmented", img_type
                )
                csv_output_folder = os.path.join(
                    self.case_study_dir_processed, name, "segmentation_result", img_type
                )
                # initialize the segmenter
                segmenter = Segmenter()
                pixel_ratio_save_format = ["csv"]
                segmenter.segment(
                    input_folder,
                    # dir_image_output = img_output_folder,
                    dir_segmentation_summary_output=csv_output_folder,
                    pixel_ratio_save_format=pixel_ratio_save_format,
                    csv_format="wide",
                )
                # img_seg = segmentation.ImageSegmentationSimple(input_folder, img_output_folder, csv_output_folder)
                # img_seg.segment_svi(batch_size_store=100)
                # img_seg.calculate_ratio()
            gsv_result_csv = os.path.join(
                self.case_study_dir_processed,
                name,
                "segmentation_result/testA/pixel_ratios.csv",
            )
            mly_result_csv = os.path.join(
                self.case_study_dir_processed,
                name,
                "segmentation_result/testB/pixel_ratios.csv",
            )
            gan_result_csv = os.path.join(
                self.case_study_dir_processed,
                name,
                "segmentation_result/gan_results/pixel_ratios.csv",
            )
            output_folder = os.path.join(
                self.case_study_dir_processed, name, "segmentation_result"
            )
            correlation_analysis = CorrelationAnalysis(
                gsv_result_csv, mly_result_csv, gan_result_csv, output_folder
            )
            correlation_analysis.merge_save()

    def custom_predict(
        self, target_variables=["building_mly", "sky_mly", "vegetation_mly"]
    ):
        self.custom_model_dict = {}
        for key in ["road_shoulder", "sidewalk"]:
            view = "perspective" if key == "road_shoulder" else "panorama"
            name = (
                "road_shoulder_cyclegan_perspective"
                if key == "road_shoulder"
                else "sidewalk_cyclegan_default"
            )
            self.custom_model_dict[key] = [
                {
                    "target_variable": target_variable,
                    "data_root": os.path.join(self.data_dir_processed, f"{key}/{view}"),
                    "csv_path": os.path.join(
                        self.case_study_dir_processed,
                        f"{name}/segmentation_result/segmentation_result.csv",
                    ),
                    "checkpoint": os.path.join(
                        self.data_dir_processed,
                        f"{key}/{view}/{target_variable}_with_gan_lightgbm_model.pkl",
                    ),
                    "results_dir": os.path.join(
                        self.case_study_dir_processed, f"{key}/{view}"
                    ),
                    "model_name_with_gan": "with_gan_lightgbm",
                }
                for target_variable in target_variables
            ]

        for key, value in self.custom_model_dict.items():
            for model in value:
                # make results dir
                os.makedirs(model["results_dir"], exist_ok=True)
                custom_predictor = CustomPredictor(
                    model["data_root"],
                    model["csv_path"],
                    model["checkpoint"],
                    model["results_dir"],
                    model["model_name_with_gan"],
                )
                custom_predictor.predict(model["target_variable"])

    def map_improvement(self):
        # read the input geojson
        input_gdf = gpd.read_file(self.input_gdf).set_crs(epsg=4326).to_crs(3857)
        # loop through the results of custom_predict
        for key, value in self.custom_model_dict.items():
            for model in value:
                # load mly points
                df = pd.read_csv(
                    os.path.join(self.case_study_dir_raw, key, "mapillary/mly_pids.csv")
                )
                response_gdf = gpd.GeoDataFrame(
                    df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326"
                ).to_crs(
                    3857
                )  # This sets the coordinate reference system to WGS 84
                # convrt mly_id to str
                response_gdf["mly_id"] = (
                    response_gdf.dropna(subset=["mly_id"])["mly_id"]
                    .astype("int64")
                    .astype(str)
                )
                # load results
                results_df = pd.read_csv(
                    os.path.join(
                        model["results_dir"],
                        f"{model['target_variable']}_{model['model_name_with_gan']}_predictions.csv",
                    )
                )
                # convert filename_key to str
                results_df["filename_key"] = (
                    results_df.dropna(subset=["filename_key"])["filename_key"]
                    .astype("int64")
                    .astype(str)
                )
                # gan model name
                gan_model_name = (
                    "road_shoulder_cyclegan_perspective"
                    if key == "road_shoulder"
                    else "sidewalk_cyclegan_default"
                )
                results_df["file_path_mly"] = results_df["filename_key"].map(
                    lambda x: os.path.join(                    
                        self.case_study_dir_processed_relative,
                        gan_model_name,
                        "test_latest/images",
                        str(x) + "_real_B.png",
                    ))
                results_df["file_path_gan"] = results_df["filename_key"].map(
                    lambda x: os.path.join(
                        self.case_study_dir_processed_relative,
                        gan_model_name,
                        "test_latest/images",
                        str(x) + "_fake_B.png",
                    )
                )
                results_df["file_path_gsv"] = results_df["filename_key"].map(
                    lambda x: os.path.join(
                        self.case_study_dir_processed_relative,
                        gan_model_name,
                        "test_latest/images",
                        str(x) + "_real_A.png",
                    )
                )
                # calculate the bias: gsv - mly
                results_df["bias"] = results_df[model["target_variable"].replace("mly", "gsv")] - results_df[model["target_variable"]]
                results_df["change"] = results_df[f"predicted_{model['target_variable']}"] - results_df[model["target_variable"]]
                results_df["improvement"] = abs(results_df["bias"]) - abs(results_df["change"])
                # left join the results with the mly points
                results_gdf = pd.merge(
                    response_gdf,
                    results_df,
                    left_on="mly_id",
                    right_on="filename_key",
                    how="left",
                )
                # spatially nearest join with the input geojson with 10m buffer
                results_gdf_with_osm_id = gpd.sjoin_nearest(
                    results_gdf, input_gdf, how="left", max_distance=50
                )
                # agg by osm_id and calculate the mean
                results_gdf_with_osm_id = results_gdf_with_osm_id.groupby(
                    "osm_id", as_index=False).mean().dropna(subset=["osm_id"]).reset_index(drop=True)
                # left join the results with the input geojson
                line_joined = pd.merge(
                    input_gdf,
                    results_gdf_with_osm_id,
                    left_on="osm_id",
                    right_on="osm_id",
                    how="left",
                )
                # save line_joined
                line_joined.to_file(
                    os.path.join(
                        model["results_dir"],
                        f"{model['target_variable']}_line_joined.geojson",
                    ),
                    driver="GeoJSON",
                )
                # save results_gdf
                results_gdf.to_file(
                    os.path.join(
                        model["results_dir"],
                        f"{model['target_variable']}_point_results.geojson",
                    ),
                    driver="GeoJSON",
                )

    def spatial_join(self):
        def join_line_point(line, point):
            return gpd.sjoin_nearest(line, point, how="left")

        # load mly points
        df = pd.read_csv(
            os.path.join(self.case_study_dir_raw, "mapillary/mly_pids.csv")
        )
        response_gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326"
        )  # This sets the coordinate reference system to WGS 84
        for name in self.names:
            # load segmentation results
            segmentation_result = pd.read_csv(
                os.path.join(
                    self.case_study_dir_processed,
                    f"{name}/segmentation_result/segmentation_result.csv",
                )
            )
            # get geometry and convert to gdf
            segmentation_result = pd.merge(
                segmentation_result,
                response_gdf,
                left_on="filename_key",
                right_on="id",
                how="left",
            )
            segmentation_result_gdf = gpd.GeoDataFrame(
                segmentation_result, geometry="geometry"
            )
            # filter out those outside of 10m buffer
            buffer_filter = (
                gpd.read_file(self.input_gdf)
                .set_crs(epsg=4326)
                .to_crs(3857)
                .buffer(10)
                .to_crs(4326)
            )
            buffer_filter["filter_flag"] = 1
            segmentation_result_gdf = gpd.sjoin(
                segmentation_result_gdf, buffer_filter, how="left"
            )
            segmentation_result_gdf = segmentation_result_gdf[
                segmentation_result_gdf["filter_flag"] == 1
            ]
            segmentation_result_gdf = segmentation_result_gdf.drop(
                ["index_right"], axis=1
            )
            # nearest join
            line_joined = join_line_point(self.input_gdf, segmentation_result_gdf)
            # group by line segment
            line_joined = line_joined.dissolve(by="osm_id", aggfunc="mean")
            # save
            line_joined.to_file(
                os.path.join(
                    self.case_study_dir_processed,
                    name,
                    "segmentation_result/line_joined.geojson",
                ),
                driver="GeoJSON",
            )

    def plot(self, output_folder):
        for name in self.names:
            line_joined = gpd.read_file(
                os.path.join(
                    self.case_study_dir_processed,
                    name,
                    "segmentation_result/line_joined.geojson",
                )
            )
            categories = [
                col.replace("gsv_", "")
                for col in line_joined.columns
                if col.startswith("gsv_")
            ]
            for category in tqdm.tqdm(categories):
                # calculate the difference
                line_joined[f"diff_{category}"] = abs(
                    line_joined[f"mly_{category}"] - line_joined[f"gan_{category}"]
                )
                # visualize
                fig, ax = plt.subplots(1, figsize=(10, 10))
                line_joined.plot(
                    column=f"diff_{category}",
                    ax=ax,
                    cmap="plasma",
                    alpha=1,
                    linewidth=3,
                    legend=True,
                )
                ax.set_axis_off()
                # add title
                title = f"{category}"
                ax.set_title(title, fontdict={"fontsize": "25", "fontweight": "3"})
                # add scalebar
                scalebar = ScaleBar(dx=1)
                plt.gca().add_artist(scalebar)

                ctx.add_basemap(
                    ax,
                    source=ctx.providers.CartoDB.PositronNoLabels,
                    crs=line_joined.crs.to_string(),
                )
                fig.savefig(
                    os.path.join(output_folder, name, f"{category}_difference.png"),
                    bbox_inches="tight",
                    dpi=400,
                )


if __name__ == "__main__":
    dotenv.load_dotenv(dotenv.find_dotenv())
    root_dir = os.getenv("ROOT_DIR")
    if not os.path.exists(root_dir):
        # list of drives from D to Z
        drives = [f"{chr(x)}:/" for x in range(ord("D"), ord("Z") + 1)] + [
            "/Volumes/ExFAT/"
        ]
        for drive in drives:
            # check if the root_dir exists
            if os.path.exists(os.path.join(drive, "road_shoulder_gan")):
                root_dir = os.path.join(drive, "road_shoulder_gan")
                break
    case_study_name = "pasir_panjang_road"
    MLY_ACCESS_TOKEN = os.getenv("MLY_ACCESS_TOKEN")
    ORGANIZATION_ID = [int(os.getenv("ORGANIZATION_ID"))]
    ORGANIZATION_ID_2 = [int(os.getenv("ORGANIZATION_ID_2"))]
    org_id_dict = {"road_shoulder": ORGANIZATION_ID, "sidewalk": ORGANIZATION_ID_2}
    input_geojson = os.path.join(
        root_dir, "data/external/pasir_panjang_rd_filtered.geojson"
    )
    # test = gpd.read_file(input_geojson).set_crs(epsg=4326)
    # get names list by getting a list of files in configs/done and remove train_ and .txt
    # names = [os.path.basename(file).replace("train_","").replace(".txt","") for file in glob.glob(os.path.join(root_dir,"configs/done/*.txt"))]
    names = ["road_shoulder_cyclegan_perspective", "sidewalk_cyclegan_default"]
    case_study = CaseStudy(root_dir, case_study_name, input_geojson, names=names)
    # case_study.download_images(MLY_ACCESS_TOKEN, org_id_dict)
    # pretrained_model_folder = os.path.join(root_dir,"data/external")
    # case_study.prepare_images(pretrained_model_folder)
    # case_study.predict()
    # case_study.segment()
    case_study.custom_predict()
    case_study.map_improvement()
    # case_study.spatial_join()
    # case_study.plot("reports/figures")
