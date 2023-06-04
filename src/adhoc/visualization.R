pacman::p_load(tidyverse,sf,basemaps,dotenv,progress,magrittr,Redmonder,
               hrbrthemes,ggspatial,lwgeom,ggimage,cropcircles,ggrepel,ggridges)


# load dataset ------------------------------------------------------------
load_dot_env()
root_dir <- Sys.getenv("ROOT_DIR")
line_joined <- st_read(paste0(root_dir,"/case_study/pasir_panjang_road/data/processed/pix2pix_filtered/segmentation_result/line_joined.geojson")) %>% 
  st_transform(3857)
buffer_filter <- st_buffer(line_joined,10) %>% 
  select(geometry)
image_points <- st_read(paste0(root_dir,"/case_study/pasir_panjang_road/data/raw/mapillary/metadata/response.geojson")) %>% 
  st_transform(3857)
segmentation_result <- read_csv(paste0(root_dir,"/case_study/pasir_panjang_road/data/processed/pix2pix_filtered/segmentation_result/segmentation_result.csv")) %>%
  mutate(file_path_mly=paste0(root_dir,"/case_study/pasir_panjang_road/data/raw/mapillary/image/", as.character(file_name),".jpg"),
         file_path_gan=paste0(root_dir,"/case_study/pasir_panjang_road/data/processed/pix2pix_filtered/results/pix2pix_filtered/test_latest/images/", as.character(file_name),"_fake_B.png")) %>% 
  left_join(.,image_points,by=c("file_name"="id")) %>% 
  st_as_sf() %>% 
  st_filter(.,buffer_filter) 
names(segmentation_result) %<>% str_replace(" ",".")
# calculate the differences between mly and gan by category
categories <- names(segmentation_result)[str_detect(names(segmentation_result),"gsv")] %>% 
  str_remove_all("gsv_")
pb <- progress_bar$new(total=length(categories))
for (category in categories){
  pb$tick()
  line_joined %<>% mutate("diff_{category}":= abs(.data[[paste0("mly_",{{category}})]] - .data[[paste0("gan_",{{category}})]]))
  segmentation_result %<>% mutate("diff_{category}":= abs(.data[[paste0("mly_",{{category}})]] - .data[[paste0("gan_",{{category}})]]))
}

# define functions for plot ------------------------------------------------
map_diff <- function(line_joined,segmentation_result, category){
  # max difference
  max_diff <- segmentation_result %>% 
    slice_max(.,order_by=.data[[paste0("diff_",{category})]], n=1) 
  min_diff <- segmentation_result %>% 
    slice_min(.,order_by=.data[[paste0("diff_",{category})]], n=1)
  # decide the xend and yend
  distance_for_image <- 213
  xend_1<-11552600
  yend_1<-143750
  xend_2<-11552100
  yend_2<-143150
  if(st_coordinates(max_diff)[1,1]>st_coordinates(min_diff)[1,1]){
    xend_min<-xend_1
    yend_min<-yend_1
    xend_max<-xend_2
    yend_max<-yend_2
  }
  else{
    xend_min<-xend_2
    yend_min<-yend_2
    xend_max<-xend_1
    yend_max<-yend_1
  }
  label_df <- tibble(
    x=c(xend_min, # the major label
        xend_max, #, # the major label
        xend_min +distance_for_image,
        xend_min -distance_for_image,
        xend_max +distance_for_image,
        xend_max -distance_for_image
        ),
    y=c(yend_min +200, # the major label
        yend_max +200, # the major label
        yend_min +150, 
        yend_min +150, 
        yend_max +150, 
        yend_max +150 
        ),
    label=c(paste0("Minimum difference: ", signif(min_diff[[paste0("diff_",category)]][1]*100, digit=2), "%"),
            paste0("Maximum difference: ", signif(max_diff[[paste0("diff_",category)]][1]*100, digit=2), "%"),
            "fake",
            "real",
            "fake",
            "real"
            )
  )
  print(label_df)
  # map the sf object
  map <- basemap_ggplot(st_bbox(line_joined), map_service="carto", map_type = "light_no_labels", alpha=0.8) +
    geom_sf(data=line_joined, mapping=aes(color=.data[[paste0("diff_",{{category}})]]*100),linewidth = 1)+
    scale_color_gradient(low="#FFFFFF",high="#7F312F", name = paste0("% difference for ", category)) +
    # add pointers
    geom_spatial_point(aes(x=st_coordinates(max_diff)[1,1],y=st_coordinates(max_diff)[1,2]),crs = 3857)+
    geom_spatial_segment(aes(x=st_coordinates(max_diff)[1,1],y=st_coordinates(max_diff)[1,2],xend=xend_max,yend=yend_max),
                         crs = 3857, lineend = "round")+
    geom_spatial_point(aes(x=st_coordinates(min_diff)[1,1],y=st_coordinates(min_diff)[1,2]),crs = 3857)+
    geom_spatial_segment(aes(x=st_coordinates(min_diff)[1,1],y=st_coordinates(min_diff)[1,2],xend=xend_min,yend=yend_min),
                         crs = 3857, lineend = "round")+
    # add images
    geom_image(aes(x=xend_max-distance_for_image, y=yend_max, image = circle_crop(max_diff[["file_path_mly"]])), size = 0.2453) +
    geom_image(aes(x=xend_max+distance_for_image, y=yend_max, image = circle_crop(max_diff[["file_path_gan"]])), size = 0.2453) +
    geom_image(aes(x=xend_min-distance_for_image, y=yend_min, image = circle_crop(min_diff[["file_path_mly"]])), size = 0.2453) +
    geom_image(aes(x=xend_min+distance_for_image, y=yend_min, image = circle_crop(min_diff[["file_path_gan"]])), size = 0.2453) +
    # geom_image(aes(x=xend_max-distance_for_image, y=yend_max, image = circle_crop("/Users/koichiito/Desktop/test.png")), size = 0.2453) +
    # geom_image(aes(x=xend_max+distance_for_image, y=yend_max, image = circle_crop("/Users/koichiito/Desktop/test.png")), size = 0.2453) +
    # geom_image(aes(x=xend_min-distance_for_image, y=yend_min, image = circle_crop("/Users/koichiito/Desktop/test.png")), size = 0.2453) +
    # geom_image(aes(x=xend_min+distance_for_image, y=yend_min, image = circle_crop("/Users/koichiito/Desktop/test.png")), size = 0.2453) +
    geom_spatial_label(mapping= aes(x,y,label=label), data = label_df, crs = 3857)+
    coord_sf(crs=3857) +
    labs(x = NULL,
         y = NULL,
         title = paste0("Difference between real and fake \n road shoulder view images for ", category),
         subtitle = NULL,
         caption = NULL)+
    theme_ipsum()+
    theme(plot.title.position="panel",
          plot.title = element_text(hjust=0.5),
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(),
          plot.margin=grid::unit(c(0,0,0,0), "mm"),
          axis.text.x=element_blank(), #remove x axis labels
          axis.ticks.x=element_blank(), #remove x axis ticks
          axis.text.y=element_blank(),  #remove y axis labels
          axis.ticks.y=element_blank(),  #remove y axis ticks
          legend.position = c(0.2, 0.6),
          legend.title = element_text(size=18),
          legend.text = element_text(size=15)
          # legend.background = element_rect(fill = alpha("white",0.2)),
          # legend.justification = c("right", "top"),
          # legend.box.just = "right",
          # legend.margin = margin(6, 6, 6, 6)
          )
  ggsave(plot=map,paste0("./reports/figures/pix2pix_filtered/",category,"_difference_map.png"), width=8, height=8)
}

# loop through categories
for (category in categories){
  map_diff(line_joined,segmentation_result, category)
}
map_diff(line_joined,segmentation_result, "building")
map_diff(line_joined,segmentation_result, "sky")
map_diff(line_joined,segmentation_result, "vegetation")


# density map -------------------------------------------------------------
plot_density <- function(segmentation_result_long){
  ggplot(segmentation_result_long, 
         aes(x = difference*100, y = category, fill = stat(x))) +
    geom_density_ridges_gradient(scale = 1.5, size = 0.3, rel_min_height = 0.001) +
    scale_fill_gradient(low="#FFFFFF",high="#7F312F",name = paste0("% difference")) +
    labs(title = 'Distribution of differences in percentage',
         x="% difference",
         y="") +
    theme_ipsum()
  ggsave(paste0("./reports/figures/pix2pix_filtered/difference_distribution.png"), width=8, height=4)
}
segmentation_result_long <- segmentation_result %>% 
  select(c(file_name,diff_building,diff_sky,diff_vegetation)) %>% 
  gather(key = "category", value = "difference",
         diff_building,diff_sky,diff_vegetation) %>% 
  mutate(category=str_remove_all(category,"diff_")) %>% 
  mutate(category=factor(category, level = c('vegetation','sky', 'building')))
plot_density(segmentation_result_long)

