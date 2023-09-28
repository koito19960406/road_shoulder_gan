pacman::p_load(
  tidyverse, sf, basemaps, dotenv, progress, magrittr, Redmonder,
  hrbrthemes, ggspatial, lwgeom, ggimage, cropcircles, ggrepel, ggridges,
  paletteer
)


# # load dataset ------------------------------------------------------------
load_dot_env()
root_dir <- Sys.getenv("ROOT_DIR")
if (!file.exists(root_dir)) {
  # List of drives from D to Z
  drives <- c(paste0(LETTERS[4:26], ":/"), "/Volumes/ExFAT/")
  
  for (drive in drives) {
    # Check if the root_dir exists
    if (file.exists(file.path(drive, "road_shoulder_gan"))) {
      root_dir <- file.path(drive, "road_shoulder_gan")
      break
    }
  }
  root_dir <- "./"
}

# line_joined <- st_read(paste0(root_dir, "/case_study/pasir_panjang_road/data/processed/pix2pix_filtered/segmentation_result/line_joined.geojson")) %>%
#   st_transform(3857)
# buffer_filter <- st_buffer(line_joined, 10) %>%
#   select(geometry)
# image_points <- st_read(paste0(root_dir, "/case_study/pasir_panjang_road/data/raw/mapillary/metadata/response.geojson")) %>%
#   st_transform(3857)
# segmentation_result <- read_csv(paste0(root_dir, "/case_study/pasir_panjang_road/data/processed/pix2pix_filtered/segmentation_result/segmentation_result.csv")) %>%
#   mutate(
#     file_path_mly = paste0(root_dir, "/case_study/pasir_panjang_road/data/raw/mapillary/image/", as.character(file_name), ".jpg"),
#     file_path_gan = paste0(root_dir, "/case_study/pasir_panjang_road/data/processed/pix2pix_filtered/results/pix2pix_filtered/test_latest/images/", as.character(file_name), "_fake_B.png")
#   ) %>%
#   left_join(., image_points, by = c("file_name" = "id")) %>%
#   st_as_sf() %>%
#   st_filter(., buffer_filter)
# names(segmentation_result) %<>% str_replace(" ", ".")
# # calculate the differences between mly and gan by category
# categories <- names(segmentation_result)[str_detect(names(segmentation_result), "gsv")] %>%
#   str_remove_all("gsv_")
# pb <- progress_bar$new(total = length(categories))
# for (category in categories) {
#   pb$tick()
#   line_joined %<>% mutate("diff_{category}" := abs(.data[[paste0("mly_", {{ category }})]] - .data[[paste0("gan_", {{ category }})]]))
#   segmentation_result %<>% mutate("diff_{category}" := abs(.data[[paste0("mly_", {{ category }})]] - .data[[paste0("gan_", {{ category }})]]))
# }

# # # define functions for plot ------------------------------------------------
# # map_diff <- function(line_joined, segmentation_result, category) {
# #   # max difference
# #   max_diff <- segmentation_result %>%
# #     slice_max(., order_by = .data[[paste0("diff_", {
# #       category
# #     })]], n = 1)
# #   min_diff <- segmentation_result %>%
# #     slice_min(., order_by = .data[[paste0("diff_", {
# #       category
# #     })]], n = 1)
# #   # decide the xend and yend
# #   distance_for_image <- 213
# #   xend_1 <- 11552600
# #   yend_1 <- 143750
# #   xend_2 <- 11552100
# #   yend_2 <- 143150
# #   if (st_coordinates(max_diff)[1, 1] > st_coordinates(min_diff)[1, 1]) {
# #     xend_min <- xend_1
# #     yend_min <- yend_1
# #     xend_max <- xend_2
# #     yend_max <- yend_2
# #   } else {
# #     xend_min <- xend_2
# #     yend_min <- yend_2
# #     xend_max <- xend_1
# #     yend_max <- yend_1
# #   }
# #   label_df <- tibble(
# #     x = c(
# #       xend_min, # the major label
# #       xend_max, # , # the major label
# #       xend_min + distance_for_image,
# #       xend_min - distance_for_image,
# #       xend_max + distance_for_image,
# #       xend_max - distance_for_image
# #     ),
# #     y = c(
# #       yend_min + 200, # the major label
# #       yend_max + 200, # the major label
# #       yend_min + 150,
# #       yend_min + 150,
# #       yend_max + 150,
# #       yend_max + 150
# #     ),
# #     label = c(
# #       paste0("Minimum difference: ", signif(min_diff[[paste0("diff_", category)]][1] * 100, digit = 2), "%"),
# #       paste0("Maximum difference: ", signif(max_diff[[paste0("diff_", category)]][1] * 100, digit = 2), "%"),
# #       "fake",
# #       "real",
# #       "fake",
# #       "real"
# #     )
# #   )
# #   print(label_df)
# #   # map the sf object
# #   map <- basemap_ggplot(st_bbox(line_joined), map_service = "carto", map_type = "light_no_labels", alpha = 0.8) +
# #     geom_sf(data = line_joined, mapping = aes(color = .data[[paste0("diff_", {{ category }})]] * 100), linewidth = 1) +
# #     scale_color_gradient(low = "#FFFFFF", high = "#7F312F", name = paste0("% difference for ", category)) +
# #     # add pointers
# #     geom_spatial_point(aes(x = st_coordinates(max_diff)[1, 1], y = st_coordinates(max_diff)[1, 2]), crs = 3857) +
# #     geom_spatial_segment(aes(x = st_coordinates(max_diff)[1, 1], y = st_coordinates(max_diff)[1, 2], xend = xend_max, yend = yend_max),
# #       crs = 3857, lineend = "round"
# #     ) +
# #     geom_spatial_point(aes(x = st_coordinates(min_diff)[1, 1], y = st_coordinates(min_diff)[1, 2]), crs = 3857) +
# #     geom_spatial_segment(aes(x = st_coordinates(min_diff)[1, 1], y = st_coordinates(min_diff)[1, 2], xend = xend_min, yend = yend_min),
# #       crs = 3857, lineend = "round"
# #     ) +
# #     # add images
# #     geom_image(aes(x = xend_max - distance_for_image, y = yend_max, image = circle_crop(max_diff[["file_path_mly"]])), size = 0.2453) +
# #     geom_image(aes(x = xend_max + distance_for_image, y = yend_max, image = circle_crop(max_diff[["file_path_gan"]])), size = 0.2453) +
# #     geom_image(aes(x = xend_min - distance_for_image, y = yend_min, image = circle_crop(min_diff[["file_path_mly"]])), size = 0.2453) +
# #     geom_image(aes(x = xend_min + distance_for_image, y = yend_min, image = circle_crop(min_diff[["file_path_gan"]])), size = 0.2453) +
# #     # geom_image(aes(x=xend_max-distance_for_image, y=yend_max, image = circle_crop("/Users/koichiito/Desktop/test.png")), size = 0.2453) +
# #     # geom_image(aes(x=xend_max+distance_for_image, y=yend_max, image = circle_crop("/Users/koichiito/Desktop/test.png")), size = 0.2453) +
# #     # geom_image(aes(x=xend_min-distance_for_image, y=yend_min, image = circle_crop("/Users/koichiito/Desktop/test.png")), size = 0.2453) +
# #     # geom_image(aes(x=xend_min+distance_for_image, y=yend_min, image = circle_crop("/Users/koichiito/Desktop/test.png")), size = 0.2453) +
# #     geom_spatial_label(mapping = aes(x, y, label = label), data = label_df, crs = 3857) +
# #     coord_sf(crs = 3857) +
# #     labs(
# #       x = NULL,
# #       y = NULL,
# #       title = paste0("Difference between real and fake \n road shoulder view images for ", category),
# #       subtitle = NULL,
# #       caption = NULL
# #     ) +
# #     theme_ipsum() +
# #     theme(
# #       plot.title.position = "panel",
# #       plot.title = element_text(hjust = 0.5),
# #       panel.grid.major = element_blank(),
# #       panel.grid.minor = element_blank(),
# #       plot.margin = grid::unit(c(0, 0, 0, 0), "mm"),
# #       axis.text.x = element_blank(), # remove x axis labels
# #       axis.ticks.x = element_blank(), # remove x axis ticks
# #       axis.text.y = element_blank(), # remove y axis labels
# #       axis.ticks.y = element_blank(), # remove y axis ticks
# #       legend.position = c(0.2, 0.6),
# #       legend.title = element_text(size = 18),
# #       legend.text = element_text(size = 15)
# #       # legend.background = element_rect(fill = alpha("white",0.2)),
# #       # legend.justification = c("right", "top"),
# #       # legend.box.just = "right",
# #       # legend.margin = margin(6, 6, 6, 6)
# #     )
# #   ggsave(plot = map, paste0("./reports/figures/pix2pix_filtered/", category, "_difference_map.png"), width = 8, height = 8)
# # }

# # # loop through categories
# # for (category in categories) {
# #   map_diff(line_joined, segmentation_result, category)
# # }
# # map_diff(line_joined, segmentation_result, "building")
# # map_diff(line_joined, segmentation_result, "sky")
# # map_diff(line_joined, segmentation_result, "vegetation")

# map improvement ------------------------------------------------------------------------
map_improvement <- function(line_joined, point_results, category, view, platform, output_dir) {
  # replace all the backslashes with forward slashes
  point_results <- point_results %>%
    mutate(file_path_mly = str_replace_all(file_path_mly, "\\\\", "/")) %>%
    mutate(file_path_gan = str_replace_all(file_path_gan, "\\\\", "/")) %>% 
    mutate(file_path_gsv = str_replace_all(file_path_gsv, "\\\\", "/"))
  # max difference
  max_diff <- point_results %>%
    slice_max(., order_by = improvement, n = 1, with_ties = FALSE)
  min_diff <- point_results %>%
    slice_min(., order_by = improvement, n = 1, with_ties = FALSE)
  # decide the xend and yend
  distance_for_image <- 355
  xend_1 <- 11552450
  yend_1 <- 143900
  xend_2 <- 11552100
  yend_2 <- 143150
  if (st_coordinates(max_diff)[1, 1] > st_coordinates(min_diff)[1, 1]) {
    xend_min <- xend_1
    yend_min <- yend_1
    xend_max <- xend_2
    yend_max <- yend_2
  } else {
    xend_min <- xend_2
    yend_min <- yend_2
    xend_max <- xend_1
    yend_max <- yend_1
  }

  label_df <- tibble(
    x = c(
      xend_min, # the major label
      xend_max, # , # the major label
      xend_min + distance_for_image,
      xend_min, # the middle image
      xend_min - distance_for_image,
      xend_max + distance_for_image,
      xend_max, # the middle image
      xend_max - distance_for_image
    ),
    y = c(
      yend_min + 200, # the major label
      yend_max + 200, # the major label
      yend_min + 150,
      yend_min + 150,
      yend_min + 150,
      yend_max + 150,
      yend_max + 150,
      yend_max + 150
    ),
    label = c(
      paste0("Minimum improvement: ", signif(min_diff["improvement"][[1]] * 100, digit = 2), "%"),
      paste0("Maximum improvement: ", signif(max_diff["improvement"][[1]] * 100, digit = 2), "%"),
      "fake",
      "real",
      "Google Street View",
      "fake",
      "real",
      "Google Street View"
    )
  )
  max_abs_improvement <- 15
  print(max_abs_improvement)
  # map the sf object
  map <- basemap_ggplot(st_bbox(line_joined), map_service = "carto", map_type = "light_no_labels", alpha = 1) +
    geom_sf(data = line_joined, mapping = aes(color = improvement * 100), linewidth = 1) +
    scale_colour_gradientn(
      colours = paletteer::paletteer_c("ggthemes::Classic Red-Blue", n = 256),
      limits = c(-max_abs_improvement, max_abs_improvement),  # Setting limits to ensure both ends are represented in the legend
      name = paste0("% improvement \n for ", category),
      na.value = "grey50",
      breaks = seq(-max_abs_improvement, max_abs_improvement, length.out = 5),
      labels = paste0(round(seq(-max_abs_improvement, max_abs_improvement, length.out = 5)), "%"),
      guide = guide_colorbar(
        title.position = "top",
        title.hjust = 0,
        ticks.colour = "black",
        label.theme = element_text(size = 12, margin = margin(t = 0, r = 0, b = 0, l = 0, unit = "pt")))
    ) +
    # scale_color_gradient(low = "#FFFFFF", high = "#7F312F", name = paste0("% improvement for ", category)) +
    # scale_color_gradientn(
    #   colors = paletteer_d("colorBlindness::Blue2DarkOrange12Steps"), 
    #   name = paste0("% improvement for ", category),
    #   midpoint = 0
    # ) +
    # scale_color_gradient2(
    #   low = "#e76f51", 
    #   mid = "#e9c46a", 
    #   high = "#264653", 
    #   midpoint = 0, 
    #   limits = c(-max_abs_improvement, max_abs_improvement),  # Setting limits to ensure both ends are represented in the legend
    #   name = paste0("% improvement \n for ", category),
    #   na.value = "grey50",
    #   breaks = seq(-max_abs_improvement, max_abs_improvement, length.out = 5),
    #   labels = paste0(round(seq(-max_abs_improvement, max_abs_improvement, length.out = 5)), "%"),
    #   guide = guide_colorbar(
    #     title.position = "top",
    #     title.hjust = 0,
    #     ticks.colour = "black",
    #     label.theme = element_text(size = 12, margin = margin(t = 0, r = 0, b = 0, l = 0, unit = "pt"))
    #   )
    # ) +
    # add pointers
    geom_spatial_point(aes(x = st_coordinates(max_diff)[1, 1], y = st_coordinates(max_diff)[1, 2]), crs = 3857) +
    geom_spatial_segment(aes(x = st_coordinates(max_diff)[1, 1], y = st_coordinates(max_diff)[1, 2], xend = xend_max, yend = yend_max),
      crs = 3857, lineend = "round"
    ) +
    geom_spatial_point(aes(x = st_coordinates(min_diff)[1, 1], y = st_coordinates(min_diff)[1, 2]), crs = 3857) +
    geom_spatial_segment(aes(x = st_coordinates(min_diff)[1, 1], y = st_coordinates(min_diff)[1, 2], xend = xend_min, yend = yend_min),
      crs = 3857, lineend = "round"
    ) +
    # add images
    geom_image(aes(x = xend_max, y = yend_max, image = circle_crop(max_diff[["file_path_mly"]])), size = 0.2453) +
    geom_image(aes(x = xend_max + distance_for_image, y = yend_max, image = circle_crop(max_diff[["file_path_gan"]])), size = 0.2453) +
    geom_image(aes(x = xend_max - distance_for_image, y = yend_max, image = circle_crop(max_diff[["file_path_gsv"]])), size = 0.2453) +
    geom_image(aes(x = xend_min, y = yend_min, image = circle_crop(min_diff[["file_path_mly"]])), size = 0.2453) +
    geom_image(aes(x = xend_min + distance_for_image, y = yend_min, image = circle_crop(min_diff[["file_path_gan"]])), size = 0.2453) +
    geom_image(aes(x = xend_min - distance_for_image, y = yend_min, image = circle_crop(min_diff[["file_path_gsv"]])), size = 0.2453) +
    # geom_image(aes(x=xend_max-distance_for_image, y=yend_max, image = circle_crop("/Users/koichiito/Desktop/test.png")), size = 0.2453) +
    # geom_image(aes(x=xend_max+distance_for_image, y=yend_max, image = circle_crop("/Users/koichiito/Desktop/test.png")), size = 0.2453) +
    # geom_image(aes(x=xend_min-distance_for_image, y=yend_min, image = circle_crop("/Users/koichiito/Desktop/test.png")), size = 0.2453) +
    # geom_image(aes(x=xend_min+distance_for_image, y=yend_min, image = circle_crop("/Users/koichiito/Desktop/test.png")), size = 0.2453) +
    geom_spatial_label(mapping = aes(x, y, label = label), data = label_df, crs = 3857) +
    coord_sf(crs = 3857) +
    labs(
      x = NULL,
      y = NULL,
      title = paste0("Improvement from Google Street View to \n predicted ", view, " ", str_replace(platform, "_", " "), " view images for ", category),
      subtitle = NULL,
      caption = NULL
    ) +
    theme_ipsum() +
    theme(
      plot.title.position = "panel",
      plot.title = element_text(hjust = 0.5),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      plot.margin = grid::unit(c(0, 0, 0, 0), "mm"),
      panel.background = element_rect(fill = "transparent", colour = NA),
      axis.text.x = element_blank(), # remove x axis labels
      axis.ticks.x = element_blank(), # remove x axis ticks
      axis.text.y = element_blank(), # remove y axis labels
      axis.ticks.y = element_blank(), # remove y axis ticks
      legend.position = c(0.12   , 0.5),
      legend.title = element_text(size = 18, hjust = 0, margin = margin(r = 20, unit = "pt")),
      legend.text = element_text(size = 15),
      # legend.background = element_rect(fill = alpha("white",0.2)),
      # legend.justification = c("left", "bottom"),
      # legend.box.just = "left",
      # legend.margin = margin(6, 6, 6, 6)
    )
  ggsave(plot = map, paste0(output_dir, "/", category, "_improvements_map.png"), width = 8, height = 8)
}

for (target_variable in c("building", "sky", "vegetation")) {
  for (platform in c("road_shoulder", "sidewalk")) {
    view <- ifelse(platform == "road_shoulder", "perspective", "panorama")
    line_joined <- st_read(paste0(
      root_dir, "/case_study/pasir_panjang_road/data/processed/", platform, "/", view, "/",
      target_variable, "_mly_line_joined.geojson"
    )) %>%
      st_transform(3857)
    point_results <- st_read(paste0(
      root_dir, "/case_study/pasir_panjang_road/data/processed/", platform, "/", view, "/",
      target_variable, "_mly_point_results.geojson"
    )) %>%
      st_transform(3857)
    model_name <- ifelse(platform == "road_shoulder", "road_shoulder_cyclegan_perspective", "sidewalk_cyclegan_default")
    output_dir <- paste0("./reports/figures/", model_name)
    # make directory if not exist
    if (!dir.exists(output_dir)) {
      dir.create(output_dir)
    }
    map_improvement(line_joined, point_results, target_variable, view, platform, output_dir)
  }
}


# plot density of bias and change -------------------------------------------------------------
# create a density plot for columns (bias and change)
plot_density_diff <- function(segmentation_result_long, output_dir) {
  ggplot(
    segmentation_result_long,
    aes(x = value * 100, y = variable, fill = category)) +
    geom_density_ridges_gradient(scale = 1.5, size = 0.3, rel_min_height = 0.001) +
    scale_fill_manual(
      values = c(bias = "#FF8E3280", change = "#99F9FF80"),
      labels = c(bias = "Google Street View", change = "CycleGAN and LightGBM"),
      aesthetics = "fill",
      name = paste0("% difference"),
      guide = guide_legend(override.aes = list(alpha = 0.5))
    ) +
    labs(
      title = paste0("Distribution of differences from ground-truth in percentage"),
      subtitle = "the closer to 0, the better",
      x = "% difference",
      y = ""
    ) +
    theme_ipsum() +
    # margin should be 0
    theme(plot.margin = grid::unit(c(0, 0, 0, 0), "mm"),
          panel.background = element_rect(fill = "transparent", colour = NA),
          plot.subtitle = element_text(hjust = 0.5))
  ggsave(paste0(output_dir, "/difference_distribution.png"), width = 8, height = 4)
}
# color ref: #1E8E99FF #993F00FF
for (platform in c("road_shoulder", "sidewalk")){
  view <- ifelse(platform == "road_shoulder", "perspective", "panorama")
  # initialize df to store the results
  segmentation_result_long <- tibble()
  for (target_var in c("building", "sky", "vegetation")){
    point_results <- st_read(paste0(
      root_dir, "/case_study/pasir_panjang_road/data/processed/", platform, "/", view, "/",
      target_var, "_mly_point_results.geojson")) %>% 
      st_drop_geometry() %>%
      # select and rename bias to bias_ + target_var and change to change_ + target_var
      select(mly_id, bias, change) %>%
      rename_with(~ ifelse(. %in% c("bias", "change"), paste0(target_var, "_", .), .))
    # merge with segmentation_result_long
    if (length(segmentation_result_long) == 0){
      segmentation_result_long <- point_results
    }
    else{
      segmentation_result_long <- segmentation_result_long %>% 
        left_join(., point_results, by = "mly_id")
    }
  }
  # convert to long format
  segmentation_result_long <- segmentation_result_long %>%
    pivot_longer(
      cols = -mly_id, 
      names_to = c("variable", "category"), 
      names_pattern = "([a-z_]+)_(bias|change)"
    )
  model_name <- ifelse(platform == "road_shoulder", "road_shoulder_cyclegan_perspective", "sidewalk_cyclegan_default")
  output_dir <- paste0("./reports/figures/", model_name)
  # make directory if not exist
  if (!dir.exists(output_dir)) {
    # create recursively
    dir.create(output_dir, recursive = TRUE)
  }
  plot_density_diff(segmentation_result_long, output_dir)
}
  

# # density map -------------------------------------------------------------
# plot_density <- function(segmentation_result_long) {
#   ggplot(
#     segmentation_result_long,
#     aes(x = difference * 100, y = category, fill = stat(x))
#   ) +
#     geom_density_ridges_gradient(scale = 1.5, size = 0.3, rel_min_height = 0.001) +
#     scale_fill_gradient(low = "#FFFFFF", high = "#7F312F", name = paste0("% difference")) +
#     labs(
#       title = "Distribution of differences in percentage",
#       x = "% difference",
#       y = ""
#     ) +
#     theme_ipsum()
#   ggsave(paste0("./reports/figures/pix2pix_filtered/difference_distribution.png"), width = 8, height = 4)
# }
# segmentation_result_long <- segmentation_result %>%
#   select(c(file_name, diff_building, diff_sky, diff_vegetation)) %>%
#   gather(
#     key = "category", value = "difference",
#     diff_building, diff_sky, diff_vegetation
#   ) %>%
#   mutate(category = str_remove_all(category, "diff_")) %>%
#   mutate(category = factor(category, level = c("vegetation", "sky", "building")))
# plot_density(segmentation_result_long)
