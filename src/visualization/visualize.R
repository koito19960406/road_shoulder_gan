# load necessary libraries
pacman::p_load(tidyverse, dotenv, paletteer, hrbrthemes)

cyclegan_generate_stats_from_log <- function(root_dir, experiment_name, line_interval=10, nb_data=10800, enforce_last_line=TRUE) {
  # define necessary parameters
  loss_log_filepath <- file.path(root_dir, "models", experiment_name, "loss_log.txt")
  if (!file.exists(loss_log_filepath)){
    message(paste0(loss_log_filepath, " doesn't exist"))
    return(NULL)
  }
  
  # check if the output already exists
  output_dir <- file.path("./reports/figures", experiment_name)
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  output_filepath <- file.path(output_dir, "train_loss_plot.png")
  
  # read data from the file
  text <- readChar(loss_log_filepath, file.info(loss_log_filepath)$size)
  
  # split the text based on the decorator lines
  sessions <- str_split(text, "================ Training Loss ")[[1]]
  
  # use only the last session
  lines <- str_split(sessions[length(sessions)], "\n")[[1]]
  
  # choose the lines to use for plotting
  lines_for_plot <- lines[seq(2, length(lines), by = line_interval)]
  if (enforce_last_line) {
    lines_for_plot <- c(lines_for_plot, tail(lines, 1))
  }
  
  # initialize dict with loss names
  dicts <- list()
  dicts$epoch <- numeric(0)
  parts <- str_split(str_split(lines_for_plot[[1]], "\\) ")[[1]][2], " ")[[1]]
  for (i in seq(1, length(parts), by = 2)) {
    if (parts[i] != "\r"){
      dicts[[str_sub(parts[i], end=-2)]] <- numeric(0)
    }
  }
  # extract all data
  pattern <- "epoch: ([0-9]+), iters: ([0-9]+)"
  for (l in lines_for_plot) {
    search <- str_match(l, pattern)
    if (is.na(search[1,2])) next
    epoch <- as.numeric(search[1,2])
    epoch_floatpart <- as.numeric(search[1,3])/nb_data
    dicts$epoch <- c(dicts$epoch, epoch + epoch_floatpart) # to allow several plots for the same epoch
    parts <- str_split(str_trim(str_split(l, '\\) ')[[1]][2]), ' ')[[1]]
    for (i in seq(1, length(parts), by = 2)) {
      if (parts[i] != "\r"){
        tryCatch({
          dicts[[str_sub(parts[i], end=-2)]] <- c(dicts[[str_sub(parts[i], end=-2)]], as.numeric(parts[i+1]))
        }, error = function(e) {
          dicts[[str_sub(parts[i], end=-2)]] <- numeric(0)
          dicts[[str_sub(parts[i], end=-2)]] <- c(dicts[[str_sub(parts[i], end=-2)]], as.numeric(parts[i+1]))
        })
      }
    }
  }
  
  # convert list to data frame
  df <- as.data.frame(dicts)
  
  # start ggplot object
  p <- ggplot() + 
    labs(title = experiment_name, x = "Epoch", y = "Loss") +
    theme_ipsum()
  
  # get color palette
  palette <- paletteer_d("rcartocolor::Safe")
  
  # add lines to the plot for each key
  for (key in names(df)) {
    condition <- case_when(
      str_detect(experiment_name, "miou") ~ key != "epoch" & str_detect(key, "_A"),
      !str_detect(experiment_name, "miou") ~ key != "epoch" & str_detect(key, "_A") & !str_detect(key, "miou")
    )
    if (condition) {
      # if (str_detect(key, "cycle")) {
      #   line_width <- 1
      # } else {
      #   line_width <- 0.3
      # }
      line_width <- 0.5
      p <- p + geom_line(data = df, aes_string(x = "epoch", y = key, color = paste("factor('", key, "')")), size = line_width)
    }
  }
  
  # finish the plot with a legend and save it
  p <- p + 
    scale_color_manual(values = palette) +  # apply color palette
    theme(legend.position = "right") +  # position the legend
    guides(color = guide_legend(title = "Loss"))  # set the title of the legend
  
  ggsave(filename = output_filepath, plot = p, width = 10, height = 3)
}

pix2pix_generate_stats_from_log <- function(root_dir, experiment_name, line_interval=10, nb_data=10800, enforce_last_line=TRUE) {
  # Define necessary parameters
  loss_log_filepath <- file.path(root_dir, "models", experiment_name, "loss_log.txt")
  
  if (!file.exists(loss_log_filepath)){
    message(paste0(loss_log_filepath, " doesn't exist"))
    return(NULL)
  }
  
  gen_train_loss_filepath <- paste0("./reports/figures/", experiment_name, "/gen_train_loss.png")
  det_train_loss_filepath <- paste0("./reports/figures/", experiment_name, "/det_train_loss.png")
  
  # read data from the file
  text <- readChar(loss_log_filepath, file.info(loss_log_filepath)$size)
  
  # split the text based on the decorator lines
  sessions <- str_split(text, "================ Training Loss ")[[1]]
  
  # use only the last session
  lines <- str_split(sessions[length(sessions)], "\n")[[1]]
  
  # Choose the lines to use for plotting
  lines_for_plot <- lines[seq(2, length(lines), by = line_interval)]
  if (enforce_last_line) {
    lines_for_plot <- c(lines_for_plot, tail(lines, 1))
  }
  
  # Initialize list with loss names
  dicts <- list()
  dicts$epoch <- numeric(0)
  
  parts <- strsplit(strsplit(lines_for_plot[[1]], "\\) ")[[1]][2], " ")[[1]]
  for (i in seq(1, length(parts), by = 2)) {
    if (parts[i] != "\r"){
      dicts[[substring(parts[i], 1, nchar(parts[i])-1)]] <- numeric(0)
    }
  }
  
  # Extract all data
  for (l in lines_for_plot) {
    search <- str_match(l, "epoch: ([0-9]+), iters: ([0-9]+)")
    if (is.na(search[1,2])) next
    epoch <- as.numeric(search[1,2])
    epoch_floatpart <- as.numeric(search[1,3])/nb_data
    dicts$epoch <- c(dicts$epoch, epoch + epoch_floatpart) # To allow several plots for the same epoch
    parts <- strsplit(strsplit(l, "\\) ")[[1]][2], " ")[[1]]
    for (i in seq(1, length(parts), by = 2)) {
      if (parts[i] != "\r"){
        tryCatch({
          dicts[[substring(parts[i], 1, nchar(parts[i])-1)]] <- c(dicts[[substring(parts[i], 1, nchar(parts[i])-1)]], as.numeric(parts[i+1]))
        }, error = function(e) {
          dicts[[substring(parts[i], 1, nchar(parts[i])-1)]] <- numeric(0)
          dicts[[substring(parts[i], 1, nchar(parts[i])-1)]] <- c(dicts[[substring(parts[i], 1, nchar(parts[i])-1)]], as.numeric(parts[i+1]))
        })
      }
    }
  }
  
  # Convert list to data frame
  df <- as.data.frame(dicts)
  
  # get color palette
  palette <- paletteer_d("rcartocolor::Safe")
  
  # Start ggplot object
  p <- ggplot() + 
    labs(title = experiment_name, x = "Epoch", y = "Loss") +
    theme_ipsum()
  
  # Plot generator loss
  for (key in names(df)) {
    condition <- case_when(
      str_detect(experiment_name, "miou") ~ key != "epoch" & !str_detect(key, "D_"),
      !str_detect(experiment_name, "miou") ~ key != "epoch" & !str_detect(key, "D_") & !str_detect(key, "miou")
    )
    if (condition) {
      p <- p + geom_line(data = df, aes_string(x = "epoch", y = key, color = paste("factor('", key, "')")), size = 0.5)
    }
  }
  
  # Finish the plot with a legend and save it
  p <- p + 
    scale_color_manual(values = palette) +  # Apply color palette
    theme(legend.position = "right") +  # Position the legend
    guides(color = guide_legend(title = "Loss"))  # Set the title of the legend
  
  ggsave(filename = gen_train_loss_filepath, plot = p, width = 10, height = 3)
  
  # Reset the plot
  p <- ggplot() + 
    labs(title = experiment_name, x = "Epoch", y = "Loss") +
    theme_ipsum()
  
  # Plot detector loss
  for (key in names(df)) {
    condition <- case_when(
      str_detect(experiment_name, "miou") ~ key != "epoch" & !str_detect(key, "G_"),
      !str_detect(experiment_name, "miou") ~ key != "epoch" & !str_detect(key, "G_") & !str_detect(key, "miou")
    )
    if (condition) {
      p <- p + geom_line(data = df, aes_string(x = "epoch", y = key, color = paste("factor('", key, "')")), size = 0.5)
    }
  }
  
  # Finish the plot with a legend and save it
  p <- p + 
    scale_color_manual(values = palette) +  # Apply color palette
    theme(legend.position = "right") +  # Position the legend
    guides(color = guide_legend(title = "Loss"))  # Set the title of the legend
  
  ggsave(filename = det_train_loss_filepath, plot = p, width = 10, height = 3)
}


seg_corr_visualizer <- function(input_file, output_file) {
  
  # Read csv and retain all the columns except for pid col (first col)
  df <- read_csv(input_file) %>%
    dplyr::select(-1)
  
  # Get unique list of column names that contain either one of "gsv_", "mly_", "gan_" and replace them with empty string
  # get column names that contain either one of "gsv_", "mly_", "gan_"
  unique_col_names <- df %>%
    select(matches("^(gsv_|mly_|gan_)")) %>%
    names() %>%
    na.omit() %>%
    str_replace_all("(gsv_|mly_|gan_)", "") %>%
    unique() 

  for (label in unique_col_names) {
    print(label)
    df_filtered <- df %>%
      dplyr::select(contains(label))

if (ncol(df_filtered) > 0) {
      # Get the correlation matrix
      corrMatrix <- cor(df_filtered)
      
      # Melt the correlation matrix for ggplot
      corr_melted <- as.data.frame(as.table(corrMatrix))
      colnames(corr_melted) <- c("x", "y", "value")
      
      # Compute the R^2 value for each pair
      corr_melted$r_squared <- corr_melted$value^2
      
      # Plot using ggplot
      p <- ggplot(corr_melted, aes(x=x, y=y, fill=value)) +
        geom_tile() +
        scale_fill_paletteer_c("viridis::inferno") +
        theme_ipsum() +
        geom_text(data=subset(corr_melted, r_squared >= 0.9), aes(label=sprintf("%.2f", r_squared)), size=5, color="black", na.rm = TRUE) + 
        geom_text(data=subset(corr_melted, r_squared < 0.9), aes(label=sprintf("%.2f", r_squared)), size=5, color="white", na.rm = TRUE) +
        labs(title = "", x = "", y = "") +
        theme(
          axis.text.x = element_text(size=14),  # Adjust font size for x-axis tick labels
          axis.text.y = element_text(size=14),  # Adjust font size for y-axis tick labels
          legend.text = element_text(size=14),   # Adjust font size for colorbar labels  
          plot.margin = margin(0, 0, 0, 0)      # Remove all margins
        )


      # Save the plot
      ggsave(filename = paste(output_file, paste0("segmentation_correlation_matrix_", label, ".png"), sep="/"), plot = p)
    }
  }
}


# define necessary parameters
root_dir <- "./"

# Create the list of drives
drives <- c(paste0(LETTERS[4:26], ":/"), "/Volumes/ExFAT/")

# Loop through the drives and check if the path exists
for (drive in drives) {
  if (file.exists(file.path(drive, "road_shoulder_gan"))) {
    root_dir <- file.path(drive, "road_shoulder_gan")
    break
  }
}
print(root_dir)

experiment_name_list <- list.dirs(path = paste0(root_dir, "/models/"), full.names = FALSE, recursive = FALSE)
print(experiment_name_list)
for (experiment_name in experiment_name_list){
  print(experiment_name)
  # if (str_detect(experiment_name, "cyclegan")){
  #   print(experiment_name)
  #   cyclegan_generate_stats_from_log(root_dir, experiment_name, line_interval = 50)
  # }
  # else if (str_detect(experiment_name, "pix2pix")){
  #   print(experiment_name)
  #   pix2pix_generate_stats_from_log(root_dir, experiment_name, line_interval = 50)
  # }
  # Segmentation result correlation plot
  input_file <- file.path(root_dir, "models", experiment_name, "segmentation_result", "segmentation_result.csv")
  output_folder <- paste0("./reports/figures/", experiment_name)

  # Call the seg_corr_visualizer function
  tryCatch({
    seg_corr_visualizer(input_file, output_folder)
  }, error = function(e) {
    print(e)
  })
}