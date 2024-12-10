# Define the directory containing .rda files
input_dir <- "./data"

# Check if the directory exists
if (!dir.exists(input_dir)) {
  stop("The specified input directory does not exist: ", input_dir)
}

# List all .rda files in the directory
rda_files <- list.files(input_dir, pattern = "\\.rda$", full.names = TRUE)

# Load required libraries
if (!requireNamespace("sf", quietly = TRUE)) {
  install.packages("sf")
}
library(sf)

# Process each .rda file
for (rda_file in rda_files) {
  cat("\nProcessing file:", rda_file, "\n")
  
  # Load the .rda file
  load(rda_file)
  
  # Get all objects in the .rda file
  loaded_objects <- ls()
  print(paste("Loaded objects:", paste(loaded_objects, collapse = ", ")))
  
  # Exclude script-defined variables
  script_vars <- c("input_dir", "rda_files", "rda_file", "loaded_objects", "script_vars")
  loaded_objects <- setdiff(loaded_objects, script_vars)
  
  # Process each valid object
  for (obj_name in loaded_objects) {
    obj <- get(obj_name)
    
    if (inherits(obj, "sf")) {
      # Convert to a data frame without geometry
      obj_no_geom <- st_drop_geometry(obj)
      
      # Define the CSV file path
      csv_file <- file.path(input_dir, paste0(tools::file_path_sans_ext(basename(rda_file)), ".csv"))
      
      # Save as CSV
      tryCatch({
        write.csv(obj_no_geom, csv_file, row.names = FALSE)
        cat("Saved:", csv_file, "\n")
      }, error = function(e) {
        cat("Error saving", obj_name, "to CSV:", e$message, "\n")
      })
    } else {
      cat("Skipping:", obj_name, "- not an sf object\n")
    }
  }
  
  # Cleanup: Remove only loaded objects, excluding script-defined variables
  rm(list = loaded_objects)
}

cat("\nProcessing completed for all files in:", input_dir, "\n")
