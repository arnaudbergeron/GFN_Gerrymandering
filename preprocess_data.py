import os
import subprocess
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
R_EXE = os.getenv("R_EXE")

if not R_EXE:
    raise ValueError("Path to Rscript.exe is not set. Please check your .env file.")

##################################################
## Get the `data` folder from the `redist` repo ##
##################################################

repo_url = "https://github.com/alarm-redist/redist.git"
repo_name = "redist"
data_dir = Path("./data")

if not data_dir.exists():
    # Clone the GitHub repository
    subprocess.run(["git", "clone", repo_url], check=True)

    # Move the `data` directory from the cloned repository
    source_path = Path(repo_name) / "data"
    if source_path.exists():
        shutil.move(str(source_path), str(data_dir))
        print(f"Moved {source_path} to {data_dir}")
    else:
        print(f"Data directory not found in {source_path}")

import os
import subprocess
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
R_EXE = os.getenv("R_EXE")
repo_name = "redist"
data_dir = Path("./data")

####################################
## Convert the .rda files to .csv ##
####################################

# Replace the R script name with the new one
r_script_path = Path("convert_rda_to_gpkg.R")

# Run the R script to convert .rda files to GeoPackages
if r_script_path.exists():
    subprocess.run([R_EXE, str(r_script_path)], check=True)
    print("R script executed successfully.")
else:
    print(f"R script not found: {r_script_path}")

#############
## Cleanup ##
#############

# Clean up all files that don't end with .gpkg
files_to_check = list(data_dir.glob("*"))  # Get all files in the directory
for file in files_to_check:
    if not file.name.endswith(".csv") or file.name.endswith("obj.csv"):
        try:
            file.unlink()  # Delete the file
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Failed to delete {file}: {e}")
