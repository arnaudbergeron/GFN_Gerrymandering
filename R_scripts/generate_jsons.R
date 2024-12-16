# workspace starts in R
setwd("code/GFN_Gerrymandering/R_scripts")

# install libraries to transfer to json
install.packages("jsonlite")
install.packages("dplyr")

# load libraries
library(jsonlite)

# Pennsylvania load and write
data_PA <- readRDS("../data/PA_cd_2020/PA_cd_2020_map.rds")
data_plans_PA <- readRDS("../data/PA_cd_2020/PA_cd_2020_plans.rds")

write_json(data_PA, "../data/PA_raw_data.json", pretty = TRUE)
write_json(data_plans_PA, "../data/PA_raw_plans.json", pretty = TRUE)

# Iowa load and write
data_IA <- readRDS("../data/IA_cd_2020/IA_cd_2020_map.rds")
data_plans_IA <- readRDS("../data/IA_cd_2020/IA_cd_2020_plans.rds")

write_json(data_IA, "../data/IA_raw_data.json", pretty = TRUE)
write_json(data_plans_IA, "../data/IA_raw_plans.json", pretty = TRUE)

# Massachusetts load and write
data_MA <- readRDS("../data/MA_cd_2020/MA_cd_2020_map.rds")
data_plans_MA <- readRDS("../data/MA_cd_2020/MA_cd_2020_plans.rds")

write_json(data_MA, "../data/MA_raw_data.json", pretty = TRUE)
write_json(data_plans_MA, "../data/MA_raw_plans.json", pretty = TRUE)

# Michigan load and write
data_MI <- readRDS("../data/MI_cd_2020/MI_cd_2020_map.rds")
data_plans_MI <- readRDS("../data/MI_cd_2020/MI_cd_2020_plans.rds")

write_json(data_MI, "../data/MI_raw_data.json", pretty = TRUE)
write_json(data_plans_MI, "../data/MI_raw_plans.json", pretty = TRUE)