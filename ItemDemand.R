# Setting up work environment and libraries -------------------------------

setwd(dir = "C:/Users/camer/Documents/Stat 348/ItemDemand/")

library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)


# Parallel Processing

# library(doParallel)
# parallel::detectCores() #How many cores do I have?
# cl <- makePSOCKcluster(num_cores)
# registerDoParallel(cl)
# #code
# stopCluster(cl)

# Initial reading of the data

rawdata <- vroom(file = "train.csv") %>%
  mutate(sales=factor(sales))
test_input <- vroom(file = "test.csv")

# Recipes

my_recipe <- recipe(type ~ ., data = rawdata) %>%
  update_role(id, new_role="id") %>% 
  step_mutate_at(color,fn = factor) %>% 
  step_dummy(color) %>% 
  step_rm(id) %>% 
  step_range(all_numeric_predictors(), min=0, max=1) #%>% 
step_normalize(all_numeric_predictors())


prep_recipe <- prep(my_recipe)
baked_data <- bake(prep_recipe, new_data = rawdata)


# Write and read function

format_and_write <- function(predictions, file){
  final_preds <- predictions %>%
    mutate(type = .pred_class) %>%
    mutate(id = test_input$id) %>%
    dplyr::select(id, type)
  
  #vroom_write(final_preds,file,delim = ",")
  #save(file="./MyFile.RData", list=c("object1", "object2",...))
}


# EDA ---------------------------------------------------------------------


