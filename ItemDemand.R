# Setting up work environment and libraries -------------------------------

setwd(dir = "C:/Users/camer/Documents/Stat 348/ItemDemand/")

library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(doParallel)
library(modeltime)
library(timetk)
library(forecast)

# Parallel Processing

# library(doParallel)
# parallel::detectCores() #How many cores do I have?
# cl <- makePSOCKcluster(num_cores)
# registerDoParallel(cl)
# #code
# stopCluster(cl)

# Initial reading of the data

rawdata <- vroom(file = "train.csv") %>%
  filter(store==3, item==13)

cv_split <- time_series_split(rawdata, assess="3 months", cumulative = TRUE)
# cv_split %>%
#   tk_time_series_cv_plan() %>% #Put into a data frame
#   plot_time_series_cv_plan(date, sales, .interactive=FALSE)


test_input <- vroom(file = "test.csv") %>%
  filter(store==3, item==13)

# Recipes

my_recipe <- recipe(sales ~ ., data = rawdata) %>%
  step_date(date, features="doy") %>%
  step_range(date_doy, min=0, max=pi) %>%
  step_mutate(sinDOY=sin(date_doy), cosDOY=cos(date_doy))


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
# library(patchwork)
# 
# p1<-rawdata %>%
#   filter(store == 1) %>% 
#   filter(item == 1) %>% 
#   pull(sales) %>% 
#   forecast::ggAcf(.)
# 
# p2<-rawdata %>%
#   filter(store == 1) %>% 
#   filter(item == 50) %>%
#   pull(sales) %>% 
#   forecast::ggAcf(.)
# 
# p3<-rawdata %>%
#   filter(store == 10) %>% 
#   filter(item == 1) %>%
#   pull(sales) %>% 
#   forecast::ggAcf(.)
# 
# p4<-rawdata %>%
#   filter(store == 10) %>% 
#   filter(item == 50) %>%
#   pull(sales) %>% 
#   forecast::ggAcf(.)
# 
# p1+p2+p3+p4


# Modeling Time Series rf----------------------------------------------------

library(ranger)

RF_mod <- rand_forest(mtry = tune(),
                       min_n=tune(),
                       trees=500) %>%
  set_engine("ranger") %>%
  set_mode("regression")

RF_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(RF_mod)

tuning_grid <- grid_regular(mtry(range=c(1,10)),
                            min_n(),
                            levels = 4)

folds <- vfold_cv(rawdata, v = 10, repeats=1)

cl <- makePSOCKcluster(4)
registerDoParallel(cl)
CV_results <- RF_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(smape))
stopCluster(cl)

bestTune <- CV_results %>%
  select_best("smape")

final_RF_wf <-
  RF_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=rawdata)


RF_predictions <- final_RF_wf %>%
  predict(new_data = test_input)

preds1 <- RF_predictions %>% 
  mutate(sales = .pred) %>%
  mutate(id = test_input$id) %>%
  dplyr::select(id, sales)

collect_metrics(CV_results) %>%
  filter(mtry==4, min_n==40) %>% 
  pull(mean)

# Exponential Smoothing ---------------------------------------------------

es_model <- exp_smoothing() %>%
  set_engine("ets") %>%
  fit(sales~date, data=training(cv_split))

## Cross-validate to tune model
cv_results <- modeltime_calibrate(es_model,
                                  new_data = testing(cv_split))

## Visualize CV results
p1<-cv_results %>%
modeltime_forecast(
                   new_data = testing(cv_split),
                   actual_data = rawdata
) %>%
plot_modeltime_forecast(.interactive=TRUE)

## Evaluate the accuracy
cv_results %>%
modeltime_accuracy() %>%
table_modeltime_accuracy(
                         .interactive = FALSE
)

## Refit to all data then forecast
es_fullfit <- cv_results %>%
modeltime_refit(data = rawdata)

es_preds <- es_fullfit %>%
  modeltime_forecast(h = "3 months") %>%
  rename(date=.index, sales=.value) %>%
  select(date, sales) %>%
  full_join(., y=test_input, by="date") %>%
  select(id, sales)

p3<-es_fullfit %>%
  modeltime_forecast(h = "3 months", actual_data = rawdata) %>%
  plot_modeltime_forecast(.interactive=FALSE)

plotly::subplot(p1,p2,p3,p4,nrows=2)


# sarima ------------------------------------------------------------------

arima_model <- arima_reg(seasonal_period=365,
                         non_seasonal_ar=5, # default max p to tune
                         non_seasonal_ma=5, # default max q to tune
                         seasonal_ar=2, # default max P to tune
                         seasonal_ma=2, #default max Q to tune
                         non_seasonal_differences=2, # default max d to tune
                         seasonal_differences=2 #default max D to tune
                         ) %>%
  set_engine("auto_arima")

arima_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(arima_model) %>%
  fit(data=training(cv_split))

cv_results <- modeltime_calibrate(arima_model,
                                  new_data = testing(cv_split))

## Visualize CV results
p1<-cv_results %>%
  modeltime_forecast(
    new_data = testing(cv_split),
    actual_data = rawdata
  ) %>%
  plot_modeltime_forecast(.interactive=TRUE)

## Evaluate the accuracy
cv_results %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = FALSE
  )

## Refit to all data then forecast
arima_fullfit <- cv_results %>%
  modeltime_refit(data = rawdata)

arima_preds <- arima_fullfit %>%
  modeltime_forecast(h = "3 months") %>%
  rename(date=.index, sales=.value) %>%
  select(date, sales) %>%
  full_join(., y=test_input, by="date") %>%
  select(id, sales)

p3<-arima_fullfit %>%
  modeltime_forecast(h = "3 months", actual_data = rawdata) %>%
  plot_modeltime_forecast(.interactive=FALSE)

plotly::subplot(p1,p2,p3,p4,nrows=2)


# Facebook's prophet model ------------------------------------------------


# all ---------------------------------------------------------------------

library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(doParallel)
library(modeltime)
library(timetk)
library(forecast)

rawdata <- vroom(file = "/kaggle/input/demand-forecasting-kernels-only/train.csv")
test_input <- vroom(file = "/kaggle/input/demand-forecasting-kernels-only/test.csv")

nStores <- 3#max(rawdata$store)
nItems <- 3#max(rawdata$item)
for(s in 1:nStores){
  for(i in 1:nItems){
    storeItemTrain <- rawdata %>%
    filter(store==s, item==i)
    storeItemTest <- test_input %>%
    filter(store==s, item==i)
    
    cv_split <- time_series_split(storeItemTrain, assess="3 months", cumulative = TRUE)
    my_recipe <- recipe(sales ~ ., data = storeItemTrain) %>%
      step_date(date, features="doy") %>%
      step_range(date_doy, min=0, max=pi) %>%
      step_mutate(sinDOY=sin(date_doy), cosDOY=cos(date_doy))
    ## Fit storeItem models here
    es_model <- exp_smoothing() %>%
      set_engine("ets") %>%
      fit(sales~date, data=training(cv_split))
    
    ## Cross-validate to tune model
    cv_results <- modeltime_calibrate(es_model,
                                      new_data = testing(cv_split))
    

    ## Refit to all data then forecast
    es_fullfit <- cv_results %>%
      modeltime_refit(data = storeItemTrain)
    
    ## Predict storeItem sales
    preds <- es_fullfit %>%
      modeltime_forecast(h = "3 months") %>%
      rename(date=.index, sales=.value) %>%
      select(date, sales) %>%
      full_join(., y=storeItemTest, by="date") %>%
      select(id, sales)
    
    ## Save storeItem predictions
    if(s==1 & i==1){
      all_preds <- preds
    } else {
      all_preds <- bind_rows(all_preds, preds)
    }
    
  }
}

vroom_write(all_preds, file='submission.csv', delim=',')

