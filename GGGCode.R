#############################################
## GHOULS, GHOSTS, AND GOBLINS KAGGLE COMP ##
#############################################

# libraries
library(vroom)
library(tidymodels)
library(tidyverse)

# read in data
train_data <- vroom("/Users/eliseclark/Documents/Fall 2025/Stat 348/GGGComp/ghouls-goblins-and-ghosts-boo/train.csv")
testData <- vroom("/Users/eliseclark/Documents/Fall 2025/Stat 348/GGGComp/ghouls-goblins-and-ghosts-boo/test.csv")

train_data <- train_data %>%
  mutate(type = as.factor(type))


############### CLASSIFICATION RF ##################

# forest recipe
forest_recipe <- recipe(type ~ ., data = train_data) %>%
  step_mutate_at((color), fn = factor) %>% 
  step_dummy(all_nominal_predictors())


forest_prep <- prep(forest_recipe)
forest_data <- bake(forest_prep, new_data = train_data)


# model 
forest_lm <- rand_forest(mtry = tune(),
                         min_n = tune(),
                         trees = tune()) %>%
  set_engine("ranger") %>%
  set_mode("classification")



# workflow
forest_wf <- workflow() %>%
  add_recipe(forest_recipe) %>% add_model(forest_lm) 

# grid of tuning values
tuning_grid <- grid_random(
  mtry(range = c(1, floor(sqrt(ncol(train_data))))),
  min_n(range = c(2, 30)),
  trees(range = c(100,200)),
  size = 25)

# set up K-fold CV
folds <- vfold_cv(train_data, v = 5, repeats = 1, strata = type)


CV_results <- forest_wf %>%
  tune_grid(
    resamples = folds,        
    grid = tuning_grid,
    metrics = metric_set(accuracy),
    control = control_grid(verbose = TRUE))


# find best tuning parameters
bestTune <- CV_results %>%
  select_best(metric = "accuracy")

# finalize workflow and predict
final_wf <- forest_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_data)

# predictions
forest_preds <- final_wf %>%
  predict(new_data = testData, type = "class")


forest_preds <- tibble(
  id = testData$id,
  type = forest_preds$.pred_class)


vroom_write(x = forest_preds, file = "/Users/eliseclark/Documents/Fall 2025/Stat 348/preds_ggg29.csv", delim = ",")





######################### META LEARNER ###########################
library(xgboost)
library(glmnet)
library(stacks)

ggg_recipe <- recipe(type ~ ., data = train_data) %>%
  step_mutate_at(color, fn = factor) %>%
  step_other(color, threshold = 0.005) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())


# Random Forest
rf_model <- rand_forest(mtry = tune(), min_n = tune(), trees = 200) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# XGBoost
xgb_model <- boost_tree(
  trees = 200, learn_rate = tune(), min_n = tune(), tree_depth = tune()) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

# Logistic Regression (baseline linear model)
multinom_model <- multinom_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("classification")



rf_wf <- workflow() %>% add_recipe(ggg_recipe) %>% add_model(rf_model)
xgb_wf <- workflow() %>% add_recipe(ggg_recipe) %>% add_model(xgb_model)
multinom_wf <- workflow() %>% add_recipe(ggg_recipe) %>% add_model(multinom_model)

folds <- vfold_cv(train_data, v = 5, strata = type)


ctrl <- control_grid(save_pred = TRUE, save_workflow = TRUE)

rf_res <- tune_grid(
  rf_wf,
  resamples = folds,
  grid = 20,
  metrics = metric_set(accuracy, mn_log_loss),
  control = ctrl)

xgb_res <- tune_grid(
  xgb_wf,
  resamples = folds,
  grid = 20,
  metrics = metric_set(accuracy, mn_log_loss),
  control = ctrl)

multinom_res <- tune_grid(
  multinom_wf,
  resamples = folds,
  grid = 20,
  metrics = metric_set(accuracy, mn_log_loss),
  control = ctrl)

# Initialize stack
ggg_stack <- stacks() %>%
  add_candidates(rf_res) %>%
  add_candidates(xgb_res) %>%
  add_candidates(multinom_res) %>%
  blend_predictions(metric = metric_set(mn_log_loss, accuracy)) %>%    # use a prob-based metric
  fit_members()

# Blend predictions
ggg_stack <- blend_predictions(ggg_stack, metric = "accuracy")

# Fit members
ggg_stack <- fit_members(ggg_stack)




# Fit on full training data
final_stack <- fit_members(ggg_stack)

# Predict on test set
stack_preds <- predict(final_stack, new_data = testData, type = "class")

# Format for Kaggle submission
submission <- tibble(
  id = testData$id,
  type = stack_preds$.pred_class
)




vroom_write(x = forest_preds, file = "/Users/eliseclark/Documents/Fall 2025/Stat 348/preds_ggg24.csv", delim = ",")










# linea svm with 20 levels 
# cv







