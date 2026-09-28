# DSN Mart Sales Prediction

A regression model that predicts total sales for a given product at a given store. The competition is scored on RMSE (lower is better).

## What's in the Notebook

The notebook runs top to bottom in this order:

1. **Setup** — imports and a couple of small compatibility patches for CatBoost 
2. **Exploratory Data Analysis** — shape and missing-value check, a look at the numeric and categorical columns, and the specific findings that shaped later decisions (a right-skewed target, a real store-by-category sales effect, inconsistent category text casing).
3. **Cleaning, Feature Engineering, and Target Encoding**
   - Missing values in `product_weight_kg` and `store_size` filled using train-only statistics.
   - `shelf_visibility == 0` treated as a missing reading (not a genuine zero) and imputed by category median.
   - Two new features: `price_per_kg` and `store_age_bucket`.
   - Out-of-fold target (mean) encoding for store, category, and the store × category combination.
4. **One-Hot Encoding** of the remaining categorical columns.
5. **Model Development**
   - A train/validation split and scaling (fit on train only).
   - A baseline comparison across Linear Regression, Ridge, ElasticNetCV, Random Forest, AdaBoost, Bagging, Gradient Boosting, LightGBM, and CatBoost, all scored on 5-fold cross-validated RMSE.
   - Hyperparameter tuning (via `RandomizedSearchCV`) for Gradient Boosting and CatBoost.
   - CatBoost tried a second way, using its native categorical handling instead of one-hot encoding.
   - A side-by-side CV comparison of the tuned models, and a simple weighted blend of the two strongest ones.
6. **Final Model and Submission** — refits the chosen model (`final_choice`) on all of the training data and writes `submission.csv`.

## How to Run It

**Requirements:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `catboost`, `lightgbm`. All of these come pre-installed on Google Colab; if `lightgbm` fails to import, run `!pip install lightgbm` in a cell above it.

**Data:** place `train.csv`, `test.csv`, and `sample_submission.csv` in the same working directory as the notebook (or upload them to your Colab session) before running. Colab sessions reset between visits, so you'll need to re-upload these each time you start a fresh runtime.

**Run order:** use *Runtime → Restart and run all* rather than running cells one at a time out of order. Several cells depend on variables created earlier (for example, the CatBoost tuning results are reused by the native-categorical comparison further down), so an out-of-order run can produce confusing `NameError`s.

**Choosing the final model:** near the bottom, set `final_choice` to `"gb"`, `"cat"`, `"cat_native"`, or `"blend"`, based on whichever has the **lowest cross-validated RMSE** in the comparison table.

CatBoost (tuned) was consistently the strongest single model across every stage of this project. Target encoding and CatBoost's native categorical handling both gave small, real improvements over the initial pipeline, but neither produced a dramatic jump, suggesting the current feature set is close to what this data can support without new external information.

## Repository Contents

Data available for download from the [Kaggle competition page](https://www.kaggle.com/competitions/dsn-bootcamp-qualification-hackathon-2026-ml-track)).
