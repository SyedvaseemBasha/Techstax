# prompt: now analize above entire code to create readme 

# Accident Severity Prediction

This project focuses on predicting the severity of traffic accidents using machine learning.  The model is trained on a dataset of accident records, and then deployed using Streamlit for user-friendly prediction.

## Data and Preprocessing

The project utilizes a dataset of traffic accidents containing various features such as location, time, weather conditions, and more.  Crucially, the `Severity` column serves as the target variable for prediction.

**Preprocessing steps include:**

* Dropping irrelevant or redundant columns.
* Converting `Start_Time` and `End_Time` to datetime objects and calculating the `Duration` of the accident.
* Handling missing values and converting data types as necessary.
* Feature scaling and encoding using a `ColumnTransformer` which is saved using `joblib`.

## Model Training

Multiple classification models are evaluated:

* SGD Classifier
* LightGBM
* Random Forest
*(Other models like Logistic Regression, Ridge Classifier, K-Neighbors Classifier, Decision Tree, SVM, XGBClassifier, CatBoosting Classifier, and AdaBoost Classifier were initially considered but final model used was LightGBM)*

The models are trained and evaluated using metrics such as Accuracy, Precision, Recall, and F1 Score.  

Due to resource constraints, hyperparameter tuning was not performed exhaustively. However, the best model was selected and saved for deployment.

**The Final Model:**

A LightGBM classifier was selected as the final model due to good performance.  The model is saved using `joblib` for later use.

## Deployment (Streamlit App)

The trained LightGBM model and preprocessor are deployed using Streamlit to create a user-friendly web application.

**App Features:**

* File upload: Users can upload CSV or Parquet files containing accident data.
* Data preview: Displays a preview of the uploaded data.
* Prediction: The app preprocesses the uploaded data and uses the trained model to predict the `Severity` of the accidents.
* Download: Users can download the predicted data as a CSV file.

**How to run the Streamlit app:**

1. Install the required libraries:
   ```bash
   pip install -r requirements.txt 
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run your_app_file.py  # Replace your_app_file.py with the name of your Python file.
   ```

## Usage

1. Visit the app in your browser.
2. Upload a CSV or Parquet file with the necessary features.
3. The app will predict the `Severity` and display the results.
4. Download the predictions as a CSV file.


## Further Development

* **More comprehensive hyperparameter tuning**  to optimize model performance.
* **Exploring additional features or data sources** to enhance the model's accuracy.
* **Investigating alternative models**  that could provide further improvement.
* **Deploy the app to a cloud platform** for greater accessibility.