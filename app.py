import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the pre-trained preprocessor and model
preprocessor = joblib.load('artifacts/preprocessor.joblib')
model = joblib.load('artifacts/lgbm_model.joblib')

# App title
st.title("Severity Prediction App")
st.write("""
This app allows you to upload a CSV or Parquet file, preprocess it, and predict the `Severity` column using the trained model.
""")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV or Parquet file", type=["csv", "parquet"])

if uploaded_file:
    # Read uploaded file
    file_type = uploaded_file.name.split('.')[-1]
    if file_type == "csv":
        data = pd.read_csv(uploaded_file)
    elif file_type == "parquet":
        data = pd.read_parquet(uploaded_file)

    # Display the uploaded data
    st.write("Uploaded Data Preview:")
    st.dataframe(data.head())

    try:
        # Drop unnecessary columns
        drop_columns = [
            'ID', 'Source', 'End_Lat', 'End_Lng', 'Weather_Timestamp', 'Description', 
            'Street', 'City', 'County', 'State', 'Zipcode', 'Country', 'Airport_Code', 
            'Wind_Chill(F)', 'Wind_Direction', 'Precipitation(in)', 'Weather_Condition',
            'Turning_Loop', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight',
            'Severity'
        ]
        data = data.drop(columns=[col for col in drop_columns if col in data.columns], errors='ignore')

        # Handle datetime and calculate duration
        data['Start_Time'] = pd.to_datetime(data['Start_Time'], format='mixed', errors='coerce')
        data['End_Time'] = pd.to_datetime(data['End_Time'], format='mixed', errors='coerce')
        data['Duration'] = (data['End_Time'] - data['Start_Time']).dt.total_seconds() / 60
        data = data.drop(columns=['Start_Time', 'End_Time'], errors='ignore')

        # Preprocess the data
        test_preprocessed = pd.DataFrame(preprocessor.transform(data), columns=preprocessor.get_feature_names_out())

        # Drop specific correlated features if present
        corr_features = {'cat__Sunrise_Sunset_Night', 'cat__Timezone_US/Pacific'}
        if corr_features.issubset(test_preprocessed.columns):
            test_preprocessed.drop(columns=corr_features, inplace=True)

        # Predict using the model
        predictions = model.predict(test_preprocessed)
        data['Severity'] = predictions

        # Display the results
        st.write("Predicted Data Preview:")
        st.dataframe(data.head())

        # Allow users to download the file
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv_data = convert_df(data)
        st.download_button(
            label="Download Predicted Data as CSV",
            data=csv_data,
            file_name="predicted_data.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
