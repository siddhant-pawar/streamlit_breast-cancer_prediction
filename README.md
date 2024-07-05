# streamlit_breast-cancer_prediction
ML and DS Projects, this repository contain datasets that used in projects.

This Streamlit application, titled "Breast Cancer Predictor," integrates modules for interactive functionality. It features a page configuration with a custom title and icon, initially expanded sidebar sliders for input values. The main section informs users about the app's purpose in diagnosing breast cancer using machine learning models and manual input sliders. Charts and predictive models are displayed in columns, with additional Pygal charts provided. Finally, CSS styling hides Streamlit's default menu and footer for a cleaner interface.

- ## Breast Cancer Detection Dashboard
This Streamlit application analyzes breast cancer data using machine learning and visualization techniques.

- ## Files:

```{}
streamlit_breast-cancer_prediction
|-- app
  |-- modules.py
  |-- app.py
|-- breastcancer
  |-- eda
    |-- modmain.ipynb
  |-- data.csv
  |-- modmain.ipynb
|-- model
  |--model.pkl
  |--scaler.pkl
```


### Overview
This dashboard allows users to explore and predict breast cancer diagnoses based on various cell nuclei measurements.

- Setup :-
  1) To run this application locally:
    - Clone this repository:
      ```bash
      git clone https://github.com/siddhant-pawar/streamlit_breast-cancer_prediction.git
      cd streamlit_breast-cancer_prediction
      ```
  2) Install dependencies:
      ```bash
      pip install -r requirements.txt
      ```
  3) Run the Streamlit app:
      ```bash
      streamlit run app.py
      ```
### Libraries Used:
  - streamlit: For building and displaying the web application.
  - pandas: For data manipulation and cleaning.
  - plotly: For creating interactive plots like the radar chart.
  - pygwalker: For generating HTML representation of the dataset.

### Functionality:
  - Data Cleaning: The data_cleaner function loads and preprocesses the breast cancer dataset.
  - Sidebar Slider: Use sliders in the sidebar to adjust input values for different cell nuclei measurements.
  - Radar Chart: Visualizes the selected measurements using a radar chart to compare mean, standard error, and worst values.
  - Prediction: Uses a machine learning model to predict whether the input values indicate a benign or malignant tumor.
  - Charts: Generates scatter plots to visualize relationships between selected features.

### what it does:
  - Data Cleaning (data_cleaner function):
    - Reads a CSV file containing breast cancer data.
    - Removes unnecessary columns (Unnamed: 32 and id).
    - Converts the diagnosis column (M for malignant and B for benign) into numerical values (1 for malignant and 0 for benign).

  - Sidebar Slider (add_sidebareslider function):
    - Creates a sidebar in the Streamlit app with sliders for various cell nuclei measurements (like radius, texture, perimeter, etc.).
    - Users can adjust these sliders to input different values for these measurements.

  - Data Scaling (makescaled_values function):
    - Takes the user-input values from the sidebar sliders.
    - Scales these values to be between 0 and 1 using the minimum and maximum values from the dataset.

  - Radar Chart (raderchart function):
    - Generates a radar chart using Plotly to visualize the scaled input values.
    - Compares mean, standard error, and worst values of the selected cell nuclei measurements.

  - Prediction (modelpredictor function):
    - Loads a pre-trained machine learning model and a scaler object using pickle files (model.pkl and scaler.pkl).
    - Uses the scaled input values to predict whether the tumor is benign or malignant.
    - Displays the prediction result ("Benign" or "Malignant") and the probability of each class.

  - Charts (reportchart function):
    - Creates scatter plots for the selected cell nuclei measurements grouped by mean, standard error, and worst values.
    - Each tab (Mean_values, Standard_error, worst, test) displays different scatter plots based on the selected features.

  - Pygwalker Chart (Pygchart function):
     - Generates an HTML representation of the cleaned dataset using pygwalker.
     - Embeds this HTML into the Streamlit app for interactive viewing.
