import streamlit as st
import pandas as pd 
import plotly.graph_objects as go
import pygwalker as pyg
import pickle
import numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from scipy import stats

def data_cleaner():
    # import and clean data
    data = pd.read_csv("G:\\siddhant\\mlds\\breastcancer\\eda\\data.csv")  
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
    return data


def add_sidebareslider():
    st.sidebar.header("Cell Nuclei Measurements")
    data = data_cleaner()
      
    slider_labels = [
            ("Radius (mean)", "radius_mean"),
            ("Texture (mean)", "texture_mean"),
            ("Perimeter (mean)", "perimeter_mean"),
            ("Area (mean)", "area_mean"),
            ("Smoothness (mean)", "smoothness_mean"),
            ("Compactness (mean)", "compactness_mean"),
            ("Concavity (mean)", "concavity_mean"),
            ("Concave points (mean)", "concave points_mean"),
            ("Symmetry (mean)", "symmetry_mean"),
            ("Fractal dimension (mean)", "fractal_dimension_mean"),
            ("Radius (se)", "radius_se"),
            ("Texture (se)", "texture_se"),
            ("Perimeter (se)", "perimeter_se"),
            ("Area (se)", "area_se"),
            ("Smoothness (se)", "smoothness_se"),
            ("Compactness (se)", "compactness_se"),
            ("Concavity (se)", "concavity_se"),
            ("Concave points (se)", "concave points_se"),
            ("Symmetry (se)", "symmetry_se"),
            ("Fractal dimension (se)", "fractal_dimension_se"),
            ("Radius (worst)", "radius_worst"),
            ("Texture (worst)", "texture_worst"),
            ("Perimeter (worst)", "perimeter_worst"),
            ("Area (worst)", "area_worst"),
            ("Smoothness (worst)", "smoothness_worst"),
            ("Compactness (worst)", "compactness_worst"),
            ("Concavity (worst)", "concavity_worst"),
            ("Concave points (worst)", "concave points_worst"),
            ("Symmetry (worst)", "symmetry_worst"),
            ("Fractal dimension (worst)", "fractal_dimension_worst"),
        ]
    #if "defaultvalues" not in st.session_state:
    #    st.session_state["defaultvalues"] = ( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    #    st.write(st.session_state.defaultvalues)
    #    for lable, column in slider_labels:
    #        st.session_state.defaultvalues = st.sidebar.slider("defaultvalues", min_value=float(0), max_value=float(data[column].max()),value=float(data[column].mean()))
    #        st.write(st.session_state.defaultvalues)
    #input_values=st.session_state.defaultvalues
    input_values={}
    for label, column in slider_labels:
        input_values[column] = st.sidebar.slider(label, min_value=float(0), max_value=float(data[column].max()),value=float(data[column].mean()))

    return input_values


# this function makes scaled values 
def makescaled_values(input_values):
    data = data_cleaner()
    X = data.drop(['diagnosis'], axis=1)
    scaled_features = {}
    for key, value in input_values.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_features[key] = scaled_value
    
    return scaled_features

def raderchart(input_values):
    st.title("Rader Chart")
    input_data = makescaled_values(input_values)
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
          input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
          input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
          input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Values'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'],
          input_data['area_se'], input_data['smoothness_se'], input_data['compactness_se'],
          input_data['concavity_se'], input_data['concave points_se'], input_data['symmetry_se'],
          input_data['fractal_dimension_se']],
        theta=categories,
        fill='toself',
        name='Standard error '
    ))
    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
          input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
          input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
          input_data['fractal_dimension_worst']],
        theta=categories,
        fill='toself',
        name='Worst'
    ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
    showlegend=True
    )
    return fig


  

def modelpredictor(input_data):
    # Load pickle file for predicting
    model = pickle.load(open("../model/model.pkl","rb"))
    scaler = pickle.load(open("../model/scaler.pkl","rb"))
    input_array = np.array(list(input_data.values())).reshape(1,-1)
    input_array_scaled = scaler.transform(input_array)
    prediction = model.predict(input_array_scaled)
    col0, col3 = st.columns([1,2])
    with col0:
        st.header("Result")
        if prediction == 0:
            
            st.success("Begin Signs!!")
        else:
            st.error("It's looks Suspicious!")
    with col3:
        st.header("Probability of Having")
        st.write("Probability of Just begin:", model.predict_proba(input_array_scaled)[0][0])
        st.write("Probability of Already  Started:", model.predict_proba(input_array_scaled)[0][1])

def reportchart(input_values):
    st.title("Chart")  
    input_data = makescaled_values(input_values)
    Mean_values ,Standard_error, Worst, test =  st.tabs(["Mean_values", "Standard_error", "worst","test"])
    with Mean_values:
        val=[input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']]
        features=[input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']]  
        plt.scatter(features, val)
        plt.xlabel('Features')
        plt.ylabel('Coefficient')
        st.pyplot(plt)

    with Standard_error:
        val=[input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'],
          input_data['area_se'], input_data['smoothness_se'], input_data['compactness_se'],
          input_data['concavity_se'], input_data['concave points_se'], input_data['symmetry_se'],
          input_data['fractal_dimension_se']]
        features=[input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'],
          input_data['area_se'], input_data['smoothness_se'], input_data['compactness_se'],
          input_data['concavity_se'], input_data['concave points_se'], input_data['symmetry_se'],
          input_data['fractal_dimension_se']]
        plt.scatter(features, val)
        plt.xlabel('Features')
        plt.ylabel('Coefficient')
        st.pyplot(plt)
    with Worst:
        val=[input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
          input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
          input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
          input_data['fractal_dimension_worst']]
        features=[input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
          input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
          input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
          input_data['fractal_dimension_worst']]
        plt.scatter(features, val)
        plt.xlabel('Features')
        plt.ylabel('Coefficient')
        st.pyplot(plt)
    with test:
        val=[input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
          input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
          input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
          input_data['fractal_dimension_worst']]
        features=[input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
          input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
          input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
          input_data['fractal_dimension_worst']]
        plt.legend() 
        plt.scatter(val, features, label= "stars", color= "green",  marker= "*", s=30) 
        st.pyplot(plt)

def Pygchart():
    st.title("Pygwalker Chart Gen")  
    df = data_cleaner()
    pyg_html = pyg.to_html(df)
    # Embed the HTML into the Streamlit app
    components.html(pyg_html, height=1000, scrolling=True)