import streamlit as st
import modules  as mod

def main():
    st.set_page_config(
        page_title="Breast cancer Predictor",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    input_values = mod.add_sidebareslider()
    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar. ")

    col1, col2 = st.columns([4,3])
    with col1:
        getchart = mod.raderchart(input_values)
        if getchart is not None:
            st.plotly_chart(getchart)
        mod.reportchart(input_values)
    with col2:
        mod.modelpredictor(input_values)
    mod.Pygchart()

if __name__ == "__main__":
    main()

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 