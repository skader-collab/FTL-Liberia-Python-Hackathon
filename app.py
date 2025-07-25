import streamlit as st
import pandas as pd
import pickle

# --- Load Model ---
@st.cache_resource
def load_model():
    with open("lasso_best_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# --- App Title & Description ---
st.set_page_config(page_title="Healthcare Shortfall Predictor", layout="wide")
st.title("🏥 Healthcare Workforce Shortfall Predictor")
st.markdown("""
This app predicts **Doctors Shortfall** based on healthcare infrastructure data.  
Model used: **Lasso Regression** (Best RMSE: 57.45)  
""")

# --- Sidebar Inputs ---
st.sidebar.header("Enter Healthcare Data")
year = st.sidebar.selectbox("Year", [2005, 2019])
subcenters = st.sidebar.number_input("SubCenters", min_value=0)
phcs = st.sidebar.number_input("PHCs", min_value=0)
chcs = st.sidebar.number_input("CHCs", min_value=0)
anm = st.sidebar.number_input("ANM/Health_Worker_Female", min_value=0)
doctors = st.sidebar.number_input("Doctors", min_value=0)
specialists = st.sidebar.number_input("Specialists", min_value=0)
radiographers = st.sidebar.number_input("Radiographers", min_value=0)
pharmacists = st.sidebar.number_input("Pharmacists", min_value=0)
lab_tech = st.sidebar.number_input("LabTechnicians", min_value=0)
nursing = st.sidebar.number_input("NursingStaff", min_value=0)
state = st.sidebar.text_input("State/UT")

# --- Prepare Input Data ---
input_data = pd.DataFrame({
    'Year': [year],
    'SubCenters': [subcenters],
    'PHCs': [phcs],
    'CHCs': [chcs],
    'ANM/Health_Worker_Female': [anm],
    'Doctors': [doctors],
    'Specialists': [specialists],
    'Radiographers': [radiographers],
    'Pharmacists': [pharmacists],
    'LabTechnicians': [lab_tech],
    'NursingStaff': [nursing],
    'State/UT': [state],
    # Feature Engineering
    'Doctors_per_PHC': [doctors / (phcs + 1)],
    'Specialists_per_CHC': [specialists / (chcs + 1)],
    'SubCenters_Year_Interaction': [subcenters * year]
})

# --- Predict ---
if st.sidebar.button("Predict Shortfall"):
    prediction = model.predict(input_data)[0]
    
    # Display Result
    st.subheader("Prediction Result")
    st.metric(label="Predicted Doctors Shortfall", value=f"{prediction:.2f}")
    
    # Risk Alert
    if prediction > 500:
        st.error("⚠ High Doctor Shortage Risk!")
    elif prediction > 200:
        st.warning("Moderate Shortage")
    else:
        st.success("Low Shortage Risk")
    
    # Comparison Chart
    st.bar_chart(pd.DataFrame({
        "Metric": ["Current Doctors", "Predicted Shortfall"],
        "Value": [doctors, prediction]
    }).set_index("Metric"))
