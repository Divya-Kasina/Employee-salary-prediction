import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💰",
    layout="wide"
)

# Load models and feature info
@st.cache_resource
def load_models():
    try:
        model = joblib.load('salary_model.pkl')
        preprocessor = joblib.load('salary_preprocessor.pkl')
        feature_info = joblib.load('feature_info.pkl')
        return model, preprocessor, feature_info
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found! Please run the training script first.")
        st.error(f"Missing file: {e}")
        st.stop()

model, preprocessor, feature_info = load_models()

# Title and description
st.title("💰 Salary Prediction App")
st.markdown("### Predict whether annual income exceeds $50K based on census data")

# Create two columns for inputs
col1, col2 = st.columns(2)

# Sidebar with example inputs
st.sidebar.header("📋 Example Test Cases")

if st.sidebar.button("Load High-Income Example"):
    st.session_state.update({
        'age': 45, 'workclass': 'Private', 'fnlwgt': 192776,
        'education': 'Bachelors', 'educational_num': 13,
        'marital_status': 'Married-civ-spouse', 'occupation': 'Exec-managerial',
        'relationship': 'Husband', 'race': 'White', 'sex': 'Male',
        'capital_gain': 5000, 'capital_loss': 0, 'hours_per_week': 60,
        'native_country': 'United-States'
    })

if st.sidebar.button("Load Low-Income Example"):
    st.session_state.update({
        'age': 22, 'workclass': 'Private', 'fnlwgt': 150000,
        'education': 'HS-grad', 'educational_num': 9,
        'marital_status': 'Never-married', 'occupation': 'Other-service',
        'relationship': 'Not-in-family', 'race': 'White', 'sex': 'Female',
        'capital_gain': 0, 'capital_loss': 0, 'hours_per_week': 25,
        'native_country': 'United-States'
    })

# Input form
with st.form("prediction_form"):
    # Left column inputs
    with col1:
        st.subheader("👤 Personal Information")
        age = st.slider("Age", 17, 90, 
                       value=st.session_state.get('age', 30))
        
        sex = st.radio("Sex", ['Male', 'Female'], 
                      index=0 if st.session_state.get('sex', 'Male') == 'Male' else 1)
        
        race = st.selectbox("Race", feature_info['all_categories']['race'],
                           index=feature_info['all_categories']['race'].index(
                               st.session_state.get('race', 'White')))
        
        native_country = st.selectbox("Native Country", 
                                    feature_info['all_categories']['native_country'],
                                    index=feature_info['all_categories']['native_country'].index(
                                        st.session_state.get('native_country', 'United-States')))
        
        st.subheader("💼 Work Information")
        workclass = st.selectbox("Work Class", feature_info['all_categories']['workclass'],
                                index=feature_info['all_categories']['workclass'].index(
                                    st.session_state.get('workclass', 'Private')))
        
        occupation = st.selectbox("Occupation", feature_info['all_categories']['occupation'],
                                 index=feature_info['all_categories']['occupation'].index(
                                     st.session_state.get('occupation', 'Exec-managerial')))
        
        hours_per_week = st.slider("Hours per Week", 1, 99,
                                  value=st.session_state.get('hours_per_week', 40))
    
    # Right column inputs
    with col2:
        st.subheader("🎓 Education & Family")
        education = st.selectbox("Education Level", feature_info['all_categories']['education'],
                               index=feature_info['all_categories']['education'].index(
                                   st.session_state.get('education', 'Bachelors')))
        
        educational_num = st.slider("Education Years", 1, 16,
                                   value=st.session_state.get('educational_num', 13),
                                   help="1=Preschool, 9=HS-grad, 13=Bachelors, 16=Doctorate")
        
        marital_status = st.selectbox("Marital Status", 
                                    feature_info['all_categories']['marital_status'],
                                    index=feature_info['all_categories']['marital_status'].index(
                                        st.session_state.get('marital_status', 'Married-civ-spouse')))
        
        relationship = st.selectbox("Relationship", feature_info['all_categories']['relationship'],
                                  index=feature_info['all_categories']['relationship'].index(
                                      st.session_state.get('relationship', 'Husband')))
        
        st.subheader("💵 Financial Information")
        fnlwgt = st.number_input("Final Weight", 10000, 1500000,
                                value=st.session_state.get('fnlwgt', 200000),
                                help="Census sampling weight")
        
        capital_gain = st.number_input("Capital Gain", 0, 100000,
                                     value=st.session_state.get('capital_gain', 0),
                                     help="Investment income")
        
        capital_loss = st.number_input("Capital Loss", 0, 5000,
                                     value=st.session_state.get('capital_loss', 0),
                                     help="Investment losses")
    
    # Submit button
    submitted = st.form_submit_button("🔮 Predict Income", use_container_width=True)

# Prediction logic
if submitted:
    # Create input dataframe
    input_data = {
        'age': age,
        'workclass': workclass,
        'fnlwgt': fnlwgt,
        'education': education,
        'educational_num': educational_num,
        'marital_status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'sex': sex,
        'capital_gain': capital_gain,
        'capital_loss': capital_loss,
        'hours_per_week': hours_per_week,
        'native_country': native_country
    }
    
    try:
        # Create DataFrame and make prediction
        input_df = pd.DataFrame([input_data])
        input_processed = preprocessor.transform(input_df)
        prediction = model.predict(input_processed)[0]
        probabilities = model.predict_proba(input_processed)[0]
        
        # Display results
        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.success("✅ **Income > $50K**")
            else:
                st.error("❌ **Income ≤ $50K**")
        
        with col2:
            confidence = max(probabilities) * 100
            st.metric("Confidence", f"{confidence:.1f}%")
        
        with col3:
            st.metric("Prediction", ">50K" if prediction == 1 else "≤50K")
        
        # Probability breakdown
        st.markdown("### 📊 Probability Breakdown")
        prob_col1, prob_col2 = st.columns(2)
        
        with prob_col1:
            st.metric("≤50K Probability", f"{probabilities[0]:.1%}")
        
        with prob_col2:
            st.metric(">50K Probability", f"{probabilities[1]:.1%}")
        
        # Show input summary
        with st.expander("📋 Input Summary"):
            st.json(input_data)
            
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        st.error("Please check that all inputs are valid and try again.")

# Information section
st.markdown("---")
st.markdown("### ℹ️ About This Model")
st.info("""
This model predicts whether a person's annual income exceeds $50K based on demographic and work-related features.
It's trained on the Adult Census Income dataset using a Random Forest classifier.

**Key Factors for High Income:**
- Higher education levels (Bachelor's, Master's, Doctorate)
- Professional/Executive occupations  
- Married status (especially married-civ-spouse)
- Full-time work hours (40+ hours/week)
- Prime working age (30-55 years)
- Capital gains from investments
""")