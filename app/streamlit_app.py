import streamlit as st
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt
import numpy as np
from predict import make_prediction

st.set_page_config(page_title="Drug Side Effect Prediction", layout="wide")
st.title("💊 Enhancing Drug Side Effect Prediction with Explainable AI for Medical Health Applications")

# Debug settings (can be modified in code)
DEBUG_MODE = False
FORCE_HIGH_RISK = False
SHOW_PROBABILITIES = True


# Load model and setup SHAP
@st.cache_resource
def load_model_and_explainer():
    model = joblib.load(os.path.join("models", "random_forest_model.pkl"))
    expected_features = joblib.load("models/model_features.pkl")

    try:
        train_data = pd.read_csv("models/cleaned_drug_data.csv")
        features = ["age", "rating", "blood_pressure", "cholesterol", "symptom_severity", "drug_name"]
        train_encoded = pd.get_dummies(train_data[features], drop_first=True)

        # Clean and align data
        for col in train_encoded.columns:
            if train_encoded[col].dtype == 'object':
                train_encoded[col] = pd.to_numeric(train_encoded[col], errors='coerce')

        train_encoded = train_encoded.fillna(0).reindex(columns=expected_features, fill_value=0).astype('float64')
        background_data = train_encoded.sample(min(100, len(train_encoded)), random_state=42)
        explainer = shap.TreeExplainer(model, background_data)

        return model, explainer, expected_features
    except Exception as e:
        st.warning(f"Could not load training data: {e}")
        return model, shap.TreeExplainer(model), expected_features


model, explainer, feature_names = load_model_and_explainer()


@st.cache_data
def get_unique_drug_names():
    return [""] + sorted(["Aspirin", "Ibuprofen", "Paracetamol", "Metformin"])


def prepare_input_for_shap(input_df):
    """Prepare and encode input data"""
    try:
        expected_features = joblib.load("models/model_features.pkl")
        training_features = ["age", "rating", "blood_pressure", "cholesterol", "symptom_severity", "drug_name"]
        input_features = input_df[training_features].copy()

        # Ensure proper data types for numeric columns
        numeric_cols = ['age', 'rating', 'symptom_severity']
        for col in numeric_cols:
            if col in input_features.columns:
                input_features[col] = pd.to_numeric(input_features[col], errors='coerce')

        input_encoded = pd.get_dummies(input_features, drop_first=True)
        input_encoded.columns = input_encoded.columns.str.replace(' ', '_').str.replace('[^A-Za-z0-9_]', '', regex=True)

        for col in input_encoded.columns:
            if input_encoded[col].dtype == 'object':
                input_encoded[col] = pd.to_numeric(input_encoded[col], errors='coerce')

        return input_encoded.fillna(0).reindex(columns=expected_features, fill_value=0).astype('float64')
    except Exception as e:
        st.error(f"Error preparing input: {e}")
        return None


def calculate_risk_score(patient_data):
    """Calculate a manual risk score based on clinical factors"""
    risk_score = 0
    risk_factors = []

    age = patient_data['age']
    bp = patient_data['blood_pressure']
    chol = patient_data['cholesterol']
    rating = patient_data['rating']
    severity = patient_data['symptom_severity']
    drug = patient_data['drug_name']

    # Age factor (elderly patients have higher risk)
    if age >= 85:
        risk_score += 3
        risk_factors.append("Very elderly (≥85)")
    elif age >= 75:
        risk_score += 2
        risk_factors.append("Elderly (≥75)")
    elif age >= 65:
        risk_score += 1
        risk_factors.append("Senior (≥65)")

    # Cardiovascular risk factors
    if bp == 'high':
        risk_score += 2
        risk_factors.append("High blood pressure")
    if chol == 'high':
        risk_score += 2
        risk_factors.append("High cholesterol")

    # Drug tolerance and symptom severity
    if rating <= 2:
        risk_score += 2
        risk_factors.append("Poor drug tolerance")
    if severity >= 8:
        risk_score += 3
        risk_factors.append("Severe symptoms")
    elif severity >= 6:
        risk_score += 2
        risk_factors.append("Moderate-severe symptoms")

    # High-risk drug combinations for elderly
    if age >= 65 and drug in ['Aspirin', 'Ibuprofen']:
        risk_score += 1
        risk_factors.append("High-risk drug for elderly")

    return risk_score, risk_factors


def create_shap_visualizations(shap_values, input_encoded):
    """Create comprehensive SHAP visualizations"""
    try:
        # Flatten SHAP values if needed
        shap_vals = shap_values.flatten() if len(shap_values.shape) > 1 else shap_values
        shap_vals = shap_vals[:len(input_encoded.columns)]

        # Prepare feature data
        feature_data = pd.DataFrame({
            'Feature': input_encoded.columns,
            'SHAP_Value': shap_vals,
            'Feature_Value': input_encoded.iloc[0].values,
            'Abs_SHAP': np.abs(shap_vals)
        })

        top_features = feature_data.nlargest(10, 'Abs_SHAP')

        # Create visualizations
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # SHAP impact plot
        colors = ['red' if x > 0 else 'green' for x in top_features['SHAP_Value']]
        bars1 = ax1.barh(range(len(top_features)), top_features['SHAP_Value'], color=colors, alpha=0.7)
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels([f.replace('_', ' ').title() for f in top_features['Feature']], fontsize=10)
        ax1.set_xlabel('SHAP Value (Impact on Prediction)')
        ax1.set_title('Feature Impact (Red = Increases Risk, Green = Decreases Risk)')
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.3)
        ax1.grid(True, alpha=0.3)

        # Add value labels
        for bar, val in zip(bars1, top_features['SHAP_Value']):
            ax1.text(val + (0.01 if val >= 0 else -0.01), bar.get_y() + bar.get_height() / 2,
                     f'{val:.3f}', ha='left' if val >= 0 else 'right', va='center', fontsize=9)

        # Feature values plot
        bars2 = ax2.barh(range(len(top_features)), top_features['Feature_Value'], color='skyblue', alpha=0.7)
        ax2.set_yticks(range(len(top_features)))
        ax2.set_yticklabels([f.replace('_', ' ').title() for f in top_features['Feature']], fontsize=10)
        ax2.set_xlabel('Feature Value')
        ax2.set_title('Current Patient Feature Values')
        ax2.grid(True, alpha=0.3)

        for bar, val in zip(bars2, top_features['Feature_Value']):
            ax2.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{val:.2f}' if isinstance(val, float) else str(val),
                     ha='left', va='center', fontsize=9)

        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

        # Feature importance table
        st.subheader("🔍 Top Feature Impacts")
        display_df = top_features[['Feature', 'Feature_Value', 'SHAP_Value']].copy()
        display_df['Impact'] = display_df['SHAP_Value'].apply(lambda x: 'Increases Risk' if x > 0 else 'Decreases Risk')
        display_df['Feature'] = display_df['Feature'].str.replace('_', ' ').str.title()

        st.dataframe(display_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error creating visualizations: {e}")


# Input Section
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("📂 Upload Patient CSV File", type=["csv"])
    input_df = None

    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
    else:
        st.subheader("Enter Patient Information")

        age = st.number_input("Age", min_value=0, max_value=120, value=30)

        drug_name = st.selectbox("Drug Name", get_unique_drug_names(),
                                 index=get_unique_drug_names().index(st.session_state.get('preset_drug', ''))
                                 if st.session_state.get('preset_drug', '') in get_unique_drug_names() else 0,
                                 help="Please select a drug name from the dropdown")

        drug_name_valid = True
        if drug_name == "":
            st.error("❌ Invalid drug name - Please select a valid drug from the dropdown")
            drug_name_valid = False

        bp_options = ["low", "normal", "high"]
        bp = st.selectbox("Blood Pressure", bp_options,
                          index=bp_options.index(st.session_state.get('preset_bp', 'normal')))

        chol_options = ["low", "normal", "high"]
        chol = st.selectbox("Cholesterol", chol_options,
                            index=chol_options.index(st.session_state.get('preset_chol', 'normal')))

        rating = st.slider("Patient Drug Rating (1-5)", 1, 5,
                           st.session_state.get('preset_rating', 4))
        severity = st.slider("Symptom Severity (1-10)", 1, 10,
                             st.session_state.get('preset_severity', 5))

        if drug_name_valid:
            input_df = pd.DataFrame([{
                "age": age, "drug_name": drug_name, "blood_pressure": bp,
                "cholesterol": chol, "rating": rating, "symptom_severity": severity
            }])
        else:
            input_df = None

with col2:
    if input_df is not None:
        st.subheader("📋 Current Patient Profile")
        patient = input_df.iloc[0]

        # Calculate manual risk score
        risk_score, risk_factors = calculate_risk_score(patient)

        st.markdown(f"""
        - **Age**: {patient['age']} years
        - **Drug**: {patient['drug_name']}
        - **Blood Pressure**: {patient['blood_pressure']}
        - **Cholesterol**: {patient['cholesterol']}
        - **Rating**: {patient['rating']}/5
        - **Symptom Severity**: {patient['symptom_severity']}/10

        **Clinical Risk Score**: {risk_score}/15
        **Risk Factors**: {', '.join(risk_factors) if risk_factors else 'None identified'}
        """)

        # Show risk assessment
        if risk_score >= 6:
            st.error("🚨 **HIGH CLINICAL RISK DETECTED**")
        elif risk_score >= 4:
            st.warning("⚠️ **MODERATE RISK**")
        else:
            st.success("✅ **LOW RISK**")

# Prediction and Analysis
if input_df is not None:
    st.markdown("---")

    try:
        # Get manual risk assessment first
        patient = input_df.iloc[0]
        manual_risk_score, manual_risk_factors = calculate_risk_score(patient)

        # Make prediction
        prediction = make_prediction(model, input_df, debug=DEBUG_MODE)
        input_encoded = prepare_input_for_shap(input_df)

        if input_encoded is not None:
            probabilities = model.predict_proba(input_encoded)[0]
            prob_no_side_effect, prob_side_effect = probabilities[0], probabilities[1]

            # Override prediction if force high risk is enabled OR if manual risk is very high
            original_prediction = prediction[0]
            if FORCE_HIGH_RISK or manual_risk_score >= 7:
                prediction = [1]
                if DEBUG_MODE:
                    st.info(
                        f"Prediction overridden: Original={original_prediction}, Override=1, Manual Risk Score={manual_risk_score}")

            # Results display
            result_col1, result_col2 = st.columns([1, 1])

            with result_col1:
                st.subheader("🔍 Prediction Result")
                if prediction[0] == 1:
                    st.error("⚠️ **Side Effect Likely**")
                    risk_level, risk_color = "HIGH RISK", "red"
                else:
                    st.success("✅ **No Side Effect Predicted**")
                    risk_level, risk_color = "LOW RISK", "green"

                st.markdown(
                    f'<div style="padding: 10px; border-radius: 5px; border: 2px solid {risk_color}; text-align: center;"><h3 style="color: {risk_color}; margin: 0;">{risk_level}</h3></div>',
                    unsafe_allow_html=True)

            with result_col2:
                st.subheader("📊 Confidence & Probability")

                if SHOW_PROBABILITIES:
                    st.metric("No Side Effect", f"{prob_no_side_effect:.1%}")
                    st.metric("Side Effect Risk", f"{prob_side_effect:.1%}")
                    st.progress(float(max(prob_no_side_effect, prob_side_effect)))

                st.metric("Clinical Risk Score", f"{manual_risk_score}/15")
                st.metric("Risk Factors", len(manual_risk_factors))

            # Debug information
            if DEBUG_MODE:
                st.info(f"**Original Model Prediction:** {original_prediction}")
                st.info(f"**Final Prediction:** {prediction[0]}")
                st.info(f"**Manual Risk Score:** {manual_risk_score}/15")
                st.info(
                    f"**Probabilities:** No Side Effect: {prob_no_side_effect:.3f}, Side Effect: {prob_side_effect:.3f}")

            # SHAP Analysis
            with st.spinner("Calculating AI explanations..."):
                try:
                    shap_values = explainer.shap_values(input_encoded)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]
                    if len(shap_values.shape) > 1:
                        shap_values = shap_values[0]

                    st.markdown("---")
                    st.header("🧠 AI Model Explanation (SHAP Analysis)")
                    create_shap_visualizations(shap_values, input_encoded)

                except Exception as e:
                    st.warning(f"SHAP analysis unavailable: {e}")
                    if DEBUG_MODE:
                        import traceback

                        st.error(traceback.format_exc())

            # Clinical Recommendations
            st.markdown("---")
            st.subheader("⚕️ Clinical Recommendations")

            if prediction[0] == 1:
                st.markdown(f"""
                ### 🚨 High Risk Patient - Actions Required:

                **Risk Factors Identified:** {', '.join(manual_risk_factors)}

                - 🏥 **Schedule immediate medical consultation**
                - 📊 **Conduct comprehensive health assessment**
                - 💊 **Consider alternative medications**
                - 🔍 **Implement close monitoring**
                - 📋 **Document symptoms and reactions**

                **Monitor for:** Kidney issues, cardiovascular complications, neurological symptoms, GI distress, blood disorders

                **Special Considerations for Age {patient['age']}:**
                - Enhanced monitoring due to age-related physiological changes
                - Consider dose adjustments
                - Regular kidney and liver function tests
                """)
            else:
                st.markdown("""
                ### ✅ Low Risk Patient - Standard Care:
                - 📅 **Follow routine monitoring**
                - 💊 **Continue prescribed dosage**
                - 📊 **Regular follow-ups**
                - 📋 **Report unusual symptoms**
                - 🔄 **Maintain preventive care**
                """)

    except Exception as e:
        st.error(f"❌ Analysis failed: {e}")
        st.info("Please check your input data and try again.")
        if DEBUG_MODE:
            import traceback

            st.error(traceback.format_exc())

elif 'drug_name_valid' in locals() and not drug_name_valid:
    st.warning("⚠️ Please select a valid drug name to proceed with the analysis.")

