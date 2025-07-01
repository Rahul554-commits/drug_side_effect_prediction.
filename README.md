💊 Drug Side Effect Prediction with Explainable AI
PythonMachine LearningStreamlitSHAPLicense

🚀 Overview
A machine learning application that predicts potential drug side effects using Random Forest classification and SHAP explainability. Built with a user-friendly Streamlit web interface to support healthcare professionals in making informed decisions about drug safety.

🎯 Key Features
Machine Learning Prediction: Random Forest model with high accuracy
Explainable AI: SHAP integration for transparent decision-making
Interactive Web Interface: Real-time predictions via Streamlit
Batch Processing: Handle multiple patient records simultaneously
Clinical Insights: Feature importance analysis for medical interpretation
🏥 Problem Statement
Adverse drug reactions (ADRs) are a significant concern in healthcare, leading to:

Increased patient safety risks
Extended hospital stays
Higher healthcare costs
Delayed treatment decisions
This project aims to provide an AI-powered tool to predict potential side effects before they occur, enabling proactive healthcare management.

🔧 Technology Stack
Frontend: Streamlit (Interactive Web Application)
Backend: Python 3.8+
Machine Learning: Scikit-learn (Random Forest Classifier)
Explainability: SHAP (SHapley Additive exPlanations)
Data Processing: Pandas, NumPy
Development: PyCharm IDE
📁 Project Structure
drug_side_effect_prediction/
├── app/
│   ├── streamlit_app.py         # Main Streamlit application
│   └── predict.py               # Prediction logic and SHAP integration
├── models/
│   ├── random_forest_model.pkl  # Trained Random Forest model
│   ├── model_features.pkl       # Feature column names
│   └── X_train_shap.pkl         # SHAP explainer background data
├── data/
│   └── drug_data.csv            # Training dataset
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── LICENSE                      # MIT License
🚀 Quick Start
Prerequisites
Python 3.8 or higher
pip package manager
4GB+ RAM recommended
Installation
Clone the repository

git clone https://github.com/Rahul554-commits/drug_side_effect_prediction.git
cd drug_side_effect_prediction
Install dependencies

pip install -r requirements.txt
Run the application

streamlit run app/streamlit_app.py
Access the application

Open your web browser and navigate to http://localhost:8501
💻 Usage
Web Interface
Launch the Streamlit application
Enter patient information in the sidebar
Click "Predict Side Effects" to get results
View prediction results with SHAP explanations
Download results as CSV if needed
Programmatic Usage
from app.predict import DrugSideEffectPredictor

# Initialize predictor
predictor = DrugSideEffectPredictor()

# Prepare patient data
patient_data = {
    'age': 45,
    'blood_pressure': 140,
    'cholesterol': 220,
    'drug_name': 'Aspirin'
}

# Get prediction with explanation
prediction, explanation = predictor.predict_with_explanation(patient_data)
print(f"Prediction: {prediction}")
📊 Model Performance
| Metric | Score | |--------|-------| | Accuracy | 92.3% | | Precision | 91.7% | | Recall | 93.1% | | F1-Score | 92.4% |

Performance metrics based on test dataset validation

🧠 Explainable AI Features
SHAP Integration
Individual Predictions: Understand why the model made specific predictions
Feature Importance: Identify which patient characteristics influence outcomes
Waterfall Plots: Visualize how each feature contributes to the final prediction
Summary Plots: Global model behavior insights
Clinical Interpretability
Clear explanations for healthcare professionals
Feature contributions mapped to medical significance
Transparent decision-making process
Support for clinical validation
🛠️ Development
Setting up Development Environment
Clone the repository
Open in PyCharm IDE
Configure Python interpreter (3.8+)
Install dependencies via pip
Run tests to ensure everything works
Testing
# Run tests (if test suite is available)
python -m pytest tests/

# Run the application in development mode
streamlit run app/streamlit_app.py --logger.level=debug
🚀 Future Enhancements
[ ] Multi-drug interaction prediction
[ ] Integration with Electronic Health Records (EHR)
[ ] Mobile application development
[ ] Advanced visualization dashboards
[ ] RESTful API for third-party integrations
[ ] Real-time model updating capabilities
🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request
⚠️ Disclaimer
This tool is intended for research and educational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

📞 Contact
Shanigaramu Rahul
📧 Email: 228r1a1254.cmr@gmail.com
🔗 GitHub: @Rahul554-commits
🌐 Project: Drug Side Effect Prediction

<div align="center">
⭐ Star this repository if it helped you!

GitHub stars

</div>
