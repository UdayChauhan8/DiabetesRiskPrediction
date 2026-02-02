# Diabetes Risk Prediction

A machine learning project that predicts an individual’s diabetes risk using demographic, lifestyle, and medical history features. The pipeline includes data cleaning, feature engineering (risk scores and interaction terms), categorical encoding, class imbalance handling, and model evaluation using ROC and Precision–Recall metrics.

The system is designed as a risk stratification tool, not a diagnostic replacement, providing early identification of high-risk individuals to support preventive healthcare and data-driven decision-making.

 **Live Demo:** [https://diabetesriskprediction-xhfq.onrender.com](https://diabetesriskprediction-xhfq.onrender.com)


## Features
- **Instant Prediction:** Returns "High Risk" or "Low Risk" with a probability score.
- **ML Model:** Random Forest Classifier trained on synthetic medical data.
- **Privacy:** No data is stored; predictions happen in real-time.

## Local Setup

1. **Clone & Install**
   ```bash
   git clone <your-repo-url>
   cd UpdatedDiabetes
   pip install -r requirements.txt
   ```

2. **Run Application**
   ```bash
   python app.py
   ```
   Access at `http://localhost:5000`

## Repository Structure
- `app.py`: Flask backend
- `trainmodel.py`: Model training script
- `diabetes_model.pkl`: Trained model
