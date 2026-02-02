# Diabetes Risk Prediction

🚀 **Live Demo:** [https://diabetesriskprediction-xhfq.onrender.com](https://diabetesriskprediction-xhfq.onrender.com)

A machine learning-powered web application that estimates diabetes risk based on health indicators.

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
