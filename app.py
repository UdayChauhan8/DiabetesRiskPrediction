from flask import Flask, render_template, request, jsonify
import joblib
import os
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load Model and Preprocessor
model_path = os.path.join(os.path.dirname(__file__), 'diabetes_model.pkl')
preprocessor_path = os.path.join(os.path.dirname(__file__), 'preprocessor.pkl')

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # 1. Create DataFrame from input
        # Note: We must ensure handling of optional gender fields
        input_data = {
            'Age': int(data['Age']),
            'Gender': data['Gender'],
            'Urban_Rural': data['Urban_Rural'],
            'BMI': float(data['BMI']),
            'Waist_Hip_Ratio': float(data['Waist_Hip_Ratio']),
            'Heart_Rate': int(data['Heart_Rate']),
            'Hypertension': data['Hypertension'],
            'Cholesterol_Level': data['Cholesterol_Level'],
            'Physical_Activity': data['Physical_Activity'],
            'Diet_Type': data['Diet_Type'],
            'Smoking_Status': data['Smoking_Status'],
            'Alcohol_Intake': data.get('Alcohol_Intake', 'Unknown'), # Default if missing
            'Stress_Level': data['Stress_Level'],
            'Sleep_Quality': data['Sleep_Quality'],
            'Family_History': data['Family_History'],
            'Medication_For_Chronic_Conditions': data['Medication_For_Chronic_Conditions'],
            'Thyroid_Condition': data['Thyroid_Condition'],
            'Polycystic_Ovary_Syndrome': data.get('Polycystic_Ovary_Syndrome', 'No'),
            'Gestational_Diabetes': data.get('Gestational_Diabetes', 'No')
        }
        
        df = pd.DataFrame([input_data])
        
        # 2. FEATURE ENGINEERING (Must match training logic EXACTLY)
        # --------------------------------------------------------
        
        # BMI Category
        df["BMI_Category"] = pd.cut(
            df["BMI"], 
            bins=[0, 18.5, 24.9, 29.9, 100], 
            labels=["Underweight", "Normal", "Overweight", "Obese"]
        )
        
        # Age Group
        df["Age_Group"] = pd.cut(
            df["Age"], 
            bins=[0, 30, 50, 120], 
            labels=["Young", "Adult", "Senior"]
        )
        
        # Interaction
        df["BMI_Age_Interaction"] = df["BMI"] * df["Age"]
        
        # Risk Factor Count
        df["Risk_Factor_Count"] = (
            (df["Hypertension"] == "Yes").astype(int) +
            (df["Family_History"] == "Yes").astype(int) +
            (df["Smoking_Status"] == "Smoker").astype(int) +
            (df["Stress_Level"] == "High").astype(int) +
            (df["Cholesterol_Level"] == "High").astype(int) +
            (df["Physical_Activity"] == "Low").astype(int) +
            (df["Sleep_Quality"] == "Poor").astype(int) +
            (df["Age"] > 50).astype(int) + 
            (df["BMI"] > 30).astype(int)
        )

        # 5. Metabolic Syndrome (Synergistic Effect)
        # Obesity + Hypertension + High Cholesterol
        df["Metabolic_Syndrome_Score"] = (
            (df["BMI"] > 30).astype(int) + 
            (df["Hypertension"] == "Yes").astype(int) + 
            (df["Cholesterol_Level"] == "High").astype(int)
        )

        # 6. High Risk Lifestyle
        # Smoking + Alcohol + Sedentary
        df["High_Risk_Lifestyle"] = (
            (df["Smoking_Status"] == "Smoker").astype(int) + 
            (df["Alcohol_Intake"] == "High").astype(int) + 
            (df["Physical_Activity"] == "Low").astype(int)
        )

        
        # 3. Handling Types for Preprocessor
        # The preprocessor expects 'object' columns to be strings for OneHotEncoder
        categorical_cols = [
            'Gender', 'Urban_Rural', 'Hypertension', 'Cholesterol_Level', 
            'Physical_Activity', 'Diet_Type', 'Smoking_Status', 'Alcohol_Intake', 
            'Stress_Level', 'Sleep_Quality', 'Family_History', 
            'Medication_For_Chronic_Conditions', 'Thyroid_Condition', 
            'Polycystic_Ovary_Syndrome', 'Gestational_Diabetes', 
            'BMI_Category', 'Age_Group'
        ]
        
        # Ensure they are strings (sometimes single row DF infers wrong type)
        for col in categorical_cols:
            df[col] = df[col].astype(str)
            
        # 4. Transform Data
        X_encoded = preprocessor.transform(df)
        
        # 5. Predict
        prediction_prob = model.predict_proba(X_encoded)[0][1] # Probability of Class 1 (Yes)
        prediction_class = "High Risk" if prediction_prob > 0.5 else "Low Risk"
        probability_pct = round(prediction_prob * 100, 1)

        return jsonify({
            'prediction': prediction_class,
            'probability': f"{probability_pct}%"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
