import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_diabetes_data(n_samples=5000):
    # 1. Generate Base Data (Demographics & Vitals)
    data = {
        'Age': np.random.randint(20, 90, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Urban_Rural': np.random.choice(['Urban', 'Rural'], n_samples),
        # Normal distribution for BMI, centered at 28 (slightly overweight average)
        'BMI': np.round(np.random.normal(28, 6, n_samples).clip(15, 60), 1),
        'Waist_Hip_Ratio': np.round(np.random.normal(0.9, 0.1, n_samples).clip(0.6, 1.2), 2),
        'Heart_Rate': np.random.randint(60, 100, n_samples),
        'Hypertension': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        'Cholesterol_Level': np.random.choice(['High', 'Normal', 'Borderline'], n_samples, p=[0.2, 0.5, 0.3]),
        'Physical_Activity': np.random.choice(['Low', 'Moderate', 'High'], n_samples, p=[0.4, 0.4, 0.2]),
        'Diet_Type': np.random.choice(['Veg', 'Non-Veg'], n_samples),
        'Smoking_Status': np.random.choice(['Smoker', 'Non-Smoker'], n_samples, p=[0.2, 0.8]),
        'Alcohol_Intake': np.random.choice(['None', 'Moderate', 'High'], n_samples, p=[0.5, 0.4, 0.1]),
        'Stress_Level': np.random.choice(['Low', 'Moderate', 'High'], n_samples, p=[0.3, 0.4, 0.3]),
        'Sleep_Quality': np.random.choice(['Good', 'Poor'], n_samples, p=[0.6, 0.4]),
        'Family_History': np.random.choice(['Yes', 'No'], n_samples, p=[0.2, 0.8]),
        'Medication_For_Chronic_Conditions': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        'Thyroid_Condition': np.random.choice(['Yes', 'No'], n_samples, p=[0.1, 0.9])
    }

    df = pd.DataFrame(data)

    # 2. Handle Gender Specific Logics
    # Initialize as 'No'
    df['Polycystic_Ovary_Syndrome'] = 'No'
    df['Gestational_Diabetes'] = 'No'

    # Assign conditions only to females
    female_mask = df['Gender'] == 'Female'
    df.loc[female_mask, 'Polycystic_Ovary_Syndrome'] = np.random.choice(['Yes', 'No'], female_mask.sum(), p=[0.15, 0.85])
    df.loc[female_mask, 'Gestational_Diabetes'] = np.random.choice(['Yes', 'No'], female_mask.sum(), p=[0.08, 0.92])

    # 3. Implement Risk Score Logic (Log-Odds) - SHARPENED for >85% Accuracy
    # Start with a lower base to ensure "healthy" people stay healthy
    logit = -6.0 

    # Age Risk: Stronger curve
    logit += (df['Age'] - 30) * 0.05
    logit += (df['Age'] > 45).astype(int) * 2.0 
    logit += (df['Age'] > 60).astype(int) * 1.0

    # BMI Risk: Stronger impact for Obesity
    logit += (df['BMI'] - 25) * 0.15
    logit += (df['BMI'] > 30).astype(int) * 2.0
    logit += (df['BMI'] > 35).astype(int) * 1.5

    # Medical History Factors (Strong predictors)
    logit += (df['Family_History'] == 'Yes').astype(int) * 2.0
    logit += (df['Hypertension'] == 'Yes').astype(int) * 1.8
    logit += (df['Cholesterol_Level'] == 'High').astype(int) * 1.5
    logit += (df['Thyroid_Condition'] == 'Yes').astype(int) * 0.8
    
    # Lifestyle Factors (Compounding)
    logit += (df['Physical_Activity'] == 'Low').astype(int) * 1.2
    logit += (df['Stress_Level'] == 'High').astype(int) * 1.0
    logit += (df['Sleep_Quality'] == 'Poor').astype(int) * 0.8
    logit += (df['Smoking_Status'] == 'Smoker').astype(int) * 1.0
    logit += (df['Alcohol_Intake'] == 'High').astype(int) * 0.8
    
    # Gender Specific Risks
    logit += (df['Polycystic_Ovary_Syndrome'] == 'Yes').astype(int) * 2.0
    logit += (df['Gestational_Diabetes'] == 'Yes').astype(int) * 2.5

    # SYNERGISTIC INTERACTIONS (The "Death Spirals")
    # 1. Metabolic Syndrome: Obesity + BP + Cholesterol
    metabolic_mask = (df['BMI'] > 30) & (df['Hypertension'] == 'Yes') & (df['Cholesterol_Level'] == 'High')
    logit[metabolic_mask] += 4.0

    # 2. Senior + Unhealthy: Age > 55 + High Risk Factors
    bad_lifestyle_mask = (df['Physical_Activity'] == 'Low') & (df['Smoking_Status'] == 'Smoker')
    logit[(df['Age'] > 55) & bad_lifestyle_mask] += 3.0

    # 4. Convert Logit to Probability (Sigmoid Function)
    probabilities = 1 / (1 + np.exp(-logit))

    # 5. Assign Target based on Probability (Simulates Realistic Noise)
    # To reach >90% accuracy, we reduce noise for clear-cut cases.
    # If prob > 0.85, force Yes. If prob < 0.15, force No.
    # Middle range remains stochastic.
    
    rand_vals = np.random.rand(n_samples)
    diabetes_status_binary = rand_vals < probabilities
    
    # Overrides for 90% accuracy target (Noise Reduction)
    # Widening the deterministic window slightly to ensure we hit >90%
    diabetes_status_binary[probabilities > 0.80] = True
    diabetes_status_binary[probabilities < 0.20] = False
    
    df['Diabetes_Status'] = np.where(diabetes_status_binary, 'Yes', 'No')

    return df

# Run generation
df_synthetic = generate_synthetic_diabetes_data()

# Save to CSV
output_file = 'diabetes_synthetic_realistic.csv'
df_synthetic.to_csv(output_file, index=False)

print(f"Successfully generated {len(df_synthetic)} patient records to {output_file}.")
print(df_synthetic['Diabetes_Status'].value_counts())
