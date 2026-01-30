// Handle Gender Change to show/hide female specific fields
document.getElementById('Gender').addEventListener('change', function () {
    const femaleFields = document.querySelectorAll('.female-only');
    if (this.value === 'Female') {
        femaleFields.forEach(el => el.classList.remove('d-none'));
    } else {
        femaleFields.forEach(el => el.classList.add('d-none'));
        // Reset values if hidden
        document.getElementById('Polycystic_Ovary_Syndrome').value = 'No';
        document.getElementById('Gestational_Diabetes').value = 'No';
    }
});

document.getElementById('predictionForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    // 1. Collect Data (Auto-collect based on IDs to be efficient)
    const fields = [
        'Age', 'Gender', 'Urban_Rural', 'BMI', 'Waist_Hip_Ratio', 'Heart_Rate',
        'Hypertension', 'Cholesterol_Level', 'Physical_Activity', 'Diet_Type',
        'Sleep_Quality', 'Smoking_Status', 'Alcohol_Intake', 'Stress_Level',
        'Family_History', 'Medication_For_Chronic_Conditions', 'Thyroid_Condition',
        'Polycystic_Ovary_Syndrome', 'Gestational_Diabetes'
    ];

    const formData = {};
    fields.forEach(id => {
        formData[id] = document.getElementById(id).value;
    });

    // 2. Visual Feedback
    const btn = document.querySelector('button[type="submit"]');
    const originalText = btn.innerText;
    btn.innerText = 'Analyzing full profile...';
    btn.disabled = true;

    try {
        // 3. Send Request
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });

        const result = await response.json();

        if (response.ok) {
            updateResultUI(result.prediction, result.probability);
        } else {
            console.error('Server Error:', result);
            alert('Error: ' + result.error);
        }

    } catch (error) {
        console.error('Network Error:', error);
        alert('An unexpected error occurred. Please check console.');
    } finally {
        btn.innerText = originalText;
        btn.disabled = false;
    }
});

function updateResultUI(prediction, probability) {
    const resultCard = document.getElementById('resultCard');
    const predText = document.getElementById('predictionText');
    const riskDesc = document.getElementById('riskDescription');
    const probText = document.getElementById('probabilityText');
    const probBar = document.getElementById('probabilityBar');

    resultCard.classList.remove('d-none');

    predText.innerText = prediction;
    probText.innerText = probability;
    probBar.style.width = probability;

    if (prediction === 'High Risk') {
        predText.className = 'display-3 fw-bold mb-2 text-danger';
        riskDesc.innerText = 'High probability of diabetes detected. Please consult a doctor.';
        probBar.className = 'progress-bar progress-bar-striped progress-bar-animated bg-danger';
    } else {
        predText.className = 'display-3 fw-bold mb-2 text-success';
        riskDesc.innerText = 'Your risk profile appears healthy. Keep up the good habits!';
        probBar.className = 'progress-bar progress-bar-striped progress-bar-animated bg-success';
    }

    // Smooth scroll to result
    resultCard.scrollIntoView({ behavior: 'smooth' });
}
