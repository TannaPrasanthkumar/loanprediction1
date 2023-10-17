from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the machine learning model
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user inputs
        gender = request.form['gender']
        married = request.form['married']
        dependents = request.form['dependents']
        education = request.form['education']
        employed = request.form['employed']
        credit = float(request.form['credit'])
        area = request.form['area']
        ApplicantIncome = float(request.form['ApplicantIncome'])
        CoapplicantIncome = float(request.form['CoapplicantIncome'])
        LoanAmount = float(request.form['LoanAmount'])
        Loan_Amount_Term = float(request.form['Loan_Amount_Term'])

        # Convert categorical data to binary values
        male = 1 if gender == "Male" else 0
        married_yes = 1 if married == "Yes" else 0

        if dependents == '1':
            dependents = 1
        elif dependents == '2':
            dependents = 2
        elif dependents == "3+":
            dependents = 3
        else:
            dependents = 3

        not_graduate = 1 if education == "Not Graduate" else 0
        employed_yes = 1 if employed == "Yes" else 0

        if area == "Semiurban":
            area = 0
        elif area == "Urban":
            area = 1
        else:
            area = 2
        # Calculate logarithms
        ApplicantIncomeLog = np.log(ApplicantIncome)
        TotalIncomeLog = np.log(ApplicantIncome + CoapplicantIncome)
        LoanAmountLog = np.log(LoanAmount)
        Loan_Amount_TermLog = np.log(Loan_Amount_Term)

        # Make a prediction using the loaded model
        prediction = model.predict([[credit, ApplicantIncomeLog, LoanAmountLog, Loan_Amount_TermLog, TotalIncomeLog,
                                     male, married_yes, dependents, not_graduate,
                                     employed_yes, area]])

        # Map the prediction result to user-friendly text
        prediction_text = "Yes" if prediction[0] == "Y" else "No"

        return render_template("prediction.html", prediction_text=f"Loan status is {prediction_text}")

    return render_template("prediction.html")


if __name__ == "__main__":
    app.run(debug=True)
