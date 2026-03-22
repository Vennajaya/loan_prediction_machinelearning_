from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        int(request.form['Age']),
        int(request.form['Income']),
        int(request.form['Credit_Score']),
        int(request.form['Loan_Amount']),
        int(request.form['Loan_Term']),
        int(request.form['Employment_Status'])
    ]

    prediction = model.predict([features])
    result = "Approved" if prediction[0] == 1 else "Not Approved"

    # 👇 Return to same homepage
    return render_template('index.html', prediction_text=f'Loan is {result}')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)