from flask import Flask, request, render_template
import model 

app = Flask(__name__)

model_path = 'spam_classifier.joblib'
spam_classifier = model.load_model(model_path)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email_text']
    
    prediction = model.predict(spam_classifier, email_text)
    
    result = 'Spam' if prediction == 1 else 'Not Spam'
    
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)