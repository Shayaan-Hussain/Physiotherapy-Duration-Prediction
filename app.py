from flask import Flask,render_template,request
import pickle as pkl
import numpy as np
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('base.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        bodypart = request.form['bodypart']
        diagnosis = request.form['diagnosis']
        duration = request.form['duration']
        intensity = request.form['intensity']
        approach = request.form['approach']
        features = np.array([diagnosisMap[diagnosis], durationMap[duration], float(intensity), approachMap[approach]])
        features = features.reshape(1, -1)
        pred = int(model.predict(features)[0])
        prediction = f'{pred} days of physiotherapy is recommended'
    return render_template('result.html', 
                            bodypart=bodypart,
                            intensity=intensity,
                            diagnosis=diagnosis,
                            approach=approach,
                            duration=duration,
                            pred=prediction)

if __name__ == '__main__':
    model = pkl.load(open('modelDecisionTree.pkl', 'rb'))
    diagnosisMap = {
        'Patella Fracture' : 0,
        'Cervical Radiculopathy' : 1,
        'Lumbar Radiculopathy' : 2,
        'Frozen Shoulder' : 3,
        'Jennis Elbow' : 4,
        'Osteoarthritis' : 5,
        'Plantar Fasciitis' : 6
    }
    durationMap = {
        'Acute' : 0,
        'Subacute' : 1,
        'Chronic' : 2
    }
    approachMap = {
        'Manual' : 0,
        'Mechanical' : 1,
        'Manual and Mechanical' : 2
    }
    app.run(debug=True, host='0.0.0.0')