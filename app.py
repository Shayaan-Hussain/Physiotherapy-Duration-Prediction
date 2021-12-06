from flask import Flask,render_template,request, redirect
import pickle as pkl
import numpy as np
app = Flask(__name__)
model = pkl.load(open('modelDecisionTree.pkl', 'rb'))
diagnosisMap = {
    'Patella Fracture' : 0,
    'Cervical Radiculopathy' : 1,
    'Lumbar Radiculopathy' : 2,
    'Frozen Shoulder' : 3,
    'Tennis Elbow' : 4,
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

@app.route('/')
def home():
    return render_template('base.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        global diagnosisMap, durationMap, approachMap
        bodypart = request.form['bodypart']
        diagnosis = request.form['diagnosis']
        duration = request.form['duration']
        intensity = request.form['intensity']
        approach = request.form['approach']
        features = np.array([diagnosisMap[diagnosis], durationMap[duration], float(intensity), approachMap[approach]])
        features = features.reshape(1, -1)
        pred = int(model.predict(features)[0])
        prediction = f'{pred} days of physiotherapy is recommended'
    else:
        return redirect('/')
    return render_template('result.html', 
                            bodypart=bodypart,
                            intensity=intensity,
                            diagnosis=diagnosis,
                            approach=approach,
                            duration=duration,
                            pred=prediction)

@app.errorhandler(404)
def invalid_routes(e):
    return render_template('notFound.html')

if __name__ == '__main__':
    app.run()