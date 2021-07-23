from flask import Flask, redirect, url_for, render_template, request
import numpy as np
import pickle
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler

# from flask_bootstrap import Bootstrap 

app = Flask(__name__)
# Bootstrap(app)

@app.route("/")
def home():
    return render_template("index.html") 

@app.route('/',methods=["POST"])

def analyze():
    if request.method == "POST":
        SPTN = request.form['SPTN']
        depth = request.form['depth']
        d30 = request.form['d30']
        d60 = request.form['d60']
        silt = request.form['silt']

    # Clean the data by convert from unicode to float 
    sample_data = [SPTN,depth, d30, d60, silt]
    clean_data = [float(i) for i in sample_data]


    ex1 = np.array(clean_data).reshape(1,-1)
    
    # load the model from disk
    filename = 'finalized_model_SVM.sav'
    filename2 = "scaler_SVM.sav"
 
    
    loaded_model = pickle.load(open(filename, 'rb'))
    scaler = pickle.load(open(filename2, "rb"))
    
    ex1_scaled = scaler.transform(ex1)
    result = loaded_model.predict(ex1_scaled)
    result = np.round(result,2)
    

    return render_template("index.html", SPTN = SPTN, 
                          depth = depth, d30 = d30,
                          d60 = d60, silt = silt,
                          result = result,
                          )

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug= True)
