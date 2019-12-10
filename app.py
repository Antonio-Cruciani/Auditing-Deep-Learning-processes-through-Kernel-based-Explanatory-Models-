from src.main.python_code.neurlanetwork.neuralnetwork import create_model,get_ny_proj
from src.main.python_code.explaining.build_explanations import explenation_qc

import numpy as np

from flask import Flask, render_template, url_for, request,Markup,request, redirect,  session, abort, flash
import os
import requests
import flask
import json

# You have to change this link with the url of your Nystrom Projector url.
nystrom_web_server_address = 'http://localhost:8080/nystrom_ws/rest/qc/getcvectorfromstring/'


app = flask.Flask(__name__)
app.secret_key = os.urandom(12)

password ="user"
username = "passowrd"


k_parameter = 3
nn_session = None
mlp = None

# Loading landmarks (id, class, string)
list_of_landmarks = []

def init():
    global session,mlp
    # Caricamento del modello pre-trained
    mlp,nn_session = create_model()
    return(mlp,nn_session)

@app.route("/")
@app.route("/home")
def home():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        global mlp,nn_session,FLAGS
        if not mlp:
            mlp,nn_session=init()
        return render_template('index.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        result = request.form['squestion']
    elif request.method ==  'GET':
        result = request.args.get('squestion', '')
        if not result or len(result):
            return home()
    else:
        return home()

    # Proiezione al web server nystrom
    params = {'text': result}
    ny_obj = requests.post(nystrom_web_server_address,data=params)
    # seleziono il c vector dalla risposta di nystrom e lo "impacchetto" per poterlo inviare alla rete neurale
    data = json.loads(ny_obj.text)
    splitted_data = data["vector"].split(' ')
    float_cleaned_data_lista = [float(i) for i in splitted_data]
    float_cleaned_data = float_cleaned_data_lista[0:]
    input_data = np.asarray(float_cleaned_data)
    projected_vector = np.array([input_data])

    question = result
    is_answer = True

    prediction = get_ny_proj(mlp,nn_session,projected_vector,question)
    qc_explanations = explenation_qc()
    positive_singleton = qc_explanations.build_explanation_positive_singleton(prediction)
    negative_singleton = qc_explanations.build_explanation_negative_singleton(prediction)
    positive_conjunctive =qc_explanations.build_explanation_positive_conjunctive(prediction,3)
    negative_conjunctive = qc_explanations.build_explanation_negative_conjunctive(prediction,3)
    positive_contrastive = qc_explanations.build_explanation_positive_contrastive(prediction)
    negative_contrastive = qc_explanations.build_explanation_negative_contrastive((prediction))


    jresponse = {
        'Question': Markup("<input type=\"text \" id=\"question\" value=\""+ question+"\" name=\"squestion\" class=\"form-control\" required>"),
        'PositiveSingleton':positive_singleton,
        'NegativeSingleton':negative_singleton,
        'PositiveContrastive':positive_contrastive,
        'NegativeContrastive':negative_contrastive,
        'PositiveConjunctive':positive_conjunctive,
        'NegativeConjunctive':negative_conjunctive,

    }
    return render_template("index.html", result=jresponse,Script = is_answer)

@app.route('/login', methods=['POST'])
def do_admin_login():
    if request.form['password'] == password and request.form['username'] == username:
        session['logged_in'] = True
    else:
        flash('wrong password!')
    return home()

@app.route("/logout")
def logout():
    session['logged_in'] = False
    flask.flash("You are now logged out.")
    return home()


if __name__ == "__main__":
    print(("* Loading model and Flask starting server... please wait until server has fully started"))
    app.secret_key = os.urandom(12)


    app.run(debug=True, host='0.0.0.0', port=4000)

