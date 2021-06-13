from flask import Flask, request
from sklearn.ensemble import RandomForestClassifier

from flask_cors import CORS

import pickle

#from model.ANN import ANN

from logging.config import dictConfig

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

  
@app.route('/predict', methods=['POST'])
def predict():
  remaining_days = request.json['remaining_days']
  len_team = request.json['team']
  remaining_points = request.json['remaining_points']

  sk_model = open('randomForest', 'rb')
  clf = pickle.load(sk_model)
  sk_model.close()
  probabilities =  clf.predict_proba([[len_team,remaining_points,remaining_days]])

  app.logger.info("PROBABILITIES: %s", type(probabilities[0]))

  return {
    "probabilities" : probabilities[0].tolist()
  }
