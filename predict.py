import pickle
from flask import Flask
from flask import request
from flask import jsonify

def predict_single(customer, dv, model):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]

model_file = 'churn-model.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# create a Flask app name "churn"
app = Flask('churn')

# adding decorator which adds some additional functionality
# route is to specify which address will live in the ping address
# GET is a HTTP method to request data from a resource
@app.route('/predict', methods=['POST'])

def predict():
    #request.get_json() will return the body of the request as python dict
    customer = request.get_json()

    y_pred = predict_single(customer,dv,model)
    #you should decide the threshhold not end user
    churn = y_pred >= 0.5
    
    result = {
        "churn_probability": float(y_pred),
        "churn": bool(churn)
    }
    #also returning a JSON file from py_dict
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0',port=9698)