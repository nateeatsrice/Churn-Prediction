import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

#data wrangling and format standardization
df = pd.read_csv('Telco-Customer-Churn.csv')

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

df.columns = df.columns.str.lower().str.replace(' ', '_')

string_columns = list(df.dtypes[df.dtypes == 'object'].index)

for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')

df.churn = (df.churn == 'yes').astype(int)

#split full training and test set
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)

df_train_full = df_train_full.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
# will use cross validation later so no need to create training and validation sets

categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
               'phoneservice', 'multiplelines', 'internetservice',
               'onlinesecurity', 'onlinebackup', 'deviceprotection',
               'techsupport', 'streamingtv', 'streamingmovies',
               'contract', 'paperlessbilling', 'paymentmethod']
numerical = ['tenure', 'monthlycharges', 'totalcharges']

def train(df, y, C):
    # Dict vectorizer inputs a dictionary
    cat = df[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    dv.fit(cat)

    X = dv.transform(cat)

    model = LogisticRegression(solver='liblinear', C=C)
    model.fit(X, y)

    return dv, model

def predict(df, dv, model):
    cat = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(cat)

    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

# using KFold Validation to see best model in using training set tested on validation set
kfold = KFold(n_splits=10, shuffle=True, random_state=1)

# the C parameter is the inverse of the regularization strength
# small value of C (e.g., 0.01 or 0.1) means a large regularization strength,
# leading to a simpler model that is less prone to overfitting
for C in [0.001, 0.01, 0.1, 0.5, 1, 10, 50]:
    # aucs will be averaged to get the average auc per model
    aucs = []

    for train_idx, val_idx in kfold.split(df_train_full):
        df_train = df_train_full.iloc[train_idx]
        df_val = df_train_full.iloc[val_idx]

        y_train = df_train.churn.values
        y_val = df_val.churn.values

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        aucs.append(auc)

    print('C=%s, auc = %0.3f ± %0.3f' % (C, np.mean(aucs), np.std(aucs)))

# seeing that the best model was with C= 0.1 we now train on full training set (training+val) and compare with test
y_train = df_train_full.churn.values
y_test = df_test.churn.values
C = 0.1
# the model and dv below are ultimitly the file we want to export
dv, model = train(df_train_full, y_train, C=C)

y_pred = predict(df_test, dv, model)
auc = roc_auc_score(y_test, y_pred)
print('auc = %.3f' % auc)

#set threshold to 0.5
cm = confusion_matrix(y_test, y_pred >= 0.5)
print(cm)

# model inference
customer = {
    'customerid': '8879-zkjof',
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'no',
    'dependents': 'no',
    'tenure': 41,
    'phoneservice': 'yes',
    'multiplelines': 'no',
    'internetservice': 'dsl',
    'onlinesecurity': 'yes',
    'onlinebackup': 'no',
    'deviceprotection': 'yes',
    'techsupport': 'yes',
    'streamingtv': 'yes',
    'streamingmovies': 'yes',
    'contract': 'one_year',
    'paperlessbilling': 'yes',
    'paymentmethod': 'bank_transfer_(automatic)',
    'monthlycharges': 79.85,
    'totalcharges': 3320.75
}

#inference on the single example above
df = pd.DataFrame([customer])
y_pred = predict(df, dv, model)
y_pred[0]

# make a function which automates the cell above
def predict_single(customer, dv, model):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]

# not likely to churn
predict_single(customer, dv, model)

import pickle

output_file = f"logregmodel_C={C}.bin"
output_file

# # 'wb' means we want to write to the file and it is going to be binary
# f_out = open(output_file,'wb')
# # pickle.dump saves the output, since dv is required we save with model as tuple
# pickle.dump((dv,model),f_out)
# #make sure to close, will use "with" function which does this automatically
# f_out.close()

# same as code above, './churn-model.bin' names the file and directory
with open('./churn-model.bin', 'wb') as f_out:
    pickle.dump((dv, model), f_out)

import requests
url = 'http://localhost:9696/predict'
response = requests.post(url, json=customer)
result = response.json()
result

