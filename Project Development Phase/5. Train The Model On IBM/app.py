# Import Libraries
import pandas as pd
import numpy as np
from flask import Flask, render_template, Response, request
import pickle
from sklearn.preprocessing import LabelEncoder
import requests

API_KEY = "hKEAjvDOQF9fNf7tINrykXASVwLWFdkulNDdbJaLgtNe"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

app = Flask(__name__)#initiate flask app

def load_model(file='crvp.sav'):#load the saved model
	return pickle.load(open(file, 'rb'))

@app.route('/')
def index():#main page
	return render_template('car.html')

@app.route('/predict_page')
def predict_page():#predicting page
	return render_template('value.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
	reg_year = int(request.args.get('regyear'))
	powerps = float(request.args.get('powerps'))
	kms= float(request.args.get('kms'))
	reg_month = int(request.args.get('regmonth'))

	gearbox = request.args.get('geartype')
	damage = request.args.get('damage')
	model = request.args.get('model')
	brand = request.args.get('brand')
	fuel_type = request.args.get('fuelType')
	veh_type = request.args.get('vehicletype')

	new_row = {'yearOfRegistration':reg_year, 'powerPS':powerps, 'kilometer':kms,
				'monthOfRegistration':reg_month, 'gearbox':gearbox,
				'notRepairedDamage':damage,
				'model':model, 'brand':brand, 'fuelType':fuel_type,
				'vehicleType':veh_type}

	print(new_row)

	new_df = pd.DataFrame(columns=['vehicleType','yearOfRegistration','gearbox',
		'powerPS','model','kilometer','monthOfRegistration','fuelType',
		'brand','notRepairedDamage'])
	new_df = new_df.append(new_row, ignore_index=True)
	labels = ['gearbox','notRepairedDamage','model','brand','fuelType','vehicleType']
	mapper = {}

	for i in labels:
		mapper[i] = LabelEncoder()
		mapper[i].classes = np.load(str('classes'+i+'.npy'), allow_pickle=True)
		transform = mapper[i].fit_transform(new_df[i])
		new_df.loc[:,i+'_labels'] = pd.Series(transform, index=new_df.index)
	labeled = new_df[['yearOfRegistration','powerPS','kilometer','monthOfRegistration'] + [x+'_labels' for x in labels]]

	X = labeled.values.tolist()
	print('\n\n', X)
	#predict = reg_model.predict(X)

	#NOTE: manually define and pass the array(s) of values to be scored in the next line
	payload_scoring = {'input_data': [{'fields': ["yearOfRegistration", "powerPS", "kilometer", "monthOfRegistration","gearbox_labels", "notRepairedDamage_labels", "model_labels","brand_labels", "fuelType_labels", "vehicleType_labels"], 'values': X}]}
	print(payload_scoring)
	response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/4bbb629f-8821-451b-89a9-4902644d9f86/predictions?version=2022-11-18', json=payload_scoring,headers={'Authorization': 'Bearer ' + mltoken})
	predictions= response_scoring.json()
	print(response_scoring.json())
	predict = predictions['predictions'][0]['values'][0][0]
	print("Final prediction :",predict)

	return render_template('predict.html',predict=predict)

if __name__=='__main__':
	reg_model = load_model()#load the saved model
	app.run(host='localhost', debug=True, threaded=False)