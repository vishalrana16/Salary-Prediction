from flask import Flask, render_template, request,abort
import numpy as np
from keras import backend as K
from keras.utils import np_utils
from keras.models import load_model
import keras.models
import math
import pickle
scalerfile = 'sc.sav'
scaler = pickle.load(open(scalerfile, 'rb'))

from load import * 

global model, graph

model, graph = init()




app = Flask(__name__)

app.config['DEBUG'] = True
      
@app.route('/index')
def write_message():
	return render_template('index.html')

WorkClass = {'Private':2,'Local-gov':1,'Self-emp-not-inc':4,
'Federal-gov':0
,'State-gov':5,'Self-emp-inc':3,'Without-pay':6,} 
MaritialStatus = {'Married':1, 'Un-Married':0}
Occupation = {'Craft-repair':2,'Prof-specialty':9,'Exec-managerial':3 ,     
'Adm-clerical':0,    
'Sales':11  ,   
'Other-service':7 ,  
'Machine-op-inspct':6 , 
'Transport-moving':13 ,
'Handlers-cleaners':5 , 
'Farming-fishing':4 ,
'Tech-support' :12 ,
'Protective-serv': 10 ,
'Priv-house-serv':8 ,
'Armed-Forces': 1  }
Relationship = {
'Husband':0,          
'Not-in-family':1 ,    
'Own-child':3,        
'Unmarried':4,         
'Wife':5,               
'Other-relative':2
}
Race = {
'White':4,                 
'Black':2,               
'Asian-Pac-Islander': 1,    
'Amer-Indian-Eskimo': 0 ,    
'Other':3          
} 
Gender = {'Male':1, 'Female':0}   
education_number = {
'HS-grad': 8 ,      
'Some-college':9,    
'Bachelors':12,       
'Masters': 13,         
'Assoc-voc': 10,      
'11th': 6,            
'Assoc-acdm':11,       
'10th':5,             
'7th-8th':3,           
'Prof-school':14,       
'9th':4,               
'12th':7,              
'Doctorate': 15,        
'5th-6th':2,           
'1st-4th': 1,          
'Preschool':0   }                                    

        
        

@app.route('/predict', methods = ['GET', 'POST'])
def write():
    if request.method == 'POST':
        age = request.form["age"]
        work_class = WorkClass[request.form["workClass"]]
        fnlwgt = request.form["fnlwgt"]
        maritial_status =MaritialStatus[request.form["MaritialStatus"]]
        occupation = Occupation[request.form["Occupation"]]
        relationship = Relationship[request.form["Relationship"]]
        race =Race[request.form["Race"]]
        gender = Gender[request.form["Gender"]]
        captial_gain = request.form["CapitalGain"]
        captial_loss = request.form["CapitalLoss"]
        hours_per_week = request.form["HoursperWeek"]
        edu_num = education_number[request.form["education"]]
        name = request.form["name"]
    else:
         age = 0
         work_class = 0
         fnlwgt = 0
         maritial_status = 0
         occupation = 0
         relationship = 0
         race = 0
         gender = 0
         captial_gain = 0
         captial_loss = 0
         hours_per_week = 0
         edu_num = 0
         name = '____'
         #age_hours = ''
         
         
    c = int(age)
    d = int(hours_per_week)  
    age_hours = c*d
    #print(age_hours)
    
    final = [age,  
         work_class,
         fnlwgt, 
         maritial_status,  
         occupation,  
         relationship, 
         race, 
         gender, 
         captial_gain,  
         captial_loss, 
         hours_per_week, 
         edu_num, 
         age_hours]
    final = np.array(final)
    final = final.reshape(1,13)
    final = scaler.transform(final)
    
    with graph.as_default():
        out = model.predict(final) 
        print(out)
        res = out[0][0]
        res = math.floor(res*100)
        print(res)    
    
    return render_template('predict.html', val = res, name = name)


if __name__ == "__main__":
    app.run(debug = True,use_reloader = False)
