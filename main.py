from fastapi import FastAPI, Request, Form, File, UploadFile
from pydantic import BaseModel
from typing import List
import pickle
import pandas as pd
from fastapi.responses import FileResponse
import re
import numpy as np

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


def load_obj(filepath):
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model

def delete_unit(x):
    try:
        return float(x.split()[0])
    except:
        return None

def delete_torque_unit(x):
    try:
        splitted = re.split('@|at', x)
        torque, max_torque_rpm = splitted[0],splitted[1]
        torque_re = re.findall("([0-9]+[,.]*[0-9]*)",torque)
        max_torque_rpm_re = re.findall("([0-9]+[,.]*[0-9]*)",max_torque_rpm)
        torque_typed = float(torque_re[0])
        max_torque_rpm_typed = np.mean([*map(lambda x: float(x.replace(',','')),max_torque_rpm_re)])
        
        return [torque_typed,max_torque_rpm_typed]
    except:
        return [None,None]

def predict(X):
    model = load_obj('model.pickle')
    scaler = load_obj('scaler.pickle')
    enc = load_obj('ohe.pickle')
    train = pd.read_csv('cars_train.csv')

    X = X.copy()    
    X[['mileage','engine','max_power']] = X[['mileage','engine','max_power']].applymap(delete_unit)
    X[['torque','max_torque_rpm']] = pd.DataFrame(X['torque'].apply(delete_torque_unit).to_list(), index= X.index) 
    train[['mileage','engine','max_power']] = train[['mileage','engine','max_power']].applymap(delete_unit)
    train[['torque','max_torque_rpm']] = pd.DataFrame(train['torque'].apply(delete_torque_unit).to_list(), index= train.index) 
    X = X.fillna(train.median())
    X[['engine','seats']] = X[['engine','seats']].applymap(int)
    X =  X.drop(columns = ['name','selling_price'])
    X[enc.get_feature_names_out(['fuel','seller_type','transmission','owner','seats'])] = pd.DataFrame(enc.transform(X[['fuel','seller_type','transmission','owner','seats']]).toarray())
    X = X.drop(columns = ['fuel','seller_type','transmission','owner','seats'])
    X['year_square'] = X['year'] * X['year']
    X['km_driven_square'] =  X['km_driven'] * X['km_driven']
    X_scaled = pd.DataFrame(scaler.transform(X))
    return model.predict(X_scaled)




@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict_item")
async def predict_item(item: Item) -> float:
    item = pd.DataFrame([x.dict() for x in [item]])
    pred = predict(item)
    return list(pred)[0]


@app.post("/predict_items",response_class=FileResponse)
async def predict_items(file: UploadFile):
    items = pd.read_csv(file.file)
    file.file.close()
    pred = predict(items)
    items['pred'] = pred
    items.to_csv('pred.csv')
    return 'pred.csv'