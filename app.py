#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import joblib
import base64
import random

import plotly.express as px 
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import dash
from dash import Dash, dcc, html, Input, Output, State
from jupyter_dash import JupyterDash
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate


# In[2]:


image_filename =  'VZLogo1.jpg'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())


# In[3]:


app = dash.Dash(external_stylesheets=[dbc.themes.JOURNAL, dbc.icons.FONT_AWESOME])

app.title = "Health Intelligence"
server = app.server


# In[4]:


app.layout = dbc.Container(
    [
        ## Header        
        html.Div([
            html.Div(
            html.Img(src='data:image/jpg;base64,{}'.format(encoded_image.decode()), height = 150)),
            html.Div([    
                html.Br(),  
                html.Label(['Machine Learning-based Diabetic Detection System'], 
                           style={'font-weight': 'bold', "font-family": 'Arial', 'color': '#5e34eb',
                                  "font-size": 55,"text-align": "center"}),
            ]),
        ],style={'display':'flex'}),        
        
        # Problem Statements
        html.Br(),
        html.Div([
            dbc.Label("Problem Statement", style={'font-weight': 'bold', "font-family": 'Arial', 'color': '#53a66f',
                                  "font-size": 40,"text-align": "center"}),
            html.Li("The objective of this project is to construct different machine learning models for predicting diabetes."),            
            html.Li("The dataset contains information of 60,000 individuals data with 17 features such as \
                    Age, Glucose Level, Blood pressure, Insulin, BMI, etc.")
            
        ]),       
        
        html.Hr(),
        html.Div([
        html.Label(['Feature Input'], style={'font-weight': 'bold', "font-family": 'Arial', 'color': '#5e34eb',
                                  "font-size": 30,"text-align": "center"}), 
        html.P("Adjust the features to predict!")
        ]),
        # Features
        html.Br(),
        html.Br(),
        html.Div([
            dbc.Button("Age", color="success", size = 'lg'),
            dbc.Input(id="id-fea1", placeholder="year", type="text"),  
            dbc.Button("Height", color="success", size = 'lg'),
            dbc.Input(id="id-fea2", placeholder="cm", type="text"),  
            dbc.Button("Weight", color="success", size = 'lg'),  
            dbc.Input(id="id-fea3", placeholder="kg", type="text",className="me-md-2"),  
        ], style = {'display': 'flex'}, className="d-grid gap-2 d-md-flex justify-content-md-end"), 
        html.Br(),
        html.Div([
            html.Div(dbc.Button("Blood Pressure (Top)", color="success", size = 'lg'),style ={'width':'25%'}),
            html.Div(dbc.Input(id="id-fea4", placeholder="mmHg", type="text"), style ={'width':'25%'}),
            html.Div(dbc.Button("Blood Pressure (Bottom)", color="success", size = 'lg'),
                     style ={"margin-left": "55px",'width':'25%'}),
            html.Div(dbc.Input(id="id-fea5", placeholder="mmHg", type="text"),  style ={'width':'25%'}),            
        ], style = {'display': 'flex'},),       
        
        
        html.Br(),
        html.Div([
            html.Div(dbc.Button("Check Your Diabetic Status!", color="primary", size = 'lg', id = 'id-predict'),
                     style = {'width':'50%', "font-size": 75,'height':'50px'},className="d-grid gap-2"),
            html.Div(dbc.Alert("The chance of having diabetic is... ", id="result-text"), style = {"margin-left": "55px",'width':'50%'}),    
        ], style = {'display': 'flex'}), 
            
    ])


# In[5]:


@app.callback(    
    Output('result-text', 'children'),
    [Input('id-fea1', 'value'),
     Input('id-fea2', 'value'),
     Input('id-fea3', 'value'),
     Input('id-fea4', 'value'),
     Input('id-fea5', 'value'),
     Input('id-predict', 'n_clicks'),
    ]
)
def load_data(*args):    
        
    if(len(args)<6):
        raise PreventUpdate
    else:
        loaded_model = joblib.load('LR_best_classification_model.pkl') 
        result = 'Model loaded .... ' 
        fea_list = []
        for arg in args[0:5]:
            if arg is None:
                result = 'Enter the value .. '
            else:
                fea_list.append(int(arg))
        
        if(args[-1]): 
            if(loaded_model.predict([fea_list])==1):
                result = "You have high chance of getting Diabetic"
            else:
                result = "You have low chance of getting Diabetic"
        
        return result 
    


# In[ ]:


if __name__ == '__main__':
    port = 5000 + random.randint(0, 999)    
    url = "http://127.0.0.1:{0}".format(port)    
    app.run_server(use_reloader=False, debug=True, port=port)


# In[ ]:




