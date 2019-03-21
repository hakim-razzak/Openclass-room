# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 10:38:04 2019

@author: Hakim Razzak
"""
from dataiku import pandasutils as pdu
import dataiku

import pandas as pd, numpy as np

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score



dataset_df = pd.read_csv('C:/Users/Hakim Razzak/Documents/Analyse_Retention_FMgames/dataset_data_attr_activity_filtered2_6months.csv', sep=',')
dataset_df.head()


def selection_cohorte(nom_application, pays, suppression_dau, sort_by, a):
    
    global df
    
    df = dataset_df[dataset_df['dim_country'] == pays]
    df = df.rename(columns=lambda x: x.replace('_sum', ''))
    df = df[df['dim_app'].str.contains(nom_application)]
    
    if suppression_dau == 1:
        df = df.drop('avg_dau', 1)
    
    df = df.sort_values(by = sort_by, ascending=a)
    
    return df


selection_cohorte('Shozila', 'US', 1, 'week', False)



def Prediction_Day30max(nb_jour_pred,first_col_retention,last_col_retention):
    
    list_quality_pred = []
    i = 0
    j = 0
    
    for j in range(nb_jour_pred):

        if i == 0:
        
            df_features = df.values[:,first_col_retention:last_col_retention]
            df_features = pd.DataFrame(df_features)
            df_target = df.values[:,last_col_retention]
            df_target = pd.DataFrame(df_target)

            df_prepared = pd.concat([df_target, df_features], axis=1)
            df_prepared.head()
            
        else:
            
            df_features = df_predict
            df_target = df.values[:,last_col_retention+j]
            df_target = pd.DataFrame(df_target)
            
            df_prepared = pd.concat([df_target, df_features], axis=1)
            df_prepared.head()


        data = df_prepared.as_matrix()
        X = data[:,1:]
        Y = data[:,0]

        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y,test_size = 0.3, random_state=0)
        print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


        # Create linear regression object
        regr = linear_model.LinearRegression()

        # Train the model using the training sets
        regr.fit(X_train, Y_train)

        # Make predictions using the testing set
        y_pred = regr.predict(X_test)


        # The coefficients
        print('Coefficients:', regr.coef_)
        # The mean squared error
        print("Mean squared error: %.2f"% mean_squared_error(Y_test, y_pred))
        # Explained variance score: 1 is perfect prediction
        r2 = r2_score(Y_test, y_pred)
        print('Variance score: %.2f' % r2)

        print(y_pred)


        pred = regr.predict(X)

        X_df = pd.DataFrame(X)
        pred_df = pd.DataFrame(pred)

        df_predict = pd.concat([X_df, pred_df], axis=1, ignore_index=True)

        df_predict
        
        i =+ 1
        
        list_quality_pred.append(r2)
        
    return df_predict, list_quality_pred
    


df_predict, list_quality_pred = Prediction_Day30max(21,5,14)

df_predict
list_quality_pred
 
