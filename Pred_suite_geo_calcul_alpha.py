# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 11:11:12 2019

@author: Hakim Razzak
"""

# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import matplotlib.pyplot as plt

# Read recipe inputs
dataset_data_attr_activity_filtered2_6months = dataiku.Dataset("dataset_data_attr_activity_filtered2_6months")
dataset_df = dataset_data_attr_activity_filtered2_6months.get_dataframe()


df = dataset_df[dataset_df['dim_country'] == 'US']
df = df[df['dim_app'].str.contains('Alizoha')]
df = df.rename(columns=lambda x: x.replace('_sum', ''))
df = df.drop('avg_dau', 1)
df = df.sort_values(by = 'week', ascending=False)
df


df_train = df.values[0:8,]
df_train = pd.DataFrame(df_train)
df_train


def weighted_average(data, col_install, col_retention):
    i=0
    val_eff=[]
    m=[]

    for j in data.columns[col_retention:]:
        val_eff=[]

        for i in range(len(data[col_install])):
            val_eff.append(float(data[col_install][i])*data[j][i])

        m.append(sum(val_eff)/sum(data[col_install]))

    return(m)
    
def calcul_alpha():
    alpha=[]
    i=0

    for i in range(len(m)-1):
        alpha.append(float(m[i+1])/m[i])

    return(alpha)
    
def model(col_retention, row_pred):
    val_col=[]
    Un=[]

    for j in df_train.columns[col_retention:]:
        val_col.append(float(df_train[j][row_pred]))
    for i in range(len(alpha)):
        Un.append(float(val_col[i])*alpha[i])

    return Un

def graph(col_retention,row_pred,Un):
    val_col=[]
    for j in df_train.columns[col_retention:]:
        val_col.append(float(df_train[j][row_pred]))

    plt.plot(df_train.columns[col_retention+1:],val_col[:-1])
    plt.plot(df_train.columns[col_retention+1:],Un)

    plt.show()

    return val_col

    
m = weighted_average(df_train, 4, 5)

alpha = calcul_alpha()
print (alpha)

Un = model(5,0)
print(Un)

val = graph(5,0,Un)

df = pd.DataFrame({'Jour-J':val[:-1], 'Alpha':alpha, 'Predj+1':Un})
df




