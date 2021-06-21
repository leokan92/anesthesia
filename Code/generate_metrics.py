# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 16:25:46 2020

@author: leona
"""


import vitalfilepy
from vitalfilepy import VitalFile
from vitalfilepy import VITALBINARY
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import re
import pandas as pd
from scipy import special
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from keras.callbacks import ModelCheckpoint
import sys


#Generate statistics for the Patients covariates
########################################################
#Test Information:
########################################################
path = os.getcwd()
path = join(path,'test')

cases = [f for f in listdir(path) if isfile(join(path, f))] 
cases_num = []
regex = re.compile(r'\d+')
for filename in cases:      
    cases_num.append(int(regex.findall(filename)[0]))
    
cases_num
age_test = []
sex_test = []
weight_test = []
height_test = []
dur_test = []
for case in cases_num:
    path = os.getcwd()
    df = pd.read_csv(os.path.join(path,'cases_all_info.csv'),sep = ';')
    df = df.loc[df['caseid'] == case]
    #input_len = df['Len'][0]
    age_test.append(float(df['age'].values[0]))
    sex_test.append(float(df['sex'].values[0]))
    weight_test.append(float(df['weight'].values[0]))
    height_test.append(float(df['height'].values[0]))
    dur_test.append(float(df['casedur'].values[0]))
print('========================================')
print('Test results for the cases')
print('========================================')
print('\n')    
print('Test mean age: ',np.mean(np.asarray(age_test)))
print('Test std age: ',np.std(np.asarray(age_test)))
print('\n')
print('Test mean sex: ',np.mean(np.asarray(sex_test)))
print('Test std sex: ',np.std(np.asarray(sex_test)))
print('\n')
print('Test mean weight: ',np.mean(np.asarray(weight_test)))
print('Test std weight: ',np.std(np.asarray(weight_test)))
print('\n')
print('Test mean height: ',np.mean(np.asarray(height_test)))
print('Test std height: ',np.std(np.asarray(height_test)))
print('\n')
print('Test mean dur: ',np.mean(np.asarray(dur_test)))
print('Test std dur: ',np.std(np.asarray(dur_test)))
print('\n')
########################################################
#Train Information:
########################################################

path = os.getcwd()
path = join(path,'train')

cases = [f for f in listdir(path) if isfile(join(path, f))] 
cases_num = []
for filename in cases:      
    cases_num.append(int(regex.findall(filename)[0]))
    
cases_num
age_train = []
sex_train = []
weight_train = []
height_train = []
dur_train = []
for case in cases_num:
    path = os.getcwd()
    df = pd.read_csv(os.path.join(path,'cases_all_info.csv'),sep = ';')
    df = df.loc[df['caseid'] == case]
    #input_len = df['Len'][0]
    age_train.append(float(df['age'].values[0]))
    sex_train.append(float(df['sex'].values[0]))
    weight_train.append(float(df['weight'].values[0]))
    height_train.append(float(df['height'].values[0]))
    dur_train.append(float(df['casedur'].values[0]))
print('========================================')    
print('Train results for the cases')
print('========================================')
print('\n')       
print('train mean age: ',np.mean(age_train))
print('train std age: ',np.std(age_train))
print('\n')
print('train mean sex: ',np.mean(sex_train))
print('train std sex: ',np.std(sex_train))
print('\n')
print('train mean weight: ',np.mean(weight_train))
print('train std weight: ',np.std(weight_train))
print('\n')
print('train mean height: ',np.mean(height_train))
print('train std height: ',np.std(height_train))
print('\n')
print('train mean dur: ',np.mean(np.asarray(dur_train)))
print('train std dur: ',np.std(np.asarray(dur_train)))
print('\n')

########################################################
#Valid Information:
########################################################


path = os.getcwd()
path = join(path,'valid')
cases = [f for f in listdir(path) if isfile(join(path, f))] 
cases_num = []
for filename in cases:      
    cases_num.append(int(regex.findall(filename)[0]))
    
cases_num
age_valid = []
sex_valid = []
weight_valid = []
height_valid = []
dur_valid = []
for case in cases_num:
    path = os.getcwd()
    df = pd.read_csv(os.path.join(path,'cases_all_info.csv'),sep = ';')
    df = df.loc[df['caseid'] == case]
    #input_len = df['Len'][0]
    age_valid.append(float(df['age'].values[0]))
    sex_valid.append(float(df['sex'].values[0]))
    weight_valid.append(float(df['weight'].values[0]))
    height_valid.append(float(df['height'].values[0]))
    dur_valid.append(float(df['casedur'].values[0]))
print('========================================')    
print('Valid results for the cases')
print('========================================')
print('\n')      
print('valid mean age: ',np.mean(age_valid))
print('valid std age: ',np.std(age_valid))
print('\n')
print('valid mean sex: ',np.mean(sex_valid))
print('valid std sex: ',np.std(sex_valid))
print('\n')
print('valid mean weight: ',np.mean(weight_valid))
print('valid std weight: ',np.std(weight_valid))
print('\n')
print('valid mean height: ',np.mean(height_valid))
print('valid std height: ',np.std(height_valid))
print('\n')
print('valid mean dur: ',np.mean(np.asarray(dur_valid)))
print('valid std dur: ',np.std(np.asarray(dur_valid)))

###################################################################
###################################################################
###################################################################


def mean_absolute_percentage_error(y_true, y_pred):
    #y_true, y_pred = check_array(y_true, y_pred)
    mape = [abs(t-p)/t for t,p in zip(y_true,y_pred) if t!=0]
    
    return np.mean(mape) * 100 #np.mean(np.abs((np.asarray(y_true) - np.asarray(y_pred)) / np.asarray(y_true))) * 100

def mean_percentage_error(y_true, y_pred):
    
    mpe = [(t-p)/t for t,p in zip(y_true,y_pred) if t!=0]
    
    return np.mean(mpe) * 100

    #return np.mean((np.asarray(y_true) - np.asarray(y_pred)) / np.asarray(y_true)) * 100
#Generate error mean erro metric with standard deviation
path = os.getcwd()
path = join(path,'result')

cases = [f for f in listdir(path) if isfile(join(path, f))] 
cases_num = []
regex = re.compile(r'\d+')
for filename in cases:      
    cases_num.append(int(regex.findall(filename)[0]))    
cases_num

rmse_vector_BIS = []
rmse_vector_MBP = []
rmse_vector_HR  = []

mape_vector_BIS = []
mape_vector_MBP = []
mape_vector_HR  = []

mpe_vector_BIS = []
mpe_vector_MBP = []
mpe_vector_HR  = []

for case in cases_num:
    name_1 = path +'\\output_BIS_for_case_' + str(case) + '.npy'
    output = np.load(name_1)
    name_2 = path +'\\target_BIS_for_case_' + str(case) + '.npy'
    target = np.load(name_2)
    rmse_BIS = sqrt(mean_squared_error(target, output))
    rmse_vector_BIS.append(rmse_BIS)
    
    mape_BIS = mean_absolute_percentage_error(target,output)
    mape_vector_BIS.append(mape_BIS)
    
    mpe_BIS = mean_percentage_error(target,output)
    mpe_vector_BIS.append(mpe_BIS)
    
    name_1 = path +'\\output_MBP_for_case_' + str(case) + '.npy'
    output = np.load(name_1)
    name_2 = path +'\\target_MBP_for_case_' + str(case) + '.npy'
    target = np.load(name_2)
    rmse_MBP = sqrt(mean_squared_error(target, output)) 
    rmse_vector_MBP.append(rmse_MBP)
    
    mape_MBP = mean_absolute_percentage_error(target,output)
    mape_vector_MBP.append(mape_MBP)
    
    mpe_MBP = mean_percentage_error(target,output)
    mpe_vector_MBP.append(mpe_MBP)
    
    name_1 = path +'\\output_HR_for_case_' + str(case) + '.npy'
    output = np.load(name_1)
    name_2 = path +'\\target_HR_for_case_' + str(case) + '.npy'
    target = np.load(name_2)
    rmse_HR = sqrt(mean_squared_error(target, output))
    rmse_vector_HR.append(rmse_HR)

    mape_HR = mean_absolute_percentage_error(target,output)
    mape_vector_HR.append(mape_HR)
    
    mpe_HR = mean_percentage_error(target,output)
    mpe_vector_HR.append(mpe_HR)
    
print('========================================')    
print('RMSE - TEST error distribution')
print('========================================')
print('\n')
print('Mean RMSE BIS :', np.mean(rmse_vector_BIS))
print('Std RMSE BIS :', np.std(rmse_vector_BIS))
print('\n')
print('Mean RMSE MBP:', np.mean(rmse_vector_MBP))
print('Std RMSE MBP:', np.std(rmse_vector_MBP))
print('\n')
print('Mean RMSE HR:', np.mean(rmse_vector_HR))
print('Std RMSE HR:', np.std(rmse_vector_HR))
print('\n')
print('========================================')    
print('MAPE- TEST error distribution')
print('========================================')
print('\n')
print('Mean MAPE BIS :', np.mean(mape_vector_BIS))
print('Std MAPE BIS :', np.std(mape_vector_BIS))
print('\n')
print('Mean MAPE MBP:', np.mean(mape_vector_MBP))
print('Std MAPE MBP:', np.std(mape_vector_MBP))
print('\n')
print('Mean MAPE HR:', np.mean(mape_vector_HR))
print('Std MAPE HR:', np.std(mape_vector_HR))
print('\n')
print('========================================')    
print('MPE - TEST error distribution')
print('========================================')
print('\n')
print('Mean MPE BIS :', np.mean(mpe_vector_BIS))
print('Std MPE BIS :', np.std(mpe_vector_BIS))
print('\n')
print('Mean MPE MBP:', np.mean(mpe_vector_MBP))
print('Std MPE MBP:', np.std(mpe_vector_MBP))
print('\n')
print('Mean MPE HR:', np.mean(mpe_vector_HR))
print('Std MPE HR:', np.std(mpe_vector_HR))
print('\n')

mean_format = '.2E'
std_format = '.2E'
result_table = pd.DataFrame([['BIS',str(format(np.mean(rmse_vector_BIS),mean_format))+'+-'+str(format(np.std(rmse_vector_BIS),std_format)),str(format(np.mean(mape_vector_BIS),std_format))+'+-'+str(format(np.std(mape_vector_BIS),std_format)),str(format(np.mean(mpe_vector_BIS),std_format))+'+-'+str(format(np.std(mpe_vector_BIS),std_format))],
                            ['MBP',str(format(np.mean(rmse_vector_MBP),mean_format))+'+-'+str(format(np.std(rmse_vector_MBP),std_format)),str(format(np.mean(mape_vector_MBP),std_format))+'+-'+str(format(np.std(mape_vector_MBP),std_format)),str(format(np.mean(mpe_vector_MBP),std_format))+'+-'+str(format(np.std(mpe_vector_MBP),std_format))],
                            ['HR',str(format(np.mean(rmse_vector_HR),mean_format))+'+-'+str(format(np.std(rmse_vector_HR),std_format)),str(format(np.mean(mape_vector_HR),std_format))+'+-'+str(format(np.std(mape_vector_HR),std_format)),str(format(np.mean(mpe_vector_HR),std_format))+'+-'+str(format(np.std(mpe_vector_HR),std_format))]],
                            columns =[' ','RMSE','MAPE','MPE'])

identifier = sys.argv[1] 
#identifier = 1
result_table.to_csv('result_table'+str(identifier)+'.csv')

name = 'RMSE_vector' + str(identifier)
np.save(name,rmse_vector_MBP)
np.save(name,rmse_vector_MBP)
np.save(name,rmse_vector_HR)

np.save('mape_vector' + str(identifier),mape_vector_BIS)
np.save('mape_vector' + str(identifier),mape_vector_MBP)
np.save('mape_vector' + str(identifier),mape_vector_HR)

np.save('mpe_vector' + str(identifier),mpe_vector_BIS)
np.save('mpe_vector' + str(identifier),mpe_vector_MBP)
np.save('mpe_vector' + str(identifier),mpe_vector_HR)
###################################
# Kolmogorov-Smirnov test
###################################
print('========================================')    
print('Kolmogorov-Smirnov test')
print('========================================')
print('\n')
# explanation: 
from scipy import stats
print(stats.kstest(rmse_vector_BIS, 'norm'))
print(stats.kstest(mape_vector_MBP, 'norm'))
print(stats.kstest(mpe_vector_HR, 'norm'))


import numpy as np
from scipy import stats
print('========================================')    
print('t-student test')
print('========================================')
print('\n')

#print(stats.ttest_ind_from_stats(13.39,5.34, 204, 9, 2.6,100))
#print(stats.ttest_ind_from_stats(28.09,20.51, 204, 13.9 , 5.3,100))
#print(stats.ttest_ind_from_stats(-20.04,23.68, 204, -0.1, 11.9,100))

### You can see that after comparing the t statistic with the critical t value (computed internally) we get a good p value of 0.0005 and thus we reject the null hypothesis and thus it proves that the mean of the two distributions are different and statistically significant.



