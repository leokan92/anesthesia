# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 01:47:40 2019

@author: Leonardo
Training neural nets

"""

# -*- coding: utf-8 -*-
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime as dt
import matplotlib.pyplot as plt
import os
# Torch related:
from keras.layers import Input,Conv1D, MaxPooling1D,BatchNormalization,UpSampling1D,Dropout,concatenate,Dense,Flatten
from keras.models import Model
from keras.optimizers import Adam
from tqdm import tqdm
import tensorflow as tf
import keras as K
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from keras.callbacks import ModelCheckpoint

def mean_absolute_percentage_error(y_true, y_pred):
    #y_true, y_pred = check_array(y_true, y_pred)
    return np.mean(np.abs((np.asarray(y_true) - np.asarray(y_pred)) / np.asarray(y_true))) * 100

def mean_percentage_error(y_true, y_pred):
    return np.mean((np.asarray(y_true) - np.asarray(y_pred)) / np.asarray(y_true)) * 100

def create_mod_time_series_BIS(x,M):
        z=[]
        z.append(np.ones(M)*80)
        for i in range(1,len(x)):
            z.append(np.append(z[i-1][1:],x[i]))
        return z

def create_mod_time_series_HR(x,M):
        z=[]
        z.append(np.ones(M)*70)
        for i in range(1,len(x)):
            z.append(np.append(z[i-1][1:],x[i]))
        return z    

def create_mod_time_series_MBP(x,M):
        z=[]
        z.append(np.ones(M)*100)
        for i in range(1,len(x)):
            z.append(np.append(z[i-1][1:],x[i]))
        return z    

def create_mod_time_series_drug(x,M):
        z=[]
        z.append(np.zeros(M))
        for i in range(1,len(x)):
            z.append(np.append(z[i-1][1:],x[i]))
        return z    

def return_mod_time_series(x,M):
        z=[]
        for i in range(1,len(x)):
            z.append(x[i,M])
            

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')



def plot_results(output,target,name = 'results', path = os.getcwd()):
    fig = plt.figure(num = None, figsize=(16, 6))
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 15}
    plt.rc('font', **font)
    #plt.title('Result with sharpe ration and TC = 0')
    plt.plot(output,'k-',label = 'Model output')
    plt.plot(target,'g-',label = 'Real values')
    plt.grid()
    plt.xlim([0, len(target)])
    plt.ylim([np.min(target), np.max(target)])
    plt.legend(loc='best')
    plt.xlabel('time steps')
    fig.savefig(os.path.join(os.path.join(path,name)), dpi=300, facecolor='w', edgecolor='k')
    plt.close()

class Net():
    def __init__(self, M):
        self.M = M
        
    def network(self):
        X1 = Input(shape = (self.M,2))
        
        X6 = Input(shape = (self.M-1,3))
        X2 = Input(shape = (4,))
        
        conv1_1 = Conv1D(64, 5, activation='relu', kernel_initializer='he_normal')(X1)
        conv2_1 = Conv1D(32, 5, activation='relu', kernel_initializer='he_normal')(conv1_1)
        conv3_1 = Conv1D(16, 5, activation='relu', kernel_initializer='he_normal')(conv2_1)
        Norm_1 = BatchNormalization()(conv3_1)
        flat_1 = Flatten()(Norm_1)
        
          
        conv1_6 = Conv1D(64, 5, activation='relu', kernel_initializer='he_normal')(X6)
        conv2_6 = Conv1D(32, 5, activation='relu', kernel_initializer='he_normal')(conv1_6)
        conv3_6 = Conv1D(16, 5, activation='relu', kernel_initializer='he_normal')(conv2_6)  
        Norm_6 = BatchNormalization()(conv3_6)
        flat_6 = Flatten()(Norm_6)
        
        dense1 = Dense(50, activation='relu')(flat_1)
        drouput1 = Dropout(0.5)(dense1)
        dense6 = Dense(50, activation='relu')(flat_6)
        drouput2 = Dropout(0.5)(dense6)
        densep = Dense(4, activation='relu')(X2)
        drouput3 = Dropout(0.5)(densep)
#        dense_case = Dense(50, activation='relu')(flat_6)
        
        conc1 = concatenate([drouput1,drouput2,drouput3])
        
        output1 = Dense(self.M,activation='linear')(conc1)
        output2 = Dense(self.M,activation='linear')(conc1)
        output3 = Dense(self.M,activation='linear')(conc1)

        model = Model(inputs=[X1, X6, X2], outputs=[output1,output2,output3])
        model.compile(optimizer=Adam(), loss='mean_squared_error')
        model.summary()
       
        return model
    
class PatientSimulator(object):
    def __init__(self, M = 20, path = os.getcwd()):
        self.M = M
        self.path = path
        self.net = Net(self.M)  
        self.network = self.net.network()
        #self.opt = optim.SGD(self.net.parameters(), lr=0.0075)
        #self.loss_fn = nn.MSELoss()
        
        
        self.validation_loss_hist = [] 
            
    
    def load_sample(self,case=3,folder = ''):
        
        fname = os.path.join(self.path,folder,'case_db'+str(case)+'.npy')
        df = np.load(fname,allow_pickle=True)
        
        #Data preprocessing
        x1 = np.nan_to_num(df[True][0][int(case)]['Solar8000/ART_MBP'])
        x1_ = np.nan_to_num(df[True][0][int(case)]['Solar8000/NIBP_MBP'])
        #x1 = x1 if x1>10 else x1_
        x1 = np.where(x1>10,x1_,x1) 
        x2 = np.nan_to_num(df[True][0][int(case)]['Solar8000/PLETH_HR'])
        x3 = np.nan_to_num(df[True][0][int(case)]['Orchestra/PPF20_RATE'])
        x4 = np.nan_to_num(df[True][0][int(case)]['Orchestra/RFTN20_RATE'])
        x5 = np.nan_to_num(df[True][0][int(case)]['BIS/BIS']) 
        y1 = np.nan_to_num(df[True][0][int(case)]['BIS/BIS'])
        y2 = np.nan_to_num(df[True][0][int(case)]['Solar8000/ART_MBP'])
        y3 = np.nan_to_num(df[True][0][int(case)]['Solar8000/PLETH_HR'])
        
        #print(df[True][0][int(case)]['Solar8000/ART_MBP'])
        #print(x1.shape, 'x1.shape before')
        
        #Removing zeros from the begining and the end of the series
        first_remove_BIS = len(x5[:300][x5[:300] <= 20])
        second_remove_BIS = len(x5[-300:][x5[-300:] <= 20])
        
        first_remove_MBP = len(x5[:300][x5[:300] <= 20])
        second_remove_MBP = len(x5[-300:][x5[-300:] <= 20])
        
        if first_remove_MBP>first_remove_BIS:
            first_remove = first_remove_MBP
        else:
            first_remove = first_remove_BIS
        
        if second_remove_MBP>second_remove_BIS:
            second_remove = second_remove_MBP
        else:
            second_remove = second_remove_BIS
        
        #print(first_remove)
        #print(second_remove)
        
        x1 = x1[first_remove:len(x1)-second_remove]
        x2 = x2[first_remove:len(x2)-second_remove]
        x3 = x3[first_remove:len(x3)-second_remove]
        x4 = x4[first_remove:len(x4)-second_remove]
        x5 = x5[first_remove:len(x5)-second_remove]
        y1 = y1[first_remove:len(y1)-second_remove]
        y2 = y2[first_remove:len(y2)-second_remove]
        y3 = y3[first_remove:len(y3)-second_remove]
        
        x1 = movingaverage(x1,10)[10:]
        x2 = movingaverage(x2,10)[10:]
        x5 = movingaverage(x5,10)[10:]
        y1 = movingaverage(y1,10)[10:]
        y2 = movingaverage(y2,10)[10:]
        y3 = movingaverage(y3,10)[10:]
        x3 = x3[10:]
        x4 = x4[10:]
        
        
        #print(x1.shape, 'x1.shape after')
        
        
        input1 =  np.asarray(create_mod_time_series_MBP(x1,self.M-1))
        input2 =  np.asarray(create_mod_time_series_HR(x2,self.M-1))
        input3 =  np.asarray(create_mod_time_series_drug(x3,self.M))
        input4 =  np.asarray(create_mod_time_series_drug(x4,self.M))
        input5 =  np.asarray(create_mod_time_series_BIS(x5,self.M-1))
        #self.input6 = np.expand_dims(self.input6, axis=-1)
        target1 = np.asarray(create_mod_time_series_BIS(y1,self.M))
        target2 = np.asarray(create_mod_time_series_MBP(y2,self.M))
        target3 = np.asarray(create_mod_time_series_HR(y3,self.M))
        #self.target = np.expand_dims(self.target, axis=-1)
        
        #(input1.shape, 'input1.shape')
        
        input_anest = np.zeros([len(input1),self.M,2])        
        input_anest[...,0] = input3
        input_anest[...,1] = input4
        
        input_patient_hist= np.zeros([len(input1),self.M-1,3])        
        input_patient_hist[...,0] = input1
        input_patient_hist[...,1] = input2
        input_patient_hist[...,2] = input5

      
        #Load the case information
        df = pd.read_csv(os.path.join(path,'cases-anestesia.csv'),sep = ';')
        df = df.loc[df['caseid'] == str(case)]
        #input_len = df['Len'][0]
        input_age = df['age'].values[0]
        input_sex = df['sex'].values[0]
        input_weight = df['weight'].values[0]
        input_height = df['height'].values[0]
        input_case = [float(input_age),float(input_sex),float(input_weight),float(input_height)]
        input_case_mod = []
        for i in range(0,len(input_anest)):
            input_case_mod.append(input_case)

        return input_anest,input_patient_hist, target1,target2,target3,np.asarray(input_case_mod)
       
    def load_data(self):

        path_train = os.path.join(self.path,'train')
        cases = [file.split('_')[1][2:-4] for  file in sorted(os.listdir(path_train))]
        sample = self.load_sample(case = cases[0],folder='train')
        self.train_anest = sample[0]
        self.train_hist = sample[1]
        self.train_target_BIS = sample[2]
        self.train_target_MBP  = sample[3]
        self.train_target_HR = sample[4]
        self.train_case = sample[5]
#        print(type(self.train_target_HR.shape))
#        print(self.train_case)
#        print(self.train_case.shape,'shape of train_case')
        cases = cases[1:]

        for case in cases:
            sample = self.load_sample(case = case, folder = 'train')
            self.train_anest = np.concatenate((self.train_anest,sample[0]),axis=0)
            self.train_hist =np.concatenate((self.train_hist,sample[1]),axis=0)
            self.train_case =np.concatenate((self.train_case,sample[5]),axis=0)
            self.train_target_BIS = np.concatenate((self.train_target_BIS,sample[2]),axis=0)
            self.train_target_MBP  = np.concatenate((self.train_target_MBP,sample[3]),axis=0)
            self.train_target_HR = np.concatenate((self.train_target_HR,sample[4]),axis=0)
            #print(self.train_target_HR.shape, 'self.shape do train_targerHR')
            #print(self.train_anest.shape, 'self.shape do train_anest')
        

        path_valid = os.path.join(self.path,'valid')
        cases = [file.split('_')[1][2:-4] for  file in os.listdir(path_valid)]

        sample = self.load_sample(case = cases[0], folder='valid')
        self.valid_anest = sample[0]
        self.valid_hist = sample[1]
        self.valid_target_BIS = sample[2]
        self.valid_target_MBP  = sample[3]
        self.valid_target_HR = sample[4]
        self.valid_case = sample[5]
        cases = cases[1:]
        
        
        for case in cases:
            sample = self.load_sample(case = case, folder='valid')
            self.valid_anest = np.concatenate((self.valid_anest,sample[0]),axis=0)
            self.valid_hist =np.concatenate((self.valid_hist,sample[1]),axis=0)
            self.valid_case =np.concatenate((self.valid_case,sample[5]),axis=0)
            self.valid_target_BIS = np.concatenate((self.valid_target_BIS,sample[2]),axis=0)
            self.valid_target_MBP  = np.concatenate((self.valid_target_MBP,sample[3]),axis=0)
            self.valid_target_HR = np.concatenate((self.valid_target_HR,sample[4]),axis=0)
        
        path_test = os.path.join(self.path,'test')
        cases = [file.split('_')[1][2:-4] for  file in os.listdir(path_test)]

        sample = self.load_sample(case = cases[0], folder='test')
        self.test_anest = sample[0]
        self.test_hist = sample[1]
        self.test_target_BIS = sample[2]
        self.test_target_MBP  = sample[3]
        self.test_target_HR = sample[4]
        self.test_case = sample[5]
        cases = cases[1:]
        
        for case in cases:
            sample = self.load_sample(case = case, folder='test')
            self.test_anest = np.concatenate((self.test_anest,sample[0]),axis=0)
            self.test_hist =np.concatenate((self.test_hist,sample[1]),axis=0)
            self.test_case =np.concatenate((self.test_case,sample[5]),axis=0)
            self.test_target_BIS = np.concatenate((self.test_target_BIS,sample[2]),axis=0)
            self.test_target_MBP  = np.concatenate((self.test_target_MBP,sample[3]),axis=0)
            self.test_target_HR = np.concatenate((self.test_target_HR,sample[4]),axis=0)



    
    def train(self,epochs,batch_size):
        print('Training...')
        
        check = ModelCheckpoint(os.path.join(self.path,'model_best_w.h5'), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        h = self.network.fit([self.train_anest,self.train_hist,self.train_case], [self.train_target_BIS,self.train_target_MBP, self.train_target_HR],
            validation_data=([self.valid_anest, self.valid_hist,self.valid_case],[self.valid_target_BIS,self.valid_target_MBP,self.valid_target_HR]), 
            batch_size=batch_size,epochs=epochs,verbose=2,
            callbacks=[check])

        fig = plt.figure(num = None, figsize=(16, 6))
        plt.plot(h.history['loss'][5:], label = 'loss')
        plt.plot(h.history['val_loss'][5:], label = 'val_loss')
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc='upper right')
        fig.savefig(os.path.join(self.path,'train_hist.png'), dpi=300, facecolor='w', edgecolor='k')
                
    def save_weights(self,name):
        print('Saving model...')
        self.network.save_weights(os.path.join(self.path,name))
            
    def load_weights(self,name):
        print('Loading model...')
        self.network.load_weights(os.path.join(self.path,name))

        
    def test_sample(self,sample_test,case):
       
        #Implementation that does not use target history but the output hist as input
        print('Sample Testing...')
        output_list_1 = []
        output_list_2 = []
        output_list_3 = []
        target_list_1 = []
        target_list_2 = []
        target_list_3 = []
        output_1 = sample_test[1][0,...,0]
        output_1 = np.expand_dims(output_1, axis=-1)
        output_2 = sample_test[1][0,...,1]
        output_2 = np.expand_dims(output_2, axis=-1)
        output_3 = sample_test[1][0,...,2]
        output_3 = np.expand_dims(output_3, axis=-1)
        sample_n = np.append(output_1,output_2,axis=-1)
        sample_n = np.append(sample_n,output_3,axis=-1)
        #print(sample_test[0].shape)
        for i in tqdm(range(0,len(sample_test[0]))):
            output_1_n,output_2_n,output_3_n = self.network.predict([np.expand_dims(sample_test[0][i],0),np.expand_dims(sample_n,0),np.expand_dims(sample_test[5][i],0)])
            #Append the prediceted value only
            output_list_1.append(output_1_n[0,-1])
            output_list_2.append(output_2_n[0,-1])
            output_list_3.append(output_3_n[0,-1])
            target_list_1.append(sample_test[-4][i,-1])
            target_list_2.append(sample_test[-3][i,-1])
            target_list_3.append(sample_test[-2][i,-1])
            output_1 = np.expand_dims(np.append(output_1[:-1],output_1_n[0,-1]),-1)
            output_2 = np.expand_dims(np.append(output_2[:-1],output_2_n[0,-1]),-1)
            output_3 = np.expand_dims(np.append(output_3[:-1],output_3_n[0,-1]),-1)
            sample_n = np.append(output_1,output_2,axis=-1)
            sample_n = np.append(sample_n,output_3,axis=-1)

        name = 'results_tes_BIS_case' + str(case) + '.png'
        plot_results(output_list_1,target_list_1,name,path=self.path)
        name = 'results_tes_MBP_case' + str(case) + '.png'
        plot_results(output_list_2,target_list_2,name,path=self.path)  
        name = 'results_tes_HR_case' + str(case) + '.png'
        plot_results(output_list_3,target_list_3,name,path=self.path)
        
        
        name = 'target_BIS_for_case_' + str(case)+'.npy'
        np.save(os.path.join(self.path,'result',name),np.asarray(target_list_1))
        name = 'output_BIS_for_case_' + str(case)+'.npy'
        np.save(os.path.join(self.path,'result',name),np.asarray(output_list_1))
        name = 'target_MBP_for_case_' + str(case)+'.npy'
        np.save(os.path.join(self.path,'result',name),np.asarray(target_list_2))
        name = 'output_MBP_for_case_' + str(case)+'.npy'
        np.save(os.path.join(self.path,'result',name),np.asarray(output_list_2))
        name = 'target_HR_for_case_' + str(case)+'.npy'
        np.save(os.path.join(self.path,'result',name),np.asarray(target_list_1))
        name = 'output_HR_for_case_' + str(case)+'.npy'
        np.save(os.path.join(self.path,'result',name),np.asarray(output_list_3))
        
#        rmse_BIS = sqrt(mean_squared_error(target_list_1, output_list_1))    
#        print('RMSE for BIS: ',round(rmse_BIS,5))
#        rmse_MBP = sqrt(mean_squared_error(target_list_2, output_list_2))    
#        print('RMSE for MBP: ',round(rmse_MBP,5))
#        rmse_HR = sqrt(mean_squared_error(target_list_3, output_list_3))    
#        print('RMSE for HR: ',round(rmse_HR,5))
#        name = 'rmse_for_case_' + str(case)+'.npy'
#        np.save(os.path.join(self.path,'result',name),np.asarray([rmse_BIS,rmse_MBP,rmse_HR]))
#        
#        mape_BIS = mean_absolute_percentage_error(target_list_1, output_list_1)
#        print('mape for BIS: ',round(mape_BIS,5))
#        mape_MBP = mean_absolute_percentage_error(target_list_2, output_list_2)  
#        print('mape for MBP: ',round(mape_MBP,5))
#        mape_HR = mean_absolute_percentage_error(target_list_3, output_list_3)   
#        print('mape for HR: ',round(mape_HR,5))
#        name = 'mape_for_case_' + str(case)+'.npy'
#        np.save(os.path.join(self.path,'result',name),np.asarray([mape_BIS,mape_MBP,mape_HR]))
#        
#        mpe_BIS = mean_percentage_error(target_list_1, output_list_1)
#        print('mpe for BIS: ',round(mpe_BIS,5))
#        mpe_MBP = mean_percentage_error(target_list_2, output_list_2)  
#        print('mpe for MBP: ',round(mpe_MBP,5))
#        mpe_HR = mean_percentage_error(target_list_3, output_list_3)   
#        print('mpe for HR: ',round(mpe_HR,5))
#        name = 'mpe_for_case_' + str(case)+'.npy'
#        np.save(os.path.join(self.path,'result',name),np.asarray([mpe_BIS,mpe_MBP,mpe_HR]))
        
    def test(self):
        print('Metric Testing...')
        output_list_1 = []
        output_list_2 = []
        output_list_3 = []
        target_list_1 = []
        target_list_2 = []
        target_list_3 = []
        #print(self.test_anest.shape)
        output_1 = np.zeros([self.M-1])
        output_1 = np.expand_dims(output_1, axis=-1)
        output_2 = np.zeros([self.M-1])
        output_2 = np.expand_dims(output_2, axis=-1)
        output_3 = np.zeros([self.M-1])
        output_3 = np.expand_dims(output_3, axis=-1)
        output_1_n,output_2_n,output_3_n = self.network.predict([self.test_anest,self.test_hist,self.test_case])
            #Append the prediceted value only
        
        output_list_1.append(output_1_n[...,-1])
        output_list_2.append(output_2_n[...,-1])
        output_list_3.append(output_3_n[...,-1])
        target_list_1 = self.test_target_BIS[...,-1]
        target_list_2 = self.test_target_MBP[...,-1]
        target_list_3 = self.test_target_HR[...,-1]
        
        #print(output_list_1[0].shape)
        #print(target_list_1.shape)
#        for i in tqdm(range(0,len(self.test_anest))):
#            output_list_1.append(output_1[0,0])
#            output_list_2.append(output_2[0,0])
#            output_list_3.append(output_3[0,0])
#            target_list_1.append(self.test_target_BIS[i,0])
#            target_list_2.append(self.test_target_MBP[i,0])
#            target_list_3.append(self.test_target_HR[i,0])
#            output_1 = np.expand_dims(np.append(output_1[1:,0],output_1_n[0,0]),-1)
#            output_2 = np.expand_dims(np.append(output_2[1:,0],output_2_n[0,0]),-1)
#            output_3 = np.expand_dims(np.append(output_3[1:,0],output_3_n[0,0]),-1)
        
        rmse = sqrt(mean_squared_error(target_list_1, output_list_1[0]))    
        print('RMSE for BIS: ',round(rmse,2))
        rmse = sqrt(mean_squared_error(target_list_2, output_list_2[0]))    
        print('RMSE for MBP: ',round(rmse,2))
        rmse = sqrt(mean_squared_error(target_list_3, output_list_3[0]))    
        print('RMSE for HR: ',round(rmse,2))
                
       
        
####################################################################
# Main
####################################################################
        
#path = r'C:\Users\catdv\Documents\Pós\anest'#r'/home/catharine.graves/data/exp'
path = os.getcwd()
patient = PatientSimulator(200,path)
#
#  train
patient.load_data()
#patient.load_weights('model_w.h5')
patient.train(50,60)
patient.save_weights('model_w.h5')

#test
patient.load_weights('model_best_w.h5')
#cases = [5501, 5502, 5508, 5509]#, 5511, 5515, 5519, 5520, 5534, 5536, 5541, 5546, 5556, 5561, 5562, 5566, 5571, 5572, 5573, 5574, 5578, 5583, 5585, 5587, 5589, 5607, 5608, 5610, 5613, 5614, 5616, 5618, 5626, 5629, 5634, 5635, 5637, 5642, 5648, 5659, 5664, 5669, 5670, 5671, 5675, 5680, 5682, 5687, 5691, 5692, 5693, 5696, 5715, 5717, 5718, 5724, 5743, 5749, 5751, 5765, 5769, 5772, 5777, 5780, 5781, 5782, 5784, 5788, 5793, 5800, 5801, 5809, 5810, 5811, 5814, 5817, 5818, 5825, 5827, 5832, 5834, 5837, 5840, 5842, 5844, 5851, 5859, 5861, 5866, 5871, 5873, 5882, 5887, 5888, 5889, 5908, 5912, 5914, 5934, 5936, 5937, 5938, 5942, 5946, 5951, 5958, 5959, 5961, 5965, 5966, 5971, 5974, 5976, 5981, 5983, 5986, 5987, 5989, 6006, 6009, 6015, 6017, 6027, 6029, 6032, 6037, 6041, 6042, 6053, 6055, 6059, 6061, 6069, 6070, 6074, 6088, 6089, 6097, 6104, 6114, 6119, 6121, 6124, 6126, 6127, 6131, 6135, 6140, 6147, 6152, 6154, 6163, 6174, 6180, 6185, 6186, 6190, 6192, 6195, 6196, 6200, 6204, 6205, 6208, 6210, 6227, 6233, 6235, 6241, 6254, 6257, 6259, 6262, 6264, 6267, 6268, 6269, 6271, 6275, 6277, 6280, 6281, 6284, 6286, 6292, 6293, 6298, 6305, 6306, 6309, 6311, 6330, 6345, 6346, 6351, 6355, 6357, 6359, 6366, 6370, 6373, 6376, 6381, 6383]
cases = [case.split('_')[-1][2:-4] for case in os.listdir(os.path.join(path,'test'))]
for case in cases:
    sample_test_data = patient.load_sample(case,folder='test')
    patient.test_sample(sample_test_data,case)
#patient.test()




   
