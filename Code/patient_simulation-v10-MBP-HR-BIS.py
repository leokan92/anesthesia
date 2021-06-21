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
            

def movingaverage(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)


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
        X2 = Input(shape = (17,))
        
        conv1_1 = Conv1D(64, 3, activation='relu', kernel_initializer='he_normal')(X1)
        conv2_1 = Conv1D(32, 3, activation='relu', kernel_initializer='he_normal')(conv1_1)
        conv3_1 = Conv1D(16, 3, activation='relu', kernel_initializer='he_normal')(conv2_1)
        Norm_1 = BatchNormalization()(conv3_1)
        MP1 = MaxPooling1D(pool_size=2)(Norm_1)
        flat_1 = Flatten()(MP1)
        
          
        conv1_6 = Conv1D(64, 3, activation='relu', kernel_initializer='he_normal')(X6)
        conv2_6 = Conv1D(32, 3, activation='relu', kernel_initializer='he_normal')(conv1_6)
        conv3_6 = Conv1D(16, 3, activation='relu', kernel_initializer='he_normal')(conv2_6)  
        Norm_6 = BatchNormalization()(conv3_6)
        MP6 = MaxPooling1D(pool_size=2)(Norm_6)
        flat_6 = Flatten()(MP6)
        
        dense1 = Dense(100, activation='relu')(flat_1)
        drouput1 = Dropout(0.5)(dense1)
        dense6 = Dense(100, activation='relu')(flat_6)
        drouput2 = Dropout(0.5)(dense6)
        densep = Dense(100, activation='relu')(X2)
        drouput3 = Dropout(0.5)(densep)
#        dense_case = Dense(50, activation='relu')(flat_6)
        
        conc1 = concatenate([drouput1,drouput2,drouput3])

        dense4 = Dense(self.M*3,activation='linear')(conc1)
        
        output1 = Dense(self.M,activation='linear')(dense4)
        output2 = Dense(self.M,activation='linear')(dense4)
        output3 = Dense(self.M,activation='linear')(dense4)

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
        #x1 = np.nan_to_num(df[True][0][int(case)]['Solar8000/ART_MBP'])
        x1 = np.nan_to_num(df[True][0][int(case)]['Solar8000/NIBP_MBP'])
        #x1 = x1 if x1>10 else x1_
        x2 = np.nan_to_num(df[True][0][int(case)]['Solar8000/PLETH_HR'])
        x3 = np.nan_to_num(df[True][0][int(case)]['Orchestra/PPF20_RATE'])
        x4 = np.nan_to_num(df[True][0][int(case)]['Orchestra/RFTN20_RATE'])
        x5 = np.nan_to_num(df[True][0][int(case)]['BIS/BIS']) 
        y1 = np.nan_to_num(df[True][0][int(case)]['BIS/BIS'])
        y2 = x1
        y3 = np.nan_to_num(df[True][0][int(case)]['Solar8000/PLETH_HR'])
        
        #print(df[True][0][int(case)]['Solar8000/ART_MBP'])
        #print(x1.shape, 'x1.shape before')
        
        #Removing zeros from the begining and the end of the series
        first_remove_BIS = len(x5[:300][x5[:300] <= 10])
        second_remove_BIS = len(x5[-300:][x5[-300:] <= 10])
        
        first_remove_MBP = len(x1[:300][x1[:300] <= 10])
        second_remove_MBP = len(x1[-300:][x1[-300:] <= 10])
        
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
        
        window = 10
        x1 = movingaverage(x1,window)
        x2 = movingaverage(x2,window)
        x5 = movingaverage(x5,window)
        y1 = movingaverage(y1,window)
        y2 = movingaverage(y2,window)
        y3 = movingaverage(y3,window)
        x3 = movingaverage(x3,window)
        x4 = movingaverage(x4,window)
        
        x1 = np.where(x1>10,x1,np.mean(x1))
        #print(x1.shape, 'x1.shape after')
        
        
        inputMBP =  np.asarray(create_mod_time_series_MBP(x1,self.M-1))
        inputHR =  np.asarray(create_mod_time_series_HR(x2,self.M-1))
        inputPPF =  np.asarray(create_mod_time_series_drug(x3,self.M))
        inputRFT =  np.asarray(create_mod_time_series_drug(x4,self.M))
        inputBIS =  np.asarray(create_mod_time_series_BIS(x5,self.M-1))
        #self.input6 = np.expand_dims(self.input6, axis=-1)
        targetBIS = np.asarray(create_mod_time_series_BIS(y1,self.M))
        targetMBP = np.asarray(create_mod_time_series_MBP(y2,self.M))
        targetHR = np.asarray(create_mod_time_series_HR(y3,self.M))
        #self.target = np.expand_dims(self.target, axis=-1)
        
        #(input1.shape, 'input1.shape')
        
        input_anest = np.zeros([len(inputBIS),self.M,2])        
        input_anest[...,0] = inputPPF
        input_anest[...,1] = inputRFT
        
        input_patient_hist= np.zeros([len(inputBIS),self.M-1,3])        
        input_patient_hist[...,0] = inputBIS
        input_patient_hist[...,1] = inputMBP
        input_patient_hist[...,2] = inputHR

      
        #Load the case information
        df = pd.read_csv(os.path.join(path,'cases_all_info.csv'),sep = ';')
        df = df.loc[df['caseid'] == int(case)]
        #input_len = df['Len'][0]
        input_opdur = df['opdur'].values[0]
        input_age = df['age'].values[0]
        input_sex = df['sex'].values[0]
        input_weight = df['weight'].values[0]
        input_height = df['height'].values[0]
        input_asa = df['asa'].values[0]
        input_anedur = df['anedur'].values[0]
        input_preop_htn = df['preop_htn'].values[0]
        input_preop_dm = df['preop_dm'].values[0]
        input_preop_hb = df['preop_hb'].values[0]
        input_preop_cr = df['preop_cr'].values[0]
        input_intraop_ppf = df['intraop_ppf'].values[0]
        input_intraop_mdz = df['intraop_mdz'].values[0]
        input_intraop_ftn = df['intraop_ftn'].values[0]
        input_intraop_eph = df['intraop_eph'].values[0]
        input_intraop_phe = df['intraop_phe'].values[0]
        
        """
        - tempo de duração 
        - idade
        - sexo
        - peso
        - altura 
        - ASA
        - anedur (duração anestesia)
        - preop_htn (pressão alta)
        - preop_dm (diabetes)
        - preop_hb (hemoglobina)
        - preop_cr (creatinina)
        - intraop_ppf (propofol bolus)
        - intraop_mdz (midazolan)
        - intraop_ftn (fentanyl)
        - intraop_eph (efedrina)
        - intraop_phe (fenilefrina)
        
        """
        
        
        
        input_case = [float(input_opdur),float(input_age),float(input_sex),float(input_weight),float(input_height),float(input_asa),
                      float(input_anedur),float(input_preop_htn),float(input_preop_dm),float(input_preop_hb),
                      float(input_preop_cr),float(input_intraop_ppf),float(input_intraop_ppf),float(input_intraop_mdz),
                      float(input_intraop_ftn),float(input_intraop_eph),float(input_intraop_phe)]
        
        input_case_mod = []
        for i in range(0,len(input_anest)):
            input_case_mod.append(input_case)

        return input_anest,input_patient_hist, targetBIS,targetMBP,targetHR,np.asarray(input_case_mod)
       
    def load_data(self):

        path_train = os.path.join(self.path,'train')
        cases = [file.split('_')[1][2:-4] for  file in sorted(os.listdir(path_train))]
        sample = self.load_sample(case = cases[0],folder='train')
        self.train_anest = sample[0][self.M:]
        self.train_hist = sample[1][self.M-1:-1]
        self.train_target_BIS = sample[2][self.M:]
        self.train_target_MBP  = sample[3][self.M:]
        self.train_target_HR = sample[4][self.M:]
        self.train_case = sample[5][self.M:]
#        print(type(self.train_target_HR.shape))
#        print(self.train_case)
#        print(self.train_case.shape,'shape of train_case')
        cases = cases[1:]

        for case in cases:
            sample = self.load_sample(case = case, folder = 'train')
            self.train_anest = np.concatenate((self.train_anest,sample[0][self.M:]),axis=0)
            self.train_hist =np.concatenate((self.train_hist,sample[1][self.M-1:-1]),axis=0)
            self.train_case =np.concatenate((self.train_case,sample[5][self.M:]),axis=0)
            self.train_target_BIS = np.concatenate((self.train_target_BIS,sample[2][self.M:]),axis=0)
            self.train_target_MBP  = np.concatenate((self.train_target_MBP,sample[3][self.M:]),axis=0)
            self.train_target_HR = np.concatenate((self.train_target_HR,sample[4][self.M:]),axis=0)
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
        
        check = ModelCheckpoint(os.path.join(self.path,'model_best_w_v10.h5'), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
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
        output_list_BIS = []
        output_list_MBP = []
        output_list_HR = []
        target_list_1 = []
        target_list_2 = []
        target_list_3 = []
        output_BIS = sample_test[1][0,...,0]
        output_BIS = np.expand_dims(output_BIS, axis=-1)
        output_MBP = sample_test[1][0,...,1]
        output_MBP = np.expand_dims(output_MBP, axis=-1)
        output_HR = sample_test[1][0,...,2]
        output_HR = np.expand_dims(output_HR, axis=-1)
        sample_n = np.append(output_BIS,output_MBP,axis=-1)
        sample_n = np.append(sample_n,output_HR,axis=-1)
        #print(sample_test[0].shape)
        for i in tqdm(range(0,len(sample_test[0]))):
            output_1_BIS,output_2_MBP,output_3_HR = self.network.predict([np.expand_dims(sample_test[0][i],0),np.expand_dims(sample_n,0),np.expand_dims(sample_test[5][i],0)])
            #Append the prediceted value only
            output_list_BIS.append(output_1_BIS[0,-1])
            output_list_MBP.append(output_2_MBP[0,-1])
            output_list_HR.append(output_3_HR[0,-1])
            target_list_1.append(sample_test[-4][i,-1]) #BIS
            target_list_2.append(sample_test[-3][i,-1]) #MBP
            target_list_3.append(sample_test[-2][i,-1]) #HR
            output_BIS = np.expand_dims(output_1_BIS[0,1:],-1)#np.expand_dims(np.append(output_BIS[:-1],output_1_BIS[0,-1]),-1)
            output_MBP = np.expand_dims(output_2_MBP[0,1:],-1)#np.expand_dims(np.append(output_MBP[:-1],output_2_MBP[0,-1]),-1)
            output_HR = np.expand_dims(output_3_HR[0,1:],-1)#np.expand_dims(np.append(output_HR[:-1],output_3_HR[0,-1]),-1)
            sample_n = np.append(output_BIS,output_MBP,axis=-1)
            sample_n = np.append(sample_n,output_HR,axis=-1)


        name = 'results_tes_BIS_case' + str(case) + '.png'
        plot_results(output_list_BIS,target_list_1,name,path=self.path)
        name = 'results_tes_MBP_case' + str(case) + '.png'
        plot_results(output_list_MBP,target_list_2,name,path=self.path)  
        name = 'results_tes_HR_case' + str(case) + '.png'
        plot_results(output_list_HR,target_list_3,name,path=self.path)
        
        
        name = 'target_BIS_for_case_' + str(case)+'.npy'
        np.save(os.path.join(self.path,'result',name),np.asarray(target_list_1))
        name = 'output_BIS_for_case_' + str(case)+'.npy'
        np.save(os.path.join(self.path,'result',name),np.asarray(output_list_BIS))
        name = 'target_MBP_for_case_' + str(case)+'.npy'
        np.save(os.path.join(self.path,'result',name),np.asarray(target_list_2))
        name = 'output_MBP_for_case_' + str(case)+'.npy'
        np.save(os.path.join(self.path,'result',name),np.asarray(output_list_MBP))
        name = 'target_HR_for_case_' + str(case)+'.npy'
        np.save(os.path.join(self.path,'result',name),np.asarray(target_list_1))
        name = 'output_HR_for_case_' + str(case)+'.npy'
        np.save(os.path.join(self.path,'result',name),np.asarray(output_list_HR))

       
        
####################################################################
# Main
####################################################################
        
#path = r'd:\usuarios\catharine.graves\Meus Documentos\Datasets\anestesia\freq10'
path = r'C:\Users\leona\Google Drive\USP\Doutorado\Teste - Robo anestesista\Test with more parameters'
#path = r'C:\Users\leona\Google Drive\USP\Doutorado\Teste - Robo anestesista\New tests with new data'

patient = PatientSimulator(200,path)
#
#  train
patient.load_data()
# #patient.load_weights('model_w.h5')
patient.train(10,100)
# patient.save_weights('model_w.h5')

#test
patient.load_weights('model_best_w_v10.h5')
#cases = [5501, 5502, 5508, 5509]#, 5511, 5515, 5519, 5520, 5534, 5536, 5541, 5546, 5556, 5561, 5562, 5566, 5571, 5572, 5573, 5574, 5578, 5583, 5585, 5587, 5589, 5607, 5608, 5610, 5613, 5614, 5616, 5618, 5626, 5629, 5634, 5635, 5637, 5642, 5648, 5659, 5664, 5669, 5670, 5671, 5675, 5680, 5682, 5687, 5691, 5692, 5693, 5696, 5715, 5717, 5718, 5724, 5743, 5749, 5751, 5765, 5769, 5772, 5777, 5780, 5781, 5782, 5784, 5788, 5793, 5800, 5801, 5809, 5810, 5811, 5814, 5817, 5818, 5825, 5827, 5832, 5834, 5837, 5840, 5842, 5844, 5851, 5859, 5861, 5866, 5871, 5873, 5882, 5887, 5888, 5889, 5908, 5912, 5914, 5934, 5936, 5937, 5938, 5942, 5946, 5951, 5958, 5959, 5961, 5965, 5966, 5971, 5974, 5976, 5981, 5983, 5986, 5987, 5989, 6006, 6009, 6015, 6017, 6027, 6029, 6032, 6037, 6041, 6042, 6053, 6055, 6059, 6061, 6069, 6070, 6074, 6088, 6089, 6097, 6104, 6114, 6119, 6121, 6124, 6126, 6127, 6131, 6135, 6140, 6147, 6152, 6154, 6163, 6174, 6180, 6185, 6186, 6190, 6192, 6195, 6196, 6200, 6204, 6205, 6208, 6210, 6227, 6233, 6235, 6241, 6254, 6257, 6259, 6262, 6264, 6267, 6268, 6269, 6271, 6275, 6277, 6280, 6281, 6284, 6286, 6292, 6293, 6298, 6305, 6306, 6309, 6311, 6330, 6345, 6346, 6351, 6355, 6357, 6359, 6366, 6370, 6373, 6376, 6381, 6383]
cases = [case.split('_')[-1][2:-4] for case in os.listdir(os.path.join(path,'test'))]
for case in cases:
    sample_test_data = patient.load_sample(case,folder='test')
    patient.test_sample(sample_test_data,case)

identifier = 'v10'
os.system(r'python generate_metrics.py '+identifier)
#patient.test()

   
