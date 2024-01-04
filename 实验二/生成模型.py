# coding=utf-8
import numpy as np
import sys
import csv  
from numpy.linalg import inv
from tqdm import tqdm
from scipy.special import expit

class data_manager():
    def __init__(self):
        self.data = {}  
    
    def read(self,name,path):
        with open(path,newline = '') as csvfile:
            rows = np.array(list(csv.reader(csvfile))[1:] ,dtype = float) #ȥ�����ж�ȡ
            if name == 'X_train':
                    #���ȡ����ֵ��ֵ��תΪ1��,div=1*106
                    self.mean = np.mean(rows,axis = 0).reshape(1,-1)
                    #���ȡ����ֵ�󷽲תΪ1��,div=1*106
                    self.std = np.std(rows,axis = 0).reshape(1,-1)+0.00000000000000001
                    self.theta = np.ones((rows.shape[1] + 1,1),dtype = float)
                    #��ÿһ�����ݽ��� z-score��׼������x-�̣�/��
                    for i in range(rows.shape[0]):
                        rows[i,:] = (rows[i,:]-self.mean)/(self.std)
            elif name == 'X_test': 
                    for i in range(rows.shape[0]):
                        rows[i,:] = (rows[i,:]-self.mean)/(self.std)
                #�������ݴ������������ֵ�������
            self.data[name] = rows


    def find_theta(self):
        #���ڸ�������ģ����Ҫ��������ѡ��ĸ��ʣ�������Ҫ�ȶ�ѵ�����ݽ��ж�����
        class_0_id = []#��¼�����Ԫ������
        class_1_id = []
        for i in range(self.data['Y_train'].shape[0]):#ͨ��Y_train��������
            if self.data['Y_train'][i][0] == 0:
                class_0_id.append(i)
            else:
                class_1_id.append(i)
        #��������ȥX_train�ҵ���ӦԪ��
        class_0 = self.data['X_train'][class_0_id]
        class_1 = self.data['X_train'][class_1_id] 
        #�ֱ�������������ֵ
        mean_0 = np.mean(class_0,axis = 0)
        mean_1 = np.mean(class_1,axis = 0)  

        #��ͬ�������Э�������
        n = class_0.shape[1]
        cov_0 = np.zeros((n,n))
        cov_1 = np.zeros((n,n))

        row_total_num=class_0.shape[0]+class_1.shape[0]
        #for i in range(class_0.shape[0]):#������1��Э����
         #   cov_0 += np.dot(np.transpose(class_0[i]-mean_0),class_0[i]-mean_0)
        #for i in range(class_1.shape[0]):#������2��Э����
         #   cov_1 += np.dot(np.transpose(class_1[i]-mean_1),class_1[i]-mean_1)
         #cov = (cov_0+cov_1)/row_total_num #�����๲��һ��Э����,����cov����Э����
        #��������ѭ������˵Э�������ļ�������г�����������󣬸ĳ�ʹ�� numpy �ṩ�ĺ��� np.cov ������Э�������

       
        cov_0 = np.cov(class_0, rowvar=False)
        cov_1 = np.cov(class_1, rowvar=False)
        cov = (cov_0 * class_0.shape[0] + cov_1 * class_1.shape[0]) / (row_total_num)
        #�����๲��һ��Э����,����cov����Э����
        

        #���ݹ�ʽ��w��b
        n_0 = class_0.shape[0]
        n_1 = class_1.shape[0]#���е�����
        self.w =  np.dot((mean_0-mean_1).transpose(),inv(cov)).transpose()
        self.b =  (-1)*np.dot(np.dot(mean_0.transpose(),inv(cov)),mean_0)/2+np.dot(np.dot(mean_1.transpose(),inv(cov)),mean_1)/2+np.log(float(n_0)/n_1)
        #����õ�w��b����sigmoid()�������
        result = self.func(self.data['X_train'])#����0-1��������õ�Ԥ��ֵ
        answer = self.predict(result)#����ѵ�����ϵ�׼ȷ��
        accuracy=1-np.mean(np.abs(self.data['Y_train']-answer))
        print("w=",self.w)
        print("b=",self.b)
        print("accuracy = ",accuracy)

    
    #������ʺ���sigmoid
    def func(self,x):
        arr = np.empty([x.shape[0],1],dtype=float)
        ##������㣬��ֹexp�����e���ݴη�
        for i in range(x.shape[0]):
            z = x[i,:].dot(self.w) + self.b
           # z *= (-1)
            #arr[i][0] = 1 / (1 + np.exp(z))
            arr[i][0] =expit(z)
            #expit()ר����� logistic sigmoid �����ļ���������Ż������⣬���ڷǳ����ǳ�С������ֵ��expit(z)����ֵ�ϸ��ȶ�������������򾫶ȶ�ʧ�����⡣
            #������ֵһ����0.842��һ����0.759����Ȼ��ֵ��ͬ����ѵ��������ģ����һ���ģ�w��b��ȫһ��
        return np.clip(arr, 1e-8, 1-(1e-8))


    def predict(self,x):#����������������ֵ��0.5�����1���������ֵ��0.5�����0
        ans = np.ones([x.shape[0],1],dtype=int)
        for i in range(x.shape[0]):
            if x[i] > 0.5:
                ans[i] = 0; 
        return ans

    def write_file(self,path):
        result = self.func(self.data['X_test'])
        answer = self.predict(result)
        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile) 
            writer.writerow(['id','label']) 
            for i in range(answer.shape[0]):
                writer.writerow([i+1,answer[i][0]])
                
                             
dm = data_manager()

dm.read('X_train','data/X_train')#��ȡ�ļ�
dm.read('Y_train','data/Y_train')
dm.read('X_test','data/X_test')
'''
dm.read('X_train','data-test/X_train')
dm.read('Y_train','data-test/Y_train')
dm.read('X_test','data-test/X_test')
'''
dm.find_theta()
dm.write_file('output.csv')
