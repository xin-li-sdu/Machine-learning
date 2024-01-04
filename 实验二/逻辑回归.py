# coding=utf-8
import numpy as np
from numpy.linalg import inv
from scipy.special import expit
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm

mean=0
std=0   #��ʼ����ֵ�ͷ���
eps = 1e-8   #����һ��΢С�����ڶ����ʽ��ʹ���Ա�֤��ĸ������0

def partition(X, Y, verify_ratio):   #�ֳ�ѵ��������֤��
    train_size = int(len(X) * (1 - verify_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]

def func(X, w, b):
    return np.clip(expit(np.matmul(X, w )+b), 1e-8, 1 - (1e-8)) ##��ֹexp�����e���ݴη�

def predict(X, w, b):  #Ԥ�⺯��
    return np.round(func(X, w, b)).astype(int)
    
def accuracy(Y_pred, Y_label):#������ȷ��
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc

lamb=0.1  #���򻯲���

def loss(y_pred, Y_label,w):#������
    #r= lamb * np.sum(w ** 2) / (2 * Y_label.shape[0])   #��loss���������
    cross_entropy = -np.dot(Y_label, np.log(y_pred)) - np.dot((1 - Y_label), np.log(1 - y_pred))
   # cross_entropy+=r
    return cross_entropy

def gradient(X, Y_label, w, b):#�߼��ع��½��ݶ�
    y_pred = func(X, w, b)
    pred_error = Y_label - y_pred
    w_grad = -np.sum(pred_error * X.T, 1)
    b_grad = -np.sum(pred_error)
    return w_grad, b_grad

def loss_show(train_loss,verify_loss,train_acc,verify_acc):#��ͼ
    # Loss����
    plt.plot(train_loss)
    plt.plot(verify_loss)
    plt.title('Loss')
    plt.legend(['train_loss', 'verify_loss'])
    plt.show()

    # Accuracy����
    plt.plot(train_acc)
    plt.plot(verify_acc)
    plt.title('Accuracy')
    plt.legend(['train_acc', 'verify_acc'])
    plt.show()

def find_theta():
    with open('./data/X_train',newline = '') as f:
        f.readline();X_train = np.array([line.strip().split(',')[1:] for line in f], dtype = float)
        #���ȡ����ֵ��ֵ��תΪ1��,div=1*106
        global mean,std
        mean = np.mean(X_train,axis = 0).reshape(1,-1)
        #���ȡ����ֵ�󷽲תΪ1��,div=1*106
        std = np.std(X_train,axis = 0).reshape(1,-1)+eps
        for i in range(X_train.shape[0]):
           X_train[i,:] = (X_train[i,:]-mean)/(std)
        
    with open('./data/Y_train',newline = '') as f:
        f.readline();Y_train = np.array([line.strip() for line in f], dtype = float)
        #��ȡy����һ���������ʦ���Ĵ�������˾���ת�ã����Դ������ѵ���ٶ�
    # �����ݷֳ�ѵ��������֤��
    verify_ratio = 0.1  #��֤������
    X_train, Y_train, X_verify, Y_verify = partition(X_train, Y_train, verify_ratio = verify_ratio)
    train_size = X_train.shape[0]
    #print(X_train.shape)
    verify_size = X_verify.shape[0]
    data_dim = X_train.shape[1]
    
    w = np.zeros((data_dim,))	# ��ʼ��Ȩ��w��b
    b = np.zeros((1,))
    iteration = 500  # ѵ��ʱ�ĳ�����
    learning_rate = 1#ѧϰ��


    # ����ÿ��iteration��loss��accuracy
    train_loss = []
    verify_loss = []
    train_acc = []
    verify_acc = []
    w_adagrad = np.zeros((data_dim,))
    b_adagrad =np.zeros((1,))
    # ����ѵ��
    for i in tqdm(range(iteration)):
        
        w_grad, b_grad = gradient(X_train, Y_train, w, b)
        #w_grad += 2*learning_rate*lamb * w  #w�ݶȼ��������
        # �ݶ��½�������
        w_adagrad += (w_grad) ** 2
        b_adagrad += (b_grad) ** 2
        
        w = w - learning_rate/np.sqrt(w_adagrad+eps) * w_grad
        b = b - learning_rate/np.sqrt(b_adagrad+eps) * b_grad
        #���²���
        y_train_pred = func(X_train, w, b)
        Y_train_pred = np.round(y_train_pred)
        y_verify_pred = func(X_verify, w, b)
        Y_verify_pred = np.round(y_verify_pred)#Ԥ��
        if i %10 == 0:        
        # ����ѵ��������֤����loss��accuracy
            train_acc.append(accuracy(Y_train_pred, Y_train))
            train_loss.append(loss(y_train_pred, Y_train,w) / train_size)

            verify_acc.append(accuracy(Y_verify_pred, Y_verify))
            verify_loss.append(loss(y_verify_pred, Y_verify,w) / verify_size)

    print('Training loss: {}'.format(train_loss[-1]))
    print('Verify loss: {}'.format(verify_loss[-1])) 
    print('Training accuracy: {}'.format(train_acc[-1]))
    print('Verify accuracy: {}'.format(verify_acc[-1]))
    #loss_show(train_loss,verify_loss,train_acc,verify_acc)#��ͼ
    sort = np.argsort(np.abs(w))[::-1]#��w�ľ���ֵ�Ӵ�С����Ȩ��Խ��Ӱ��Խ��
    with open('./data/X_test') as f:
        content = f.readline().strip('\n').split(',')
    features = np.array(content)
    print("")
    print("Characteristics with the greatest impact is:",features[sort[0]], w[sort[0]])#���Ȩֵ��������
    return w,b


if __name__ == '__main__':
    w,b=find_theta()
    with open('./data/X_test',newline = '') as f:
         f.readline();X_test = np.array([line.strip().split(',')[1:] for line in f], dtype = float)
         for i in range(X_test.shape[0]):
                X_test[i,:] = (X_test[i,:]-mean)/(std)
    # Ԥ����Լ�
    predictions = predict(X_test, w, b)
    # ���浽output_logistic.csv
    with open('output.csv', 'w') as f:
        f.write('id,label\n')
        for i, label in  enumerate(predictions):
            f.write('{},{}\n'.format(i, label))

