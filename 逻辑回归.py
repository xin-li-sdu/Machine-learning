# coding=utf-8
import numpy as np
from numpy.linalg import inv
from scipy.special import expit
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm

mean=0
std=0   #初始化均值和方差
eps = 1e-8   #设置一个微小量，在多个公式中使用以保证分母不等于0

def partition(X, Y, verify_ratio):   #分成训练集和验证集
    train_size = int(len(X) * (1 - verify_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]

def func(X, w, b):
    return np.clip(expit(np.matmul(X, w )+b), 1e-8, 1 - (1e-8)) ##防止exp溢出，e的幂次方

def predict(X, w, b):  #预测函数
    return np.round(func(X, w, b)).astype(int)
    
def accuracy(Y_pred, Y_label):#计算正确率
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc

lamb=0.1  #正则化参数

def loss(y_pred, Y_label,w):#交叉熵
    #r= lamb * np.sum(w ** 2) / (2 * Y_label.shape[0])   #在loss中添加正则化
    cross_entropy = -np.dot(Y_label, np.log(y_pred)) - np.dot((1 - Y_label), np.log(1 - y_pred))
   # cross_entropy+=r
    return cross_entropy

def gradient(X, Y_label, w, b):#逻辑回归下降梯度
    y_pred = func(X, w, b)
    pred_error = Y_label - y_pred
    w_grad = -np.sum(pred_error * X.T, 1)
    b_grad = -np.sum(pred_error)
    return w_grad, b_grad

def loss_show(train_loss,verify_loss,train_acc,verify_acc):#作图
    # Loss曲线
    plt.plot(train_loss)
    plt.plot(verify_loss)
    plt.title('Loss')
    plt.legend(['train_loss', 'verify_loss'])
    plt.show()

    # Accuracy曲线
    plt.plot(train_acc)
    plt.plot(verify_acc)
    plt.title('Accuracy')
    plt.legend(['train_acc', 'verify_acc'])
    plt.show()

def find_theta():
    with open('./data/X_train',newline = '') as f:
        f.readline();X_train = np.array([line.strip().split(',')[1:] for line in f], dtype = float)
        #求读取的数值均值，转为1行,div=1*106
        global mean,std
        mean = np.mean(X_train,axis = 0).reshape(1,-1)
        #求读取的数值求方差，转为1行,div=1*106
        std = np.std(X_train,axis = 0).reshape(1,-1)+eps
        for i in range(X_train.shape[0]):
           X_train[i,:] = (X_train[i,:]-mean)/(std)
        
    with open('./data/Y_train',newline = '') as f:
        f.readline();Y_train = np.array([line.strip() for line in f], dtype = float)
        #读取y，这一步相比于老师给的代码进行了矩阵转置，可以大大提升训练速度
    # 把数据分成训练集和验证集
    verify_ratio = 0.1  #验证集比例
    X_train, Y_train, X_verify, Y_verify = partition(X_train, Y_train, verify_ratio = verify_ratio)
    train_size = X_train.shape[0]
    #print(X_train.shape)
    verify_size = X_verify.shape[0]
    data_dim = X_train.shape[1]
    
    w = np.zeros((data_dim,))	# 初始化权重w和b
    b = np.zeros((1,))
    iteration = 500  # 训练时的超参数
    learning_rate = 1#学习率


    # 保存每次iteration的loss和accuracy
    train_loss = []
    verify_loss = []
    train_acc = []
    verify_acc = []
    w_adagrad = np.zeros((data_dim,))
    b_adagrad =np.zeros((1,))
    # 迭代训练
    for i in tqdm(range(iteration)):
        
        w_grad, b_grad = gradient(X_train, Y_train, w, b)
        #w_grad += 2*learning_rate*lamb * w  #w梯度加入正则项。
        # 梯度下降法更新
        w_adagrad += (w_grad) ** 2
        b_adagrad += (b_grad) ** 2
        
        w = w - learning_rate/np.sqrt(w_adagrad+eps) * w_grad
        b = b - learning_rate/np.sqrt(b_adagrad+eps) * b_grad
        #更新参数
        y_train_pred = func(X_train, w, b)
        Y_train_pred = np.round(y_train_pred)
        y_verify_pred = func(X_verify, w, b)
        Y_verify_pred = np.round(y_verify_pred)#预测
        if i %10 == 0:        
        # 计算训练集和验证集的loss和accuracy
            train_acc.append(accuracy(Y_train_pred, Y_train))
            train_loss.append(loss(y_train_pred, Y_train,w) / train_size)

            verify_acc.append(accuracy(Y_verify_pred, Y_verify))
            verify_loss.append(loss(y_verify_pred, Y_verify,w) / verify_size)

    print('Training loss: {}'.format(train_loss[-1]))
    print('Verify loss: {}'.format(verify_loss[-1])) 
    print('Training accuracy: {}'.format(train_acc[-1]))
    print('Verify accuracy: {}'.format(verify_acc[-1]))
    #loss_show(train_loss,verify_loss,train_acc,verify_acc)#作图
    sort = np.argsort(np.abs(w))[::-1]#对w的绝对值从大到小排序，权重越大影响越大
    with open('./data/X_test') as f:
        content = f.readline().strip('\n').split(',')
    features = np.array(content)
    print("")
    print("Characteristics with the greatest impact is:",features[sort[0]], w[sort[0]])#输出权值最大的特征
    return w,b


if __name__ == '__main__':
    w,b=find_theta()
    with open('./data/X_test',newline = '') as f:
         f.readline();X_test = np.array([line.strip().split(',')[1:] for line in f], dtype = float)
         for i in range(X_test.shape[0]):
                X_test[i,:] = (X_test[i,:]-mean)/(std)
    # 预测测试集
    predictions = predict(X_test, w, b)
    # 保存到output_logistic.csv
    with open('output.csv', 'w') as f:
        f.write('id,label\n')
        for i, label in  enumerate(predictions):
            f.write('{},{}\n'.format(i, label))

