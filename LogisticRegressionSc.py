import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


data = pd.read_csv("customer_data.csv")
#print(data)
#print(f"null values: {data.isnull().sum()}")
#data.info()

z = (data - data.min()) / (data.max() - data.min())
#print(z)


corr1=data.corr()['purchased']
print(corr1.sort_values())
corr=data.corr()
sns.heatmap(corr,annot=True)
plt.title('Correlation Matrix', fontsize=16)
#plt.show()

X = z.drop(columns=['purchased'])
Y = z['purchased']
#print(X)
#print(Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=None)


class LogisticRegressionFromScratch():
   
    def fit(self ,x ,y , alpha , epochs=100):
        self.inter= np.ones((x.shape[0], 1))
        self.x_train = np.concatenate((self.inter, x), axis=1)
        self.y_train = y
        self.weight = np.zeros(self.x_train.shape[1])
        self.alpha = alpha
        self.epochs = epochs
        for i in range(self.epochs):
            z= self.sigmoid(self.x_train , self.weight)
            #self.loss(z  ,self.y_train)
            dw= self.gradientDescent(self.x_train , self.y_train , z)
            self.weight = self.weight - alpha * dw
        return self


    def predict(self , nx , lamda):
        self.inter= np.ones((nx.shape[0], 1))
        nx = np.concatenate( (self.inter , nx ), axis=1)
        res = self.sigmoid(nx , self.weight)
        res = res >= lamda
        y_pred = np.zeros(res.shape[0])
        for i in range(len(y_pred)):
            if res[i] == True: 
                y_pred[i] = 1
            else:
                continue
                 
        return y_pred



    def sigmoid(self , x , weight):
        
        return 1.0/(1+np.exp(-np.dot(x , weight)))

    def costFunction(y , hx):
        return -1* (np.sum(y*np.log(hx) + (1-y) * np.log(1-hx)))

    def gradientDescent(self , x , y , hx):
         return np.dot(x.T, (hx - y)) / y.shape[0]
    
    def loss(self, hx, y):
        return (-y * np.log(hx) - (1 - y) * np.log(1 - hx)).mean()
        
    def accuracy(y , hx):
        return np.sum(y==hx) / len(y)


alpha = 0
list = [0.000000001,  0.1 ,  0.01 , 0.001 , 0.0001 , 0.00001 , 0.0000001 ]
max =0
model = LogisticRegressionFromScratch()
for i in range(len(list)):
    #print(i)
    
    model.fit(x_train ,y_train  , list[i] , 10000)
    hx =model.predict(x_test , 0.5)
    acc = (sum(hx == y_test)) / hx.shape[0]
    if acc > max:
        max = acc
        alpha = list[i]
#model = LogisticRegressionFromScratch()
model.fit(x_train , y_train , alpha , 10000)
hx =model.predict(x_test , 0.5)
acc = sum(hx == y_test) / hx.shape[0]

print("accuracy:  ",acc *100)
print("alpha: ",alpha)