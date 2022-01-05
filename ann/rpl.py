import numpy as np                                  # importing dataset
import pandas as pd
from tensorflow.keras.models import Sequential
from keras.layers import Dense
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split

np.random.seed(10)                                  # sequence random no 

df = pd.read_csv("RPL.csv")                         # importing data set

# data cleansing and eda part ******************************************
df.info()                                           # checking for data types and null values
df.describe()                                       # checking for mean ,median and sd                       
df=df.drop(["RowNumber","Surname"], axis = 1)       # dummy column for categorical column  
df.duplicated().sum()
df.nunique()
df=pd.get_dummies(df,drop_first=(True))
plt.boxplot(df)                                      # checking outliers
df.corr()                                            # checking correlation
df.skew()                                            # checking skewness
df.kurtosis()                                        # checking kurtosis

#predictors
x =df.loc[:, df.columns != 'Exited'].values.astype("float32")   
#target
y= df["Exited"].values.astype("float32")                        

#splitting data to train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30,random_state=20)

scaler = StandardScaler()                    # scaling by standard scalar

x_train_scaled= scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)
                                     
# model building *********************************************************************

# model initialising and input layers
model = Sequential()
model.add(Dense(512, input_dim =12,activation = 'sigmoid'))

# hidden layers
model.add(Dense(256,activation='sigmoid'))
model.add(Dense(128,activation='sigmoid'))
model.add(Dense(64,activation='sigmoid'))
model.add(Dense(32,activation='sigmoid'))
model.add(Dense(16,activation='sigmoid'))

#output layer
model.add(Dense(1, activation='sigmoid'))

# loss function
mse = MeanSquaredError()
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

history = model.fit(x_train_scaled, y_train, epochs=500 , batch_size= 50 ,validation_split=0.3)

# Plotting the history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history[ 'accuracy' ])
plt.plot(history.history[ 'val_accuracy' ])
plt.title( 'model accuracy' )
plt.ylabel( 'accuracy' )
plt.xlabel( 'epoch' )
plt.legend([ 'train' , 'test' ], loc= 'lower right' )
plt.show()

#predictions for test
test_profit_pred = model.predict(x_test_scaled)

test_res = y_test-test_profit_pred
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse

#predictions for train
train_profit_pred = model.predict(x_train_scaled)

train_res = y_train-train_profit_pred
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse





