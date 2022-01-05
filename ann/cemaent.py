import numpy as np                            # importing dataset
import pandas as pd
from tensorflow.keras.models import Sequential
from keras.layers import Dense
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split

np.random.seed(10)                           # sequence random no 

df = pd.read_csv("concrete.csv")          # importing data set

# data cleansing and eda part ******************************************
df.info()                                    # checking for data types and null values
df.describe()                                # checking for mean ,median and sd                       
df.duplicated().sum()                        # checking for duplicate records 
df.drop_duplicates()
df.nunique()
plt.boxplot(df)                              # checking outliers
df.corr()                                    # checking correlation
df.skew()                                    # checking skewness
df.kurtosis()                                # checking kurtosis

#predictors
x =df.loc[:, df.columns != 'strength'].values.astype("float32")   
#target
y= df["strength"].values.astype("float32")                        

#splitting data to train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30,random_state=20)

scaler = StandardScaler()                    # scaling by standard scalar

x_train_scaled= scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)
                                     
# model building *********************************************************************

# model initialising and input layers
model = Sequential()
model.add(Dense(512, input_dim =8,activation = 'relu'))

# hidden layers
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))

#output layer
model.add(Dense(1, activation='linear'))

# loss function
mse = MeanSquaredError()
model.compile(loss=mse,  optimizer=Adam(learning_rate=0.05), metrics=[mse])

history = model.fit(x_train_scaled, y_train, epochs=50, batch_size=20,validation_split=0.3,shuffle=False)

# Plotting the history
def plot_history(history, key):
  plt.plot(history.history[key])
  plt.plot(history.history['val_'+key])
  plt.xlabel("Epochs")
  plt.ylabel(key)
  plt.legend([key, 'val_'+key])
  plt.show()
plot_history(history, 'mean_squared_error')

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

#bar plot actual vs predicted
actual = pd.Series(y_test)
predicted =pd.Series((test_profit_pred).flatten())

df1 = pd.DataFrame(columns=['actual','predicted'])

df1['actual']=actual
df1['predicted']=predicted

df1 = df1
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()




