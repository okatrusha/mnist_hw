
import keras
from keras import layers
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
import sklearn.model_selection as selection
import plot_rnd 
import DigitClassification as dc
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import tensorflow as tf

dftrain = pd.read_csv('train.csv')

l = dftrain.shape[0]

X = (dftrain.iloc[:l,1:].values).astype('float32')
y = (dftrain.iloc[:l,0].values).astype('int32')

#plot_rnd.rplt(X,y)

X = X / 255.0
X = X.reshape(-1,28,28,1)

y = to_categorical(y, num_classes=10)

print(X.shape)
print(y.shape)

X_train, X_val, y_train, y_val = selection.train_test_split(X, y, test_size=0.2, random_state=42) 

print(X_train.shape)
print(X_val.shape)
#print(y_train.shape)
#print(y_test.shape)

digitClassifier = dc.DigitClassification()
digitClassifier.compile_model()

history = digitClassifier.train_model(X_train, y_train,X_val, y_val,batch_size=64, epochs=20)

history_df = pd.DataFrame(history.history)
history_df.loc[:,['loss']].plot()

#print(history.history.keys())

plt.show()

'''model.compile(optimizer='adam',
              loss='categorical_crossentropy')

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))
'''
