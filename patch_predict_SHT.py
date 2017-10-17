# predict patch image's count using trained fully connected model

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Activation,LSTM
from keras.optimizers import SGD
import numpy as np
import scipy.io as sci
from features2XY import features2XY

test_data_dir = 'data/test_B_SHT.mat'
print('load testing data...')
test_data = sci.loadmat(test_data_dir)
X_test, Y_test = features2XY(test_data['features'][0], test_data['counts'][0])

# load trained model from disk
json_file = open('model/model_B_SHT.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model/model_B_SHT.h5")
print("Loaded model from disk")

print(model.summary())
# exit() 

model.compile(optimizer='Adam',
              loss='mean_squared_error',
              metrics=['mean_absolute_error'])

result = model.evaluate(X_test, Y_test, batch_size=1000, verbose=1, sample_weight=None)
print(result)

predictions = model.predict(X_test, batch_size=1000, verbose=0)
sci.savemat('data/predictions_B_SHT.mat', {'predictions':predictions})
