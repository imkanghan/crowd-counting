from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Activation,LSTM
from keras.optimizers import SGD
import numpy as np
import scipy.io as sci
from features2XY import features2XY

data_dir = 'data/features_UCF.mat'
data = sci.loadmat(data_dir)
features = data['features'][0]
counts = data['counts'][0]

partition = sci.loadmat('data/partition_UCF.mat')['partition']
predictions = []

for index, test_index in enumerate(partition):
    X_test, Y_test = features2XY(features[test_index], counts[test_index])

    # load trained model from disk
    json_file = open('model/model' + str(index) + '_UCF.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model/model" + str(index) + "_UCF.h5")
    print("Loaded model from disk")

    print(model.summary())
    # exit() 

    model.compile(optimizer='Adam',
                    loss='mean_squared_error',
                    metrics=['mean_absolute_error'])
    result = model.evaluate(X_test, Y_test, batch_size=200, verbose=1, sample_weight=None)
    print(result)

    p = model.predict(X_test, batch_size=200, verbose=0)
    predictions.append(p)

sci.savemat('data/predictions_UCF.mat', {'predictions':predictions})

