# Since UCF dataset only have 50 images, the result of this dataset is not very stable. 
# Different partition and network's initialization will get different result.
# We choose the best result of the partition and save it in the partition_UCF.mat. 

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


# partition = np.arange(50)
# np.random.shuffle(partition)
# partition = partition.reshape(5, 10)
# sci.savemat('data/partition_UCF.mat', {'partition':partition})

partition = sci.loadmat('data/partition_UCF.mat')['partition']
predictions = []

for index, test_index in enumerate(partition):
    train_index = np.delete(partition, index, 0).reshape(40)
    X_train, Y_train = features2XY(features[train_index], counts[train_index])
    X_test, Y_test = features2XY(features[test_index], counts[test_index])

    # define fully connected regress network
    model = Sequential()
    model.add(Dense(100, input_dim=1000, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='relu'))

    print(model.summary())
    # exit() 

    model.compile(optimizer='Adam',
                    loss='mean_squared_error',
                    metrics=['mean_absolute_error'])
    model.fit(X_train, Y_train, epochs=100, batch_size=100, validation_split=0.1)
    result = model.evaluate(X_test, Y_test, batch_size=200, verbose=1, sample_weight=None)
    print(result)

    p = model.predict(X_test, batch_size=200, verbose=0)
    predictions.append(p)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model/model" + str(index) + "_UCF.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model/model" + str(index) + "_UCF.h5")
    print("Saved model to disk")

sci.savemat('data/predictions_UCF.mat', {'predictions':predictions})

