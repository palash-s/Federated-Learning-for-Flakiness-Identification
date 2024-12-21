import os
import pickle
import time
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import flast
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, precision_score, recall_score

def build_neural_network(input_shape):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def preprocess_data(dataPointsList, dataLabelsList):
    dataPointsList = np.array([dp.flatten() for dp in dataPointsList])
    dataLabelsList = np.array(dataLabelsList)
    return dataPointsList, dataLabelsList

def create_tf_dataset(dataPointsList, dataLabelsList):
    dataset = tf.data.Dataset.from_tensor_slices((dataPointsList, dataLabelsList))
    dataset = dataset.batch(32)
    return dataset


def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. This is equivalent to scaled avg of the weights'''
    avg_grad = list()
    # Get the average grad across all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
    return avg_grad

def test_model(X_test, Y_test, model, comm_round):
    cce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    logits = model.predict(X_test)
    loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc, loss))
    return acc, loss

def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. This is equivalent to scaled avg of the weights'''
    avg_grad = list()
    # Get the average grad across all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
    return avg_grad

def test_model(X_test, Y_test, model, comm_round):
    cce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    logits = model.predict(X_test)
    loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc, loss))
    return acc, loss



@tf.function
def predict(model, data):
    return model(data, training=False)

def flastFederatedLearning(outDir, projectBasePath, projectName, kf, dim, eps):
    v0 = time.perf_counter()
    dataPointsFlaky, dataPointsNonFlaky = flast.getDataPointsInfo(projectBasePath, projectName)
    dataPoints = dataPointsFlaky + dataPointsNonFlaky
    Z = flast.flastVectorization(dataPoints, dim=dim, eps=eps)
    dataPointsList = np.array([Z[i].toarray() for i in range(Z.shape[0])])
    dataLabelsList = np.array([1]*len(dataPointsFlaky) + [0]*len(dataPointsNonFlaky))
    v1 = time.perf_counter()
    vecTime = v1 - v0

    # Storage calculation
    nn = (dataPointsList, dataLabelsList)
    pickleDumpNN = os.path.join(outDir, "flast-nn.pickle")
    with open(pickleDumpNN, "wb") as pickleFile:
        pickle.dump(nn, pickleFile)
    storage = os.path.getsize(pickleDumpNN)
    os.remove(pickleDumpNN)

    avgP, avgR = 0, 0
    avgTPrep, avgTPred = 0, 0
    avgFlakyTrain, avgNonFlakyTrain, avgFlakyTest, avgNonFlakyTest = 0, 0, 0, 0
    successFold, precisionFold = 0, 0
    count = 0

    for (trnIdx, tstIdx) in kf.split(dataPointsList, dataLabelsList):
        trainData, testData = dataPointsList[trnIdx], dataPointsList[tstIdx]
        trainLabels, testLabels = dataLabelsList[trnIdx], dataLabelsList[tstIdx]
        if sum(trainLabels) == 0 or sum(testLabels) == 0:
            print("Skipping fold...")
            print(" Flaky Train Tests", sum(trainLabels))
            print(" Flaky Test Tests", sum(testLabels))
            continue

        successFold += 1
        avgFlakyTrain += sum(trainLabels)
        avgNonFlakyTrain += len(trainLabels) - sum(trainLabels)
        avgFlakyTest += sum(testLabels)
        avgNonFlakyTest += len(testLabels) - sum(testLabels)

        # Prepare the data in the right format for federated learning
        trainData, trainLabels = preprocess_data(trainData, trainLabels)
        testData, testLabels = preprocess_data(testData, testLabels)
        train_dataset = create_tf_dataset(trainData, trainLabels)
        test_dataset = create_tf_dataset(testData, testLabels)

        # Initialize the global model
        global_model = build_neural_network(input_shape=(trainData.shape[1],))

        # Simulate federated learning
        num_clients = 5
        client_data = np.array_split(trainData, num_clients)
        client_labels = np.array_split(trainLabels, num_clients)
        client_weights = []

        for client in range(num_clients):
            local_model = build_neural_network(input_shape=(trainData.shape[1],))
            local_model.set_weights(global_model.get_weights())
            local_model.fit(client_data[client], client_labels[client], epochs=1, batch_size=32, verbose=0)
            client_weights.append(local_model.get_weights())

        # Aggregate client weights
        new_weights = sum_scaled_weights(client_weights)
        global_model.set_weights(new_weights)

        # Evaluate the global model
        test_metrics = global_model.evaluate(testData, testLabels, verbose=0)
        print(f'Test metrics: {test_metrics}')

        preparationTime = (vecTime * len(trainData) / len(dataPoints)) + test_metrics[0]
        predictionTime = (vecTime / len(dataPoints)) + (test_metrics[0] / len(testData))
        precision = precision_score(testLabels, (predict(global_model, testData) > 0.5).numpy().astype("int32"), zero_division=1)
        recall = recall_score(testLabels, (predict(global_model, testData) > 0.5).numpy().astype("int32"), zero_division=1)
        count += 1
        print("count is ", count)

        print(precision, recall)
        if precision != "-":
            precisionFold += 1
            avgP += precision
        avgR += recall
        avgTPrep += preparationTime
        avgTPred += predictionTime

    if precisionFold == 0:
        avgP = "-"
    else:
        avgP /= precisionFold
    avgR /= successFold
    avgTPrep /= successFold
    avgTPred /= successFold
    avgFlakyTrain /= successFold
    avgNonFlakyTrain /= successFold
    avgFlakyTest /= successFold
    avgNonFlakyTest /= successFold

    num_predicted_flaky = sum((predict(global_model, testData) > 0.5).numpy().astype("int32"))
    num_predicted_non_flaky = len(testLabels) - num_predicted_flaky
    print("Number of predicted flaky labels (1):", num_predicted_flaky)
    print("Number of predicted non-flaky labels (0):", num_predicted_non_flaky)
    
    # Print total predicted classes and list of predicted classes
    print("Total predicted classes:", len(testLabels))
    print("List of predicted classes:", (predict(global_model, testData) > 0.5).numpy().astype("int32"))

    return (avgFlakyTrain, avgNonFlakyTrain, avgFlakyTest, avgNonFlakyTest, avgP, avgR, storage, avgTPrep, avgTPred, num_predicted_flaky, num_predicted_non_flaky)



if __name__ == "__main__":
    projectBasePath = "dataset"
    projectList = [
        "achilles",
        "alluxio-tachyon",
        "ambari",
        "hadoop",
        "jackrabbit-oak",
        "jimfs",
        "ninja",
        "okhttp",
        "oozie",
        "oryx",
        "spring-boot",
        "togglz",
        "wro4j",
    ]
    outDir = "results/"
    outFile = "eff-eff-FEDL.csv"
    os.makedirs(outDir, exist_ok=True)
    with open(os.path.join(outDir, outFile), "w") as fo:
        fo.write("dataset,flakyTrain,nonFlakyTrain,flakyTest,nonFlakyTest,precision,recall,storage,preparationTime,predictionTime,predictedFlaky,predictedNonFlaky\n")

    numSplit = 30
    testSetSize = 0.2
    kf = StratifiedShuffleSplit(n_splits=numSplit, test_size=testSetSize)

    # FLAST
    dim = 0  # number of dimensions (0: JL with error eps)
    eps = 0.3  # JL eps
    for projectName in projectList:
        print(projectName.upper(), "FLAST")
        (flakyTrain, nonFlakyTrain, flakyTest, nonFlakyTest, avgP, avgR, storage, avgTPrep, avgTPred, num_predicted_flaky, num_predicted_non_flaky) = flastFederatedLearning(outDir, projectBasePath, projectName, kf, dim, eps)
        with open(os.path.join(outDir, outFile), "a") as fo:
            fo.write("{},{},{},{},{},{},{},{},{},{},{},{}\n".format(projectName, flakyTrain, nonFlakyTrain, flakyTest, nonFlakyTest, avgP, avgR, storage, avgTPrep, avgTPred, num_predicted_flaky, num_predicted_non_flaky))
        print(f"Finished processing project: {projectName.upper()}")










































# import os
# import pickle
# import time
# import numpy as np
# from sklearn.model_selection import StratifiedShuffleSplit
# import flast
# import tensorflow as tf
# from tensorflow.keras import layers, models
# from sklearn.metrics import accuracy_score, precision_score, recall_score

# def build_neural_network(input_shape):
#     model = models.Sequential()
#     model.add(layers.Input(shape=input_shape))
#     model.add(layers.Dense(128, activation='relu'))
#     model.add(layers.Dense(128, activation='relu'))
#     model.add(layers.Dense(1, activation='sigmoid'))
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# def preprocess_data(dataPointsList, dataLabelsList):
#     dataPointsList = np.array([dp.flatten() for dp in dataPointsList])
#     dataLabelsList = np.array(dataLabelsList)
#     return dataPointsList, dataLabelsList

# def create_tf_dataset(dataPointsList, dataLabelsList):
#     dataset = tf.data.Dataset.from_tensor_slices((dataPointsList, dataLabelsList))
#     dataset = dataset.batch(32)
#     return dataset

# def sum_scaled_weights(scaled_weight_list):
#     '''Return the sum of the listed scaled weights. This is equivalent to scaled avg of the weights'''
#     avg_grad = list()
#     # Get the average grad across all client gradients
#     for grad_list_tuple in zip(*scaled_weight_list):
#         layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
#         avg_grad.append(layer_mean)
#     return avg_grad

# def test_model(X_test, Y_test, model, comm_round):
#     cce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#     logits = model.predict(X_test)
#     loss = cce(Y_test, logits)
#     acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
#     print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc, loss))
#     return acc, loss

# def flastFederatedLearning(outDir, projectBasePath, projectName, kf, dim, eps):
#     v0 = time.perf_counter()
#     dataPointsFlaky, dataPointsNonFlaky = flast.getDataPointsInfo(projectBasePath, projectName)
#     dataPoints = dataPointsFlaky + dataPointsNonFlaky
#     Z = flast.flastVectorization(dataPoints, dim=dim, eps=eps)
#     dataPointsList = np.array([Z[i].toarray() for i in range(Z.shape[0])])
#     dataLabelsList = np.array([1]*len(dataPointsFlaky) + [0]*len(dataPointsNonFlaky))
#     v1 = time.perf_counter()
#     vecTime = v1 - v0

#     # Storage calculation
#     nn = (dataPointsList, dataLabelsList)
#     pickleDumpNN = os.path.join(outDir, "flast-nn.pickle")
#     with open(pickleDumpNN, "wb") as pickleFile:
#         pickle.dump(nn, pickleFile)
#     storage = os.path.getsize(pickleDumpNN)
#     os.remove(pickleDumpNN)

#     avgP, avgR = 0, 0
#     avgTPrep, avgTPred = 0, 0
#     avgFlakyTrain, avgNonFlakyTrain, avgFlakyTest, avgNonFlakyTest = 0, 0, 0, 0
#     successFold, precisionFold = 0, 0
#     count = 0

#     for (trnIdx, tstIdx) in kf.split(dataPointsList, dataLabelsList):
#         trainData, testData = dataPointsList[trnIdx], dataPointsList[tstIdx]
#         trainLabels, testLabels = dataLabelsList[trnIdx], dataLabelsList[tstIdx]
#         if sum(trainLabels) == 0 or sum(testLabels) == 0:
#             print("Skipping fold...")
#             print(" Flaky Train Tests", sum(trainLabels))
#             print(" Flaky Test Tests", sum(testLabels))
#             continue

#         successFold += 1
#         avgFlakyTrain += sum(trainLabels)
#         avgNonFlakyTrain += len(trainLabels) - sum(trainLabels)
#         avgFlakyTest += sum(testLabels)
#         avgNonFlakyTest += len(testLabels) - sum(testLabels)

#         # Prepare the data in the right format for federated learning
#         trainData, trainLabels = preprocess_data(trainData, trainLabels)
#         testData, testLabels = preprocess_data(testData, testLabels)
#         train_dataset = create_tf_dataset(trainData, trainLabels)
#         test_dataset = create_tf_dataset(testData, testLabels)

#         # Initialize the global model
#         global_model = build_neural_network(input_shape=(trainData.shape[1],))

#         # Simulate federated learning
#         num_clients = 5
#         client_data = np.array_split(trainData, num_clients)
#         client_labels = np.array_split(trainLabels, num_clients)
#         client_weights = []

#         for client in range(num_clients):
#             local_model = build_neural_network(input_shape=(trainData.shape[1],))
#             local_model.set_weights(global_model.get_weights())
#             local_model.fit(client_data[client], client_labels[client], epochs=1, batch_size=32, verbose=0)
#             client_weights.append(local_model.get_weights())

#         # Aggregate client weights
#         new_weights = sum_scaled_weights(client_weights)
#         global_model.set_weights(new_weights)

#         # Evaluate the global model
#         test_metrics = global_model.evaluate(testData, testLabels, verbose=0)
#         print(f'Test metrics: {test_metrics}')

#         preparationTime = (vecTime * len(trainData) / len(dataPoints)) + test_metrics[0]
#         predictionTime = (vecTime / len(dataPoints)) + (test_metrics[0] / len(testData))
#         precision = precision_score(testLabels, (global_model.predict(testData) > 0.5).astype("int32"))
#         recall = recall_score(testLabels, (global_model.predict(testData) > 0.5).astype("int32"))
#         count += 1
#         print("count is ", count)

#         print(precision, recall)
#         if precision != "-":
#             precisionFold += 1
#             avgP += precision
#         avgR += recall
#         avgTPrep += preparationTime
#         avgTPred += predictionTime

#     if precisionFold == 0:
#         avgP = "-"
#     else:
#         avgP /= precisionFold
#     avgR /= successFold
#     avgTPrep /= successFold
#     avgTPred /= successFold
#     avgFlakyTrain /= successFold
#     avgNonFlakyTrain /= successFold
#     avgFlakyTest /= successFold
#     avgNonFlakyTest /= successFold

#     num_predicted_flaky = sum((global_model.predict(testData) > 0.5).astype("int32"))
#     num_predicted_non_flaky = len(testLabels) - num_predicted_flaky
#     print("Number of predicted flaky labels (1):", num_predicted_flaky)
#     print("Number of predicted non-flaky labels (0):", num_predicted_non_flaky)
    
#     # Print total predicted classes and list of predicted classes
#     print("Total predicted classes:", len(testLabels))
#     print("List of predicted classes:", (global_model.predict(testData) > 0.5).astype("int32"))

#     return (avgFlakyTrain, avgNonFlakyTrain, avgFlakyTest, avgNonFlakyTest, avgP, avgR, storage, avgTPrep, avgTPred, num_predicted_flaky, num_predicted_non_flaky)


# if __name__ == "__main__":
#     projectBasePath = "dataset"
#     projectList = [
#         "achilles",
#         "alluxio-tachyon",
#         "ambari",
#         "hadoop",
#         "jackrabbit-oak",
#         "jimfs",
#         "ninja",
#         "okhttp",
#         "oozie",
#         "oryx",
#         "spring-boot",
#         "togglz",
#         "wro4j",
#     ]
#     outDir = "results/"
#     outFile = "eff-eff-FEDL.csv"
#     os.makedirs(outDir, exist_ok=True)
#     with open(os.path.join(outDir, outFile), "w") as fo:
#         fo.write("dataset,flakyTrain,nonFlakyTrain,flakyTest,nonFlakyTest,precision,recall,storage,preparationTime,predictionTime,predictedFlaky,predictedNonFlaky\n")

#     numSplit = 30
#     testSetSize = 0.2
#     kf = StratifiedShuffleSplit(n_splits=numSplit, test_size=testSetSize)

#     # FLAST
#     dim = 0  # number of dimensions (0: JL with error eps)
#     eps = 0.3  # JL eps
#     for projectName in projectList:
#         print(projectName.upper(), "FLAST")
#         (flakyTrain, nonFlakyTrain, flakyTest, nonFlakyTest, avgP, avgR, storage, avgTPrep, avgTPred, num_predicted_flaky, num_predicted_non_flaky) = flastFederatedLearning(outDir, projectBasePath, projectName, kf, dim, eps)
#         with open(os.path.join(outDir, outFile), "a") as fo:
#             fo.write("{},{},{},{},{},{},{},{},{},{},{},{}\n".format(projectName, flakyTrain, nonFlakyTrain, flakyTest, nonFlakyTest, avgP, avgR, storage, avgTPrep, avgTPred, num_predicted_flaky, num_predicted_non_flaky))
#         print(f"Finished processing project: {projectName.upper()}")