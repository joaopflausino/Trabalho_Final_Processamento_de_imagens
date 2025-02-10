import os
import cv2
import numpy as np
from sklearn import preprocessing
from progress.bar import Bar
from sklearn.cluster import MiniBatchKMeans
import time

def main():
    mainStartTime = time.time()
    trainImagePath = './images_split/train/'
    testImagePath = './images_split/test/'
    trainFeaturePath = './features_labels/orb/train/'
    testFeaturePath = './features_labels/orb/test/'
    os.makedirs(trainFeaturePath, exist_ok=True)
    os.makedirs(testFeaturePath, exist_ok=True)
    
    print(f'[INFO] ========= TRAINING IMAGES ========= ')
    trainData = getData(trainImagePath)
    if trainData is None:
        return
    trainImages, trainLabels = trainData
    
    trainEncodedLabels, encoderClasses = encodeLabels(trainLabels)
    trainOrbDescriptors = extractOrbDescriptors(trainImages)
    kmeans, k = trainKMeans(trainOrbDescriptors)
    trainFeatures = buildHistogram(trainOrbDescriptors, kmeans, k)
    saveData(trainFeaturePath, trainEncodedLabels, trainFeatures, encoderClasses)
    
    print(f'[INFO] =========== TEST IMAGES =========== ')
    testData = getData(testImagePath)
    if testData is None:
        return
    testImages, testLabels = testData
    
    testEncodedLabels, encoderClasses = encodeLabels(testLabels)
    testOrbDescriptors = extractOrbDescriptors(testImages)
    testFeatures = buildHistogram(testOrbDescriptors, kmeans, k)
    saveData(testFeaturePath, testEncodedLabels, testFeatures, encoderClasses)
    
    elapsedTime = round(time.time() - mainStartTime, 2)
    print(f'[INFO] Code execution time: {elapsedTime}s')

def getData(path):
    images = []
    labels = []
    if not os.path.exists(path):
        print(f'[ERROR] Path does not exist: {path}')
        return None
    
    for dirpath, dirnames, filenames in os.walk(path):
        if len(filenames) > 0:
            folder_name = os.path.basename(dirpath)
            bar = Bar(f'[INFO] Getting images and labels from {folder_name}', max=len(filenames), suffix='%(index)d/%(max)d Duration:%(elapsed)ds')
            for index, file in enumerate(filenames):
                label = folder_name
                labels.append(label)
                full_path = os.path.join(dirpath, file)
                image = cv2.imread(full_path)
                if image is None:
                    print(f'[ERROR] Failed to read image: {full_path}')
                else:
                    images.append(image)
                bar.next()
            bar.finish()
    
    if not images or not labels:
        print(f'[ERROR] No images or labels found in path: {path}')
        return None
    
    return images, np.array(labels, dtype=object)

def extractOrbDescriptors(images):
    orbDescriptorsList = []
    bar = Bar('[INFO] Extracting ORB descriptors...', max=len(images), suffix='%(index)d/%(max)d  Duration:%(elapsed)ds')
    orb = cv2.ORB_create()
    for image in images:
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(image, None)
        if descriptors is not None:
            orbDescriptorsList.append(descriptors)
        bar.next()
    bar.finish()
    return np.array(orbDescriptorsList, dtype=object)

def trainKMeans(orbDescriptors):
    import time
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans

    print('[INFO] Clustering the ORB descriptors of all train images...')
    k = 100
    all_descriptors = np.vstack(orbDescriptors)
    if all_descriptors.shape[0] < k:
        print(f'[WARNING] Número de amostras ({all_descriptors.shape[0]}) menor que k ({k}). Ajustando k para {all_descriptors.shape[0]}.')
        k = all_descriptors.shape[0]
    kmeans = MiniBatchKMeans(n_clusters=k, n_init='auto', random_state=42)
    startTime = time.time()
    kmeans.fit(all_descriptors)
    elapsedTime = round(time.time() - startTime, 2)
    print(f'[INFO] Clustering done in {elapsedTime}s')
    return kmeans, k


def buildHistogram(orbDescriptors, kmeans_model, n_clusters):
    print('[INFO] Building histograms...')
    startTime = time.time()
    histogramList = []
    for i in range(len(orbDescriptors)):
        histogram = np.zeros(n_clusters)
        idx_arr = kmeans_model.predict(orbDescriptors[i])
        for d in range(len(idx_arr)):
            histogram[idx_arr[d]] += 1 
        histogramList.append(histogram)
    elapsedTime = round(time.time() - startTime, 2)
    print(f'[INFO] Histogram built in {elapsedTime}s')
    return np.array(histogramList, dtype=object)

def encodeLabels(labels):
    startTime = time.time()
    print(f'[INFO] Encoding labels to numerical labels')
    encoder = preprocessing.LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    elapsedTime = round(time.time() - startTime, 2)
    print(f'[INFO] Encoding done in {elapsedTime}s')
    return np.array(encoded_labels, dtype=object), encoder.classes_

def saveData(path, labels, features, encoderClasses):
    startTime = time.time()
    print(f'[INFO] Saving data')
    label_filename = f'{labels=}'.split('=')[0]+'.csv'
    feature_filename = f'{features=}'.split('=')[0]+'.csv'
    encoder_filename = f'{encoderClasses=}'.split('=')[0]+'.csv'
    np.savetxt(os.path.join(path, label_filename), labels, delimiter=',', fmt='%i')
    np.savetxt(os.path.join(path, feature_filename), features, delimiter=',')
    np.savetxt(os.path.join(path, encoder_filename), encoderClasses, delimiter=',', fmt='%s') 
    elapsedTime = round(time.time() - startTime, 2)
    print(f'[INFO] Saving done in {elapsedTime}s')

if __name__ == "__main__":
    main()
