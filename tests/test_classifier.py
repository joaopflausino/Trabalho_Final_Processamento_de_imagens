import pytest
import numpy as np
import os
from classifiers.mlp_classifier import trainMLP, getFeatures, getLabels
from classifiers.svm_classifier import trainSVM
from classifiers.rf_classifier import trainRandomForest

from classifiers.run_all_classifiers import main as run_classifiers
from featureExtractors.orb_FeatureExtraction import main as orb_main


@pytest.fixture
def dummy_data():
    #dataset sintético
    X = np.random.rand(10, 5)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    return X, y

def test_mlp_training(dummy_data):
    # Teste do treinamento de MLP
    X, y = dummy_data
    model = trainMLP(X, y)
    assert model.n_iter_ > 0, "MLP training failed"

def test_svm_training(dummy_data):
    # Teste do treinamento de SVM
    X, y = dummy_data
    model = trainSVM(X, y)
    assert hasattr(model, "support_vectors_"), "SVM training failed"

def test_rf_training(dummy_data):
    # Teste do treinamento de random forest
    X, y = dummy_data
    model = trainRandomForest(X, y)
    assert len(model.estimators_) == 100, "RF training failed"

def test_feature_loading(tmp_path):
    # Teste do I/O do arquivo file para o carregamento da feature
    dummy_features = np.array([[1, 2], [3, 4]])
    feature_path = tmp_path / "features.csv"
    np.savetxt(feature_path, dummy_features, delimiter=',')
    loaded_features = getFeatures(tmp_path, "features.csv")
    assert np.array_equal(loaded_features, dummy_features), "Feature loading failed"

def test_label_loading(tmp_path):
    # Teste do I/O do arquivo file para o carregamento da label
    dummy_labels = np.array([0, 1])
    label_path = tmp_path / "labels.csv"
    np.savetxt(label_path, dummy_labels, delimiter=',', fmt='%d')
    loaded_labels = getLabels(tmp_path, "labels.csv")
    assert np.array_equal(loaded_labels, dummy_labels), "Label loading failed"
