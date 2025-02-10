import pytest
import numpy as np
from featureExtractors.orb_FeatureExtraction import extractOrbDescriptors, trainKMeans, buildHistogram


@pytest.fixture
def test_orb_descriptor_extraction():
    dummy_image = [np.zeros((100, 100), dtype=np.uint8)]
    descriptors = extractOrbDescriptors(dummy_image)
    assert len(descriptors) == 1, "Descriptor extraction failed"

def test_kmeans_clustering():
    dummy_descriptors = [np.random.randint(0, 255, (150, 32)).astype(np.uint8)]
    kmeans, k = trainKMeans(dummy_descriptors)
    assert k == 100, "KMeans cluster count mismatch"



def test_histogram_generation():
    dummy_descriptors = [np.random.randint(0, 255, (10, 32)).astype(np.uint8)]
    kmeans, k = trainKMeans(dummy_descriptors)
    histograms = buildHistogram(dummy_descriptors, kmeans, k)
    assert histograms.shape[1] == k, "Histogram dimension mismatch"