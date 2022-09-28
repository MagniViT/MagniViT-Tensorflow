from sklearn.metrics import roc_auc_score, roc_curve
import tensorflow as tf
import numpy as np

def get_cam_1d(classifier, features):
    tweight = list(classifier.parameters())[-2]
    cam_maps = tf.einsum('bgf,cf->bcg', [features, tweight])
    return cam_maps
