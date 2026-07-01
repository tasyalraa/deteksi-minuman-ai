# utils.py
import numpy as np

def preprocess(img):
    img = img.resize((224,224))
    img = np.array(img)/255.0
    return np.expand_dims(img, axis=0)