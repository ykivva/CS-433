import tensorflow.keras.backend as K


def F1_score(y_true, y_pred, delta=1e-8):
    y_pred = K.round(y_pred)
    TP = K.sum(y_true * y_pred)
    FP = K.sum((1-y_true) * y_pred)
    FN = K.sum(y_true * (1-y_pred))
    precision = TP / (TP + FP + delta)
    recall = TP / (TP + FN + delta)
    f1_score = 2 * precision * recall / (precision + recall + delta) 
    return f1_score
    