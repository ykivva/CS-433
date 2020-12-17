import tensorflow.keras.backend as K

#custom losses

def soft_dice_loss(y_true, y_pred, smooth = 1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union_approx = K.sum(y_true_f) + K.sum(y_pred_f)
    inters_oer_union_approx = (2 * intersection + smooth) / (union_approx + smooth)
    return 1 -  inters_oer_union_approx


def weighted_binary_crossentropy(y_true, y_pred, weight=4):
    '''
    weight: int
        weight to be set to road labels while retaining 1 weight for non-road labels
    '''
    weights = y_true * (weight-1) + 1
    weighted_bce = K.binary_crossentropy(y_true, y_pred) * weights
    return K.mean(weighted_bce)


def total_variation(image, norm='l2'):
    x_diff = K.abs(image[:,1:,:,:] - image[:,:-1,:,:])
    y_diff = K.abs(image[:,:,1:,:] - image[:,:,:-1,:])
    res = None
    if norm == 'l1':
        res = K.mean(x_diff)+K.mean(y_diff)
    elif norm == 'l2':
        res = K.sqrt(K.mean(K.square(x_diff))) + K.sqrt(K.mean(K.square(x_diff)))
    else:
        raise Exception('Invalid norm value')
    return res


class SoftDiceLossRegularized():
    
    def __init__(self, lambd = 1e-2, smooth = 1, norm='l2'):
        
        self.lambd = lambd
        self.smooth = smooth
        self.norm = norm
    
    def soft_dice_loss_regularized(self, y_true, y_pred):
        
        return soft_dice_loss(y_true, y_pred, self.smooth) + \
            self.lambd * total_variation(y_pred, self.norm)


class WeightedBCERegularized():

    def __init__(self, lambd = 1e-2, weight = 4, norm='l2'):
        
        self.lambd = lambd
        self.weight = weight
        self.norm = norm
    
    def weighted_bce_regularized(self, y_true, y_pred):
        
        return weighted_binary_crossentropy(y_true, y_pred, self.weight) + \
            self.lambd * total_variation(y_pred, self.norm)


#custom metrics

def F1_score(y_true, y_pred, delta=1e-8):
    y_pred = K.round(y_pred)
    TP = K.sum(y_true * y_pred)
    FP = K.sum((1-y_true) * y_pred)
    FN = K.sum(y_true * (1-y_pred))
    precision = TP / (TP + FP + delta)
    recall = TP / (TP + FN + delta)
    f1_score = 2 * precision * recall / (precision + recall + delta) 
    return f1_score


class TV():
    
    def __init__(self, lambd = 1e-2, norm='l2'):
        
        self.lambd = lambd
        self.norm = norm
    
    def tv(self, y_true, y_pred):
        
        return self.lambd * total_variation(y_pred, self.norm)