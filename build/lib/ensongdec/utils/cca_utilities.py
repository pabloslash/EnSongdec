from sklearn.cross_decomposition import CCA
import numpy as np

def fit_CCA_model(X, Y, n_components):
    """
    X = [samples * features]
    Y = [samples * features]
    n_components = number of CCA dimensions
    """
    cca = CCA(n_components=n_components, max_iter=10000)
    cca.fit(X, Y)
    return cca

def X_transform_to_canonical(X, cca):
    '''Equivalent to X_c = cca.transform(X)'''
    X_norm = (X - cca._x_mean)/cca._x_std
    X_c = X_norm @ cca.x_rotations_
    return X_c

def Y_transform_to_canonical(Y, cca):
    '''Equivalent to _, Y_c = cca.transform(X, Y)'''
    Y_norm = (Y - cca._y_mean)/cca._y_std
    Y_c = Y_norm @ cca.y_rotations_
    return Y_c

def X_transform_to_Y(X, cca):
    '''Align X data to Y data'''
    X_c = X_transform_to_canonical(X, cca)
    Xy_norm_recovered = X_c @ np.linalg.inv(cca.y_rotations_)
    Xy_recovered = (Xy_norm_recovered * cca._y_std) + cca._y_mean
    return Xy_recovered

def Y_transform_to_X(Y, cca):
    '''Align Y data to X data'''
    Y_c = Y_transform_to_canonical(Y, cca)
    Yx_norm_recovered = Y_c @ np.linalg.inv(cca.x_rotations_)
    Yx_recovered = (Yx_norm_recovered * cca._x_std) + cca._x_mean
    return Yx_recovered

def X_transform_to_X(X, cca):
    '''Align X data to X data'''
    X_c = X_transform_to_canonical(X, cca)
    Xx_norm_recovered = X_c @ np.linalg.inv(cca.x_rotations_)
    Xx_recovered = (Xx_norm_recovered * cca._x_std) + cca._x_mean
    return Xx_recovered

def Y_transform_to_Y(Y, cca):
    '''Align Y data to Y data'''
    Y_c = Y_transform_to_canonical(Y, cca)
    Yy_norm_recovered = Y_c @ np.linalg.inv(cca.y_rotations_)
    Yy_recovered = (Yy_norm_recovered * cca._y_std) + cca._y_mean
    return Yy_recovered