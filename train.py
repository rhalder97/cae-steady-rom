import numpy as np
import pyvista as pv
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from model import *

#load data for u and v from cases
def load_data(n_cases):
    grid = pv.UnstructuredGrid('data/case1.vtk')

    U = grid.cell_arrays['U']
    u = U[:, 0]
    v = U[:, 1]

    N = u.shape[0]
    n = int(np.sqrt(N))

    #data matrices for u and v
    u_snap = np.zeros((n_cases, N))
    v_snap = np.zeros((n_cases, N))

    u_snap[0, :] = u
    v_snap[0, :] = v

    for i in range(1, n_cases):
        fileName = 'data/case' + str(i+1) + '.vtk'
        grid = pv.UnstructuredGrid(fileName)
        U = grid.cell_arrays['U']
        u = U[:, 0]
        v = U[:, 1]
        u_snap[i, :] = u
        v_snap[i, :] = v

    return u_snap, v_snap

#min max scaling, use train and val for scaling
def scale_data(train, val, test):
    scaler = MinMaxScaler()
    scaler.fit(np.vstack((train, val)))

    train = scaler.transform(train)
    val = scaler.transform(val)
    test = scaler.transform(test)

    return scaler, train, val, test

#reshape data for use in autoencoder
def reshape_data(u_data, v_data):
    n_samples = u_data.shape[0]
    n = int(np.sqrt(u_data.shape[1]))

    X = np.zeros((n_samples, n, n, 2))
    for i in range(0, n_samples):
        X[i, :, :, 0] = np.reshape(u_data[i, :], (n, n))
        X[i, :, :, 1] = np.reshape(v_data[i, :], (n, n))

    return X

#collapse data from autoencoder into original shape
def collapse_data(data):
    n_samples = data.shape[0]
    N = data.shape[1]**2

    u_data = np.zeros((n_samples, N))
    v_data = np.zeros((n_samples, N))

    for i in range(0, n_samples):
        u_data[i, :] = data[i, :, :, 0].flatten()
        v_data[i, :] = data[i, :, :, 1].flatten()

    return u_data, v_data

#calculate relative error between data and prediction
def calc_err(data, prediction, return_mean=True):
    diff = np.abs(data - prediction)

    diff_norm = np.linalg.norm(diff, axis=1)
    data_norm = np.linalg.norm(data, axis=1)

    err = np.divide(diff_norm, data_norm)

    if return_mean is True:
        return np.mean(err)
    else:
        return err

N_CASES = 500  #total number of cases

#use 10 percent of total for training/val
F_TRAIN = 0.8
F_TEST = 0.1
F_VAL = 0.1

N_TRAIN = int(N_CASES*F_TRAIN)
N_TEST = int(N_CASES*F_TEST)
N_VAL = int(N_CASES*F_VAL)

#use 5-fold cross-validation on data
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS)

LATENT_DIM = 5
LEARNING_RATE = 3e-4
EPOCHS = 7500
BATCH_SIZE = 8
EARLY_ITERS = 500

SAVE_WEIGHTS = False

u_snap, v_snap = load_data(N_CASES)
kf.get_n_splits(u_snap)

if __name__ == "__main__":

    k = 0 #current fold index
    for train_index, test_index in kf.split(u_snap):

        u_train = u_snap[train_index, :]
        u_test = u_snap[test_index, :]

        v_train = v_snap[train_index, :]
        v_test = v_snap[test_index, :]

        u_val = u_test[N_TEST:N_VAL+N_TEST, :]
        v_val = v_test[N_TEST:N_VAL+N_TEST, :]

        u_test = u_test[0:N_TEST, :]
        u_test_unscaled = u_test + 0

        v_test = v_test[0:N_TEST, :]
        v_test_unscaled = v_test + 0

        u_scaler, u_train, u_val, u_test = scale_data(u_train, u_val, u_test)
        v_scaler, v_train, v_val, v_test = scale_data(v_train, v_val, v_test)

        X_train = reshape_data(u_train, v_train)
        X_val = reshape_data(u_val, v_val)
        X_test = reshape_data(u_test, v_test)

        autoencoder = Autoencoder(LATENT_DIM)
        autoencoder.compile(optimizer=Adam(LEARNING_RATE), loss='mse')

        callback = EarlyStopping(
            monitor='val_loss', min_delta=0, patience=EARLY_ITERS, verbose=0,
            mode='auto', baseline=None, restore_best_weights=False)

        autoencoder.fit(
            X_train,
            X_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=True,
            validation_data=(X_val, X_val),
            callbacks=[callback]
        )

        if SAVE_WEIGHTS is True:
           filepath = "weights/weights_" + str(LATENT_DIM) + "_fold_" + str(k+1)
           autoencoder.save_weights(filepath=filepath)

        #calculate projection of test data onto autoencoder
        proj = autoencoder.predict(X_test)

        u_proj, v_proj = collapse_data(proj)

        #unscale inputs
        u_proj = u_scaler.inverse_transform(u_proj)
        v_proj = v_scaler.inverse_transform(v_proj)

        mean_err_u = calc_err(u_test_unscaled, u_proj)
        mean_err_v = calc_err(v_test_unscaled, v_proj)

        k += 1
