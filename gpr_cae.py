from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler
from model import *
from train import *

def gpr(dp_train, dp_test, coeff_train):

    scaler = StandardScaler()
    scaler.fit(np.vstack((dp_train, dp_test)))

    dp_train = scaler.transform(dp_train)
    dp_test = scaler.transform(dp_test)

    coeff_pred = np.zeros((N_TEST, LATENT_DIM))

    kernel = Matern()

    for i in range(0, LATENT_DIM):
        gpr = GaussianProcessRegressor(kernel=kernel, random_state=0,
                            normalize_y=True).fit(dp_train, coeff_train[:,i])

        coeff_pred[:,i] = gpr.predict(dp_test)

    return coeff_pred

dp = np.loadtxt('data/design_params.txt')

LATENT_DIM = 10

if __name__ == "__main__":

    k = 0 #current fold index
    for train_index, test_index in kf.split(u_snap):

        u_train = u_snap[train_index, :]
        u_test = u_snap[test_index, :]

        v_train = v_snap[train_index, :]
        v_test = v_snap[test_index, :]

        dp_train = dp[train_index, :]
        dp_test = dp[test_index, :]

        u_val = u_test[N_TEST:N_VAL+N_TEST, :]
        v_val = v_test[N_TEST:N_VAL+N_TEST, :]

        u_test = u_test[0:N_TEST, :]
        u_test_unscaled = u_test + 0

        v_test = v_test[0:N_TEST, :]
        v_test_unscaled = v_test + 0

        dp_test = dp_test[0:N_TEST, :]

        u_scaler, u_train, u_val, u_test = scale_data(u_train, u_val, u_test)
        v_scaler, v_train, v_val, v_test = scale_data(v_train, v_val, v_test)

        X_train = reshape_data(u_train, v_train)
        X_val = reshape_data(u_val, v_val)
        X_test = reshape_data(u_test, v_test)

        autoencoder = Autoencoder(LATENT_DIM)

        filepath = "weights/weights_" + str(LATENT_DIM) + "_fold_" + str(k+1)
        autoencoder.load_weights(filepath=filepath)
        autoencoder.compile(optimizer=Adam(LEARNING_RATE), loss='mse')

        coeff_train = autoencoder.encoder(X_train).numpy()
        coeff_pred = gpr(dp_train, dp_test, coeff_train)

        pred = autoencoder.decoder(coeff_pred).numpy()

        u_pred, v_pred = collapse_data(pred)

        #unscale inputs
        u_pred = u_scaler.inverse_transform(u_pred)
        v_pred = v_scaler.inverse_transform(v_pred)

        mean_err_u = calc_err(u_test_unscaled, u_pred)
        mean_err_v = calc_err(v_test_unscaled, v_pred)

        k += 1
