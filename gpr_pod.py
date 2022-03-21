from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler
from model import *
from train import *

def calc_pod(snap, n_basis):

    psi, _, _ = np.linalg.svd(snap, False)

    psi = psi[:, 0:n_basis]

    coeff_mat = np.dot(psi.T, snap)

    return psi, coeff_mat.T

def gpr(dp_train, dp_test, coeff_train):
    assert N_BASIS == coeff_train.shape[1]

    scaler = StandardScaler()
    scaler.fit(np.vstack((dp_train, dp_test)))

    dp_train = scaler.transform(dp_train)
    dp_test = scaler.transform(dp_test)

    coeff_pred = np.zeros((N_TEST, N_BASIS))

    kernel = Matern()

    for i in range(0, N_BASIS):
        gpr = GaussianProcessRegressor(kernel=kernel, random_state=0,
                            normalize_y=True).fit(dp_train, coeff_train[:,i])

        coeff_pred[:,i] = gpr.predict(dp_test)

    return coeff_pred

dp = np.loadtxt('data/design_params.txt')

N_BASIS = 35

if __name__ == "__main__":

    for train_index, test_index in kf.split(u_snap):

        u_train = u_snap[train_index, :]
        u_test = u_snap[test_index, :]

        v_train = v_snap[train_index, :]
        v_test = v_snap[test_index, :]

        dp_train = dp[train_index, :]
        dp_test = dp[test_index, :]

        u_test = u_test[0:N_TEST, :]
        v_test = v_test[0:N_TEST, :]
        dp_test = dp_test[0:N_TEST, :]

        u_basis, u_coeff_train = calc_pod(u_train.T, N_BASIS)
        v_basis, v_coeff_train = calc_pod(v_train.T, N_BASIS)

        u_coeff_pred = gpr(dp_train, dp_test, u_coeff_train)
        v_coeff_pred = gpr(dp_train, dp_test, v_coeff_train)

        u_pred = np.dot(u_basis, u_coeff_pred.T)
        v_pred = np.dot(v_basis, v_coeff_pred.T)

        mean_err_u = calc_err(u_test, u_pred.T)
        mean_err_v = calc_err(v_test, v_pred.T)
