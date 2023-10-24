import numpy as np
import torch
from numpy.linalg import inv, svd
from scipy.stats import zscore
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold

device = "cuda" if torch.cuda.is_available() else "cpu"

# error functions
def corr(X, Y):
    return np.mean(zscore(X) * zscore(Y), axis=0)


def R2(Pred, Real):
    SSres = np.mean((Real - Pred) ** 2, 0)
    SStot = np.var(Real, 0)
    return np.nan_to_num(1 - SSres / SStot)

def R2r(Pred, Real):
    R2rs = R2(Pred, Real)
    ind_neg = R2rs < 0
    R2rs = np.abs(R2rs)
    R2rs = np.sqrt(R2rs)
    R2rs[ind_neg] *= - 1
    return R2rs

# functions on getting the weights
def ridge(X, Y, lmbda):
    # so i think the x axis is the timestamp features from bert and y is the fMRI features
    # np eye : Return a 2-D array with ones on the diagonal and zeros elsewhere. probably this is for the slope
    # T is making it possible for X and Y to be used in the dot method because dot methods does matrix multiplication and
    # in order for that to happen the rows of one matrix needs to match the number of columns of the other matrix
    # X.T.dot(X) sum of square residuals . you do the transpose, so you can use the dot product on itself
    # lmbda*np.eye(X.shape[1]) then you add to the square residuals the lambda value times the slope^2 which is a matrix of 40x40 with ones and zeros
    # then you invert the whole thing back, so you can do a dot product with the other matrix
    # the other matrix is the dot product of extracted features and fMRI data
    # X.T is used, so it can be multiple Y
    # so essentially the outer dot product is giving a product of matrix multiplication of the ridge regression model and the product of extracted features and fMRI features
    # this is probably the weights of the model
    return np.dot(inv(X.T.dot(X) + lmbda * np.eye(X.shape[1])), X.T.dot(Y))


def ridge_sk(X, Y, lmbda):
    # same thing as ridge function but using sklearn library instead doing it manually
    rd = Ridge(alpha=lmbda, random_state=42)
    rd.fit(X, Y)
    return rd.coef_.T

def ridgeCV_sk(X, Y, lmbdas):
    rd = RidgeCV(alphas=lmbdas)
    rd.fit(X, Y)
    return rd.coef_.T

def ridge_svd(X, Y, lmbda):
    U, s, Vt = svd(X, full_matrices=False)
    d = s / (s ** 2 + lmbda)
    return np.dot(Vt, np.diag(d).dot(U.T.dot(Y)))


# functions that get the heights and output an error score for each lambda for each set of weights
def ridge_by_lambda(X, Y, Xval, Yval, lambdas=np.array([0.1, 1, 10, 100, 1000])):
    # x,y,xval,yval are the same thing just split into training and testing is to calculate the errors
    #
    error = np.zeros((lambdas.shape[0], Y.shape[1]))
    # for every lambda calculate an error
    # lambdas are an array of values
    for idx, lmbda in enumerate(lambdas):
        # weights = ridge(X, Y, lmbda)
        weights = ridge(X, Y, lmbda)
        # error for every lambda is calculated
        # 1-R2
        # get back an error for every lambda
        # np.dot(Xval,weights) these are the predictions. Essential is the question if we combine the model weights with the extracted features from testing can we predict
        # accurately the fMRI recordings
        # Yval these are the labels
        # error[idx] = 1 - R2(np.dot(Xval, weights), Yval)
        error[idx] = 1 - R2(Xval @ weights, Yval)
    return error


def ridge_by_lambda_sk(X, Y, Xval, Yval, lambdas=np.array([0.1, 1, 10, 100, 1000])):
    error = np.zeros((lambdas.shape[0], Y.shape[1]))
    for idx, lmbda in enumerate(lambdas):
        weights = ridge_sk(X, Y, lmbda)
        error[idx] = 1 - R2(np.dot(Xval, weights), Yval)
    return error


def ridge_by_lambda_svd(X, Y, Xval, Yval, lambdas=np.array([0.1, 1, 10, 100, 1000])):
    error = np.zeros((lambdas.shape[0], Y.shape[1]))
    # svd stands for singular value decomposition
    # decompose matrix X into the product of 3 different matrices = U ,S,V(transpose)
    # so if a has shape MxN then  U = MxM S=MxN and Vt=NxN
    # U and Vt are unitary matrices. if we multiply one of these matrices by its transpose (or the other way around), the result equals the identity matrix.
    # On the other hand, the matrix S
    # is diagonal, and it stores non-negative singular values ordered by relevance.
    U, s, Vt = svd(X, full_matrices=False)
    for idx, lmbda in enumerate(lambdas):
        # S matrix is diagonal, only the first n row diagonal values are worth keeping.
        # s columns are order in descending order
        # they show how important each column is for U and Vt

        # Indeed the last n rows of S are filled with 0s. For this reason, it is very common to keep only the first r×r non-negative diagonal values of S
        # we can cut off some rows from s for being 0 or close to 0 because that means they are not that important
        # I am assuming the line below does that. Cutting off the lines based on the slop(identity matrix) and lambda
        # because in ridge regression the penalty is applied by slope^2+lambda
        # so cutting the rows by the penalty
        # that gives a new identity matrix
        d = s / (s ** 2 + lmbda)
        # d = s divided by s^2 + lambda
        # this equation will be a dot product of
        # 1. Vt
        # 2. second input is a dot product of
        #   the diagonal of d and U transpose times Y
        # np.diag Extract a diagonal or construct a diagonal array.
        # so basically because d is just a list in order to use it in matrix multiplication
        # np.diag constructs a matrix of 40x40 with everything zero except the diagonal.
        # d is the identity matrix
        # Vt and U represents the important columns based on penalty
        # so bellow is the ridge regression of how far the timestamped bert features are (Vt) using the regression model with the Identity matrix
        # times the timestamped data and the fMRI data.-`
        weights = np.dot(Vt, np.diag(d).dot(U.T.dot(Y)))
        error[idx] = 1 - R2(np.dot(Xval, weights), Yval)
    return error


def kernel_ridge(X, Y, lmbda):
    # so the kernel ridge regression is defined as
    # xTranspose times X times inverted(XTranspose times X + lambda times identity matrix) times Y
    # X.T is the first transpose XTranspose
    # inv is the inverted product
    # X.dot(X.T) is the X times X transpose
    # np.eye is the identity matrix
    # lmbda * np.eye  lambda times identity matrix
    # so the whole thing gives us the ridge regression with kernel
    return np.dot(X.T.dot(inv(X.dot(X.T) + lmbda * np.eye(X.shape[0]))), Y)


def kernel_ridge_by_lambda(X, Y, Xval, Yval, lambdas=np.array([0.1, 1, 10, 100, 1000])):
    error = np.zeros((lambdas.shape[0], Y.shape[1]))
    for idx, lmbda in enumerate(lambdas):
        weights = kernel_ridge(X, Y, lmbda)
        error[idx] = 1 - R2(np.dot(Xval, weights), Yval)
    return error


def kernel_ridge_svd(X, Y, lmbda):
    U, s, Vt = svd(X.T, full_matrices=False)
    d = s / (s ** 2 + lmbda)
    return np.dot(np.dot(U, np.diag(d).dot(Vt)), Y)


def kernel_ridge_by_lambda_svd(X, Y, Xval, Yval, lambdas=np.array([0.1, 1, 10, 100, 1000])):
    error = np.zeros((lambdas.shape[0], Y.shape[1]))
    # using X transpose instead of X
    # getting X^2 for the kernel
    U, s, Vt = svd(X.T, full_matrices=False)
    for idx, lmbda in enumerate(lambdas):
        # creating the new identity matrix using the ridge regression penalty to cut off the unimportant rows.
        d = s / (s ** 2 + lmbda)
        # ses how far from is the timestamped(U,Vt) bert features from the fMRI recordings(Y)
        # np.dot(U,np.diag(d).dot(Vt)) this is the ridge regression model
        # combining the important columns based on the new identity matrix
        weights = np.dot(np.dot(U, np.diag(d).dot(Vt)), Y)
        error[idx] = 1 - R2(np.dot(Xval, weights), Yval)
    return error


# main loop that utilises the above functions
def cross_val_ridge(train_features, train_data, n_splits=10,
                    lambdas=np.array([10 ** i for i in range(-6, 10)]),
                    method='plain',
                    do_plot=False):
    ridge_1 = dict(plain=ridge_by_lambda,
                   svd=ridge_by_lambda_svd,
                   kernel_ridge=kernel_ridge_by_lambda,
                   kernel_ridge_svd=kernel_ridge_by_lambda_svd,
                   ridge_sk=ridge_by_lambda_sk)[method]
    ridge_2 = dict(plain=ridge,
                   svd=ridge_svd,
                   kernel_ridge=kernel_ridge,
                   kernel_ridge_svd=kernel_ridge_svd,
                   ridge_sk=ridge_sk)[method]

    n_voxels = train_data.shape[1]
    nL = lambdas.shape[0]
    r_cv = np.zeros((nL, train_data.shape[1]))

    kf = KFold(n_splits=n_splits)
    # start_t = time.time()
    for icv, (trn, val) in enumerate(kf.split(train_data)):
        # print('ntrain = {}'.format(train_features[trn].shape[0]))
        cost = ridge_1(train_features[trn], train_data[trn],
                       train_features[val], train_data[val],
                       lambdas=lambdas)
        if do_plot:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(cost, aspect='auto')
        # get the cost for every lambda value
        r_cv += cost
        # if icv%3 ==0:
        #    print(icv)
        # print('average iteration length {}'.format((time.time()-start_t)/(icv+1)))
    if do_plot:
        plt.figure()
        plt.imshow(r_cv, aspect='auto', cmap='RdBu_r')
    # get the best run index (lambda) for every voxel
    argmin_lambda = np.argmin(r_cv, axis=0)
    weights = np.zeros((train_features.shape[1], train_data.shape[1]))
    for idx_lambda in range(lambdas.shape[0]):  # this is much faster than iterating over voxels!
        idx_vox = argmin_lambda == idx_lambda
        # get the best possible prediction for a different lambda
        if any(idx_vox):
            weights[:, idx_vox] = ridge_2(train_features, train_data[:, idx_vox], lambdas[idx_lambda])
    if do_plot:
        plt.figure()
        plt.imshow(weights, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)

    return weights, np.array([lambdas[i] for i in argmin_lambda])
