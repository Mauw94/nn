import numpy as np

def conv2d(X, filters, biases):
    # print(X.shape)
    # X shape: (batch_size, height, width, in_channels)
    # filters shape: (num_filters, filter_height, filter_width, in_channels)
    batch_size, H, W, C = X.shape
    F, fH, fW, _ = filters.shape
    out_H = H - fH + 1
    out_W = W - fW + 1
    output = np.zeros((batch_size, out_H, out_W, F))

    for b in range(batch_size):
        for f in range(F):
            for i in range(out_H):
                for j in range(out_W):
                    region = X[b, i:i+fH, j:j+fW, :]
                    output[b, i, j, f] = np.sum(region * filters[f]) + biases[f]
    return output

def maxpool2d(X, size=2):
    batch_size, H, W, C = X.shape
    out_H, out_W = H // size, W // size
    output = np.zeros((batch_size, out_H, out_W, C))
    for b in range(batch_size):
        for c in range(C):
            for i in range(out_H):
                for j in range(out_W):
                    region = X[b, i*size:(i+1)*size, j*size:(j+1)*size, c]
                    output[b, i, j, c] = np.max(region)
    return output

def flatten(X):
    return X.reshape(X.shape[0], -1)