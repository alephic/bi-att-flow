
import numpy as np
import json

def load_json_file(path):
    with open(path) as fh:
        return json.load(fh)

def get_shared_words(xi, q):
    x_ws = set()
    x_ws.update(*([w.lower() for w in xij] for xij in xi))
    x_ws.intersection_update([w.lower() for w in q])
    return x_ws

def sign(x):
    return -1 if x < 0 else 1

def fit(data_train, shared_train):
    x = []
    y = []
    for i in data_train['idxs']:
        rx = data_train['*x'][i]
        xi = shared_train['x'][rx[0]][rx[1]]
        q = data_train['q'][i]
        yi = -1 if data_train['y'][i][0][0][0] == len(xi) else 1
        x.append(len(get_shared_words(xi, q)))
        y.append(yi)
    A = np.vstack((np.array(x), np.ones(len(x)))).T
    return np.linalg.lstsq(A, np.array(y))[0]

def test(data_test, shared_test, params):
    m, c = params
    correct = []
    for i in data_test['idxs']:
        rx = data_test['*x'][i]
        xi = shared_test['x'][rx[0]][rx[1]]
        q = data_test['q'][i]
        yi = -1 if data_test['y'][i][0][0][0] == len(xi) else 1
        yp = m * len(get_shared_words(xi, q)) + c
        correct.append(1 if sign(yp) == yi else 0)
    return sum(correct)/len(correct)

if __name__ == "__main__":
    data_train = load_json_file('data/squad/data_neg_train.json')
    shared_train = load_json_file('data/squad/shared_train.json')
    data_dev = load_json_file('data/squad/data_neg_dev.json')
    shared_dev = load_json_file('data/squad/shared_dev.json')
    params = fit(data_train, shared_train)
    print('Params:', params)
    print('Dev accuracy:', test(data_dev, shared_dev, params))