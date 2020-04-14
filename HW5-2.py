import pandas as pd
import numpy as np

df = pd.read_excel("HW5-2.xls")
rows, columns = df.shape[0], df.shape[1]

ss = np.array(df.iloc[:,:])
q = np.random.random((4, 100))
q = np.array(q)

EPOCHS = 10000
LR = 0.005
DATA = 100

for i in range(DATA):
    # q1, q2, q3, q4 = q[0, i], q[1, i], q[2, i], q[3, i]

    for epoch in range(EPOCHS):
        loss = 0
        jac_tran = np.array([[-2 * q[2, i], 2 * q[1, i],       0],
                             [ 2 * q[3, i], 2 * q[0, i], -4 * q[1, i]],
                             [-2 * q[0, i], 2 * q[3, i], -4 * q[2, i]],
                             [ 2 * q[1, i], 2 * q[2, i],       0]])

        ss[i, :] = ss[i, :] / np.linalg.norm(ss[i, :])

        f = np.array([[2 * (q[1, i] * q[3, i] - q[0, i] * q[2, i]) - ss[i, 0]],
                      [2 * (q[0, i] * q[1, i] + q[2, i] * q[3, i]) - ss[i, 1]],
                      [2 * (0.5 - q[1, i] * q[1, i] - q[2, i] * q[2, i]) - ss[i, 2]]])

        grad = np.dot(jac_tran, f)
        update = - LR * grad
        q[:, i] += update[:, 0]
        loss = np.sum(f)
        if (epoch % 1000 == 0):
            print("data: {d}, epoch: {e}, loss: {l}".format(d = i, e = epoch, l = loss))

print("\n====== gradient descent finished =====\n")

q = q.transpose()
q = np.mat(q)
df = pd.DataFrame(q)
df.columns = ['q1', 'q2', 'q3', 'q4']
df.to_excel("5-2_ans.xls", index = False)
