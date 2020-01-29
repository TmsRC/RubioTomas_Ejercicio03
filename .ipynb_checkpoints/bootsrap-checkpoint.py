import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.linear_model

datosFull = np.loadtxt('notas_andes.dat', skiprows=1)
Y = datosFull[:,4]
X = datosFull[:,:4]

betas=[[],[],[],[]]
for a in range(0,1000):
    indices = np.random.randint(0,69,(69,4))
    X_train = [[],[],[],[]]
    Y_train = []
    for i in range(0,69):
        X_train[0].append(X[indices[i][0]][0])
        X_train[1].append(X[indices[i][1]][1])
        X_train[2].append(X[indices[i][2]][2])
        X_train[3].append(X[indices[i][3]][3])
        Y_train.append(Y[indices[i]])
    X_train, Y_train = np.array(X_train).T,np.array(Y_train)
    #print(X_train.shape)
    regresion = sklearn.linear_model.LinearRegression()
    regresion.fit(X_train, Y_train)
    for b in range(0,4):
        betas[b].append(regresion.coef_[b])
betas = np.array(betas)

fig,axii = plt.subplots(2,2)
axes = list(axii[0])+list(axii[1])

fig,axii = plt.subplots(2,2)
axes = list(axii[0])+list(axii[1])

for i in range(0,4):
    a = axes[i]
    a.hist(betas[i])
    a.set_title('Beta '+str(i+1))
fig.tight_layout()