import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


gridNumber = 125
dt = 0.001
iterNum = 100000
difConstant = 100

def matGen(difCoe,gridNum):

    gridNum += 2
    startMat = np.zeros((np.power(gridNum,2),np.power(gridNum,2)))

    for x in range(1,gridNum-1):
        for y in range(1,gridNum-1):
            startMat[x+y*gridNum, x+ y*gridNum] = -4
            startMat[x+y*gridNum,x+1+y*gridNum] = 1
            startMat[x + y * gridNum, x - 1 + y * gridNum] = 1
            startMat[x + y * gridNum, x + (y + 1) * gridNum] = 1
            startMat[x + y * gridNum, x + (y - 1) * gridNum] = 1

    return difCoe*startMat

def initialCond(x,y):
    return 0.001*np.power(np.e,(-(x-gridNumber/2)**2-(y-gridNumber/2)**2)/1696) * (np.cos((x+y-gridNumber)/2) + 1)/2

def initialGen(gridNum):
    gridNum += 2
    startInt = np.zeros((np.power(gridNum,2),1))

    for x in range(1,gridNum-1):
        for y in range(1,gridNum-1):
            startInt[x+y*gridNum,0] = initialCond(x,y)
    return startInt

def heatSource(gridNum):
    gridNum +=2
    toReturn = np.zeros((gridNum,gridNum))
    angle = np.linspace(0,2*np.pi,num=3*gridNum)
    toReturn[np.round(gridNum/4*(np.cos(angle))+gridNum/2).astype(int)[:],np.round(gridNum/4*(np.sin(angle))+gridNum/2).astype(int)[:]] += 1
    return 2*toReturn.reshape(np.power(gridNum,2),1)

inputHeat = heatSource(gridNumber)


compMat = matGen(difConstant, gridNumber)
distribution = initialGen(gridNumber)


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

x = np.arange(gridNumber + 2)
y = np.arange(gridNumber + 2)
x, y = np.meshgrid(x, y)


for time in range(iterNum):

    vel = np.matmul(compMat,distribution)
    vel += inputHeat
    distribution += dt * vel

    z = distribution.reshape((gridNumber + 2, gridNumber + 2))

    ax.clear()
    ax.set_zlim(0, 1)
    ax.plot_surface(x, y, z, cmap="inferno")


    plt.pause(0.001)

plt.show()
