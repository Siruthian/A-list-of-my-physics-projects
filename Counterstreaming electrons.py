import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

#Edge condition: wrap around

numparticle = 4000
numgrid = 200

epsilon = 1
vinitial = 10
dx = 1
dt = 0.1
mass = 1
pertubation = 100
pertubationmax = 100000
numsim = 20000
edensity = 1


# setup particle and field

vt = np.zeros((numparticle,3))
efield = np.zeros((round(numgrid),3))


vt[:round(numparticle/2), 0] = (1 * vinitial)
vt[round(numparticle/2):, 0] = -(1 * vinitial)

xcord = np.arange(numgrid)

for x in range(round(numparticle/2)):
    vt[x, 1] = ((x)*dx* 2 * numgrid/numparticle )
    vt[x + round(numparticle/2), 1] = (x*dx * 2 * numgrid/numparticle)

for x in range(round(numparticle)):
    vt[x, 0] += 1 / pertubationmax * np.random.uniform(-pertubation, pertubation) * vinitial

for z in range(numsim):

    # Updating charge distribution

    j = np.floor(vt[:, 1] / dx).astype(int)
    jm = j + 1
    weightf = (dx*jm - vt[: , 1]) / dx
    weightc = (vt[: , 1] - dx* j) / dx
    jm = np.mod(jm, numgrid) # we assume that the particle will wrap around
    efield[: , 2] = (np.bincount(j, weights = weightf, minlength = numgrid) + np.bincount(jm, weights = weightc, minlength = numgrid)) * edensity * numgrid / (numparticle * dx) # Normalisation



    # solving for electric potential

    mat1 = np.zeros((numgrid,numgrid))
    for i in range(1, numgrid-1):
        mat1[i, i] = -2
        mat1[i, i+1] = 1
        mat1[i, i-1] = 1
    mat1[0, numgrid - 1] = 1
    mat1[0,0] = -2
    mat1[0,1] = 1
    mat1[numgrid - 1, 0] = 1
    mat1[numgrid - 1, numgrid - 2] = 1
    mat1[numgrid - 1, numgrid - 1] = -2

    mat1 /= dx**2

    efield[:, 0] = spsolve(mat1, efield[:, 2] - edensity)

    # solving for electric field
    for x in range(1,numgrid-1):
        efield[x , 1] = - (efield[x + 1 , 0] - efield[x -1, 0])/(2*dx)
    efield[0 ,1] = -(efield[1, 0] - efield[numgrid-1, 0])/(2*dx)
    efield[numgrid - 1, 1] = -(efield[numgrid - 2, 0] - efield[0, 0]) / (2 * dx)


    # updating motion

    vt[:,2] = np.interp(vt[:, 1], xcord, efield[:, 1])
    vt[:,0] += - vt[:,2] / mass / 2 *dt
    vt[:,1] += dt* vt[:,0]
    vt[:,1] = np.mod(vt[:,1], numgrid)


    plt.cla()
    plt.scatter(vt[:int(numparticle/2),1],vt[:int(numparticle/2),0], s=.4, color='blue', alpha=0.5)
    plt.scatter(vt[int(numparticle/2):,1],vt[int(numparticle/2):,0], s=.4, color='red', alpha=0.5)
    plt.legend()

    plt.text(2,np.max(vt[:,0])+1.5,"current sim number: " + str(z) + " of interval " + str(dt) + " out of " + str( numsim ))
    plt.pause(0.001)


plt.show()
