import numpy as np
import matplotlib.pyplot as plt

# Set up
iThe = [np.pi/2,np.pi/2,np.pi/2] # Angle
iOme = [0,0,0] # Initial Angular velocity
l = 1 # Length
g = -10 # Gravity
dt = 0.005 # Time step
simLength = 999999999 # How many steps to run
staThresh = 80 # % Deviation from original energy, used to determine whether or not the simulation is stable

# optimisation
daSize = np.size(iThe)
i = np.arange(daSize).reshape(daSize,1)
j = np.arange(daSize)



# func
def AInv(The):
    return np.linalg.inv((daSize - np.maximum(i, j))*np.cos(The[j] - The[i]))

def R(The,Ome):
    return g/l * (daSize - j) * np.sin(The[j]) - np.sum(np.square(Ome[j]) * np.sin(The[i] - The[j]),axis = 1)

def Acc(The,Ome):
    return np.matmul(AInv(The), R(The,Ome))

def k1(The,Ome):
    return np.array([Ome,Acc(The,Ome)])

def k2(The,Ome,Ome1,Acc1):
    return np.array([Ome + dt/2*Acc1, Acc(The+ dt/2*Ome1,Ome+dt/2*Acc1)])

def k3(The,Ome,Ome2,Acc2):
    return np.array([Ome + dt/2*Acc2, Acc(The+ dt/2*Ome2,Ome+dt/2*Acc2)])

def k4(The,Ome,Ome3,Acc3):
    return np.array([Ome + dt*Acc3, Acc(The+ dt*Ome3,Ome+dt*Acc3)])

def iterate(The,Ome):
    O1,A1 = k1(The,Ome)
    O2,A2 = k2(The,Ome,O1,A1)
    O3,A3 = k3(The, Ome, O2, A2)
    O4,A4 = k4(The, Ome, O3, A3)

    return (np.array([O1,A1]) + np.array([O2,A2])/2 + np.array([O3,A3])/2 + np.array([O4,A4]))*dt/6 + np.array([The,Ome])

# Initilisation
x = np.cumsum(l*np.sin(iThe))
y = -np.cumsum(l*np.cos(iThe))
x = np.insert(x, 0, 0)
y = np.insert(y, 0, 0)
hasReport = False

# Etest
EInitial = np.sum((x - l*daSize)*g)

for u in range(simLength):

    xold = x
    yold = y

    iThe, iOme = iterate(np.array(iThe),np.array(iOme))

    x = np.cumsum(l*np.sin(iThe))
    y = -np.cumsum(l*np.cos(iThe))
    x = np.insert(x, 0, 0)
    y = np.insert(y, 0, 0)

    Energy = np.sum((np.square(2*(y - yold)/dt) + np.square(2*(x - xold)/dt))*1/2) - np.sum(y*g)
    Deviation = EInitial + Energy



    plt.cla()
    plt.cla()

    plt.plot(x,y,'bo-')

    plt.xlim(-l*daSize-1, l*daSize+1)
    plt.ylim(-l*daSize-1, l*daSize+1)


    plt.text(-l*daSize, l*daSize*(1-3/10), "Is stable: " + str(not hasReport))
    plt.text(-l*daSize, l*daSize*(1+1/10), "Time: " + str(np.round(u*dt,5)) + " Seconds")
    plt.text(-l*daSize, l*daSize+0,"Initial Energy: " + str(EInitial) + " J")
    plt.text(-l*daSize, l*daSize*(1-1/10),"Current Energy: " + str(np.round(Deviation,4)) + " J")
    plt.text(-l*daSize, l*daSize*(1-2/10), "Error: " + str(np.round(Energy/EInitial * 100, 4)) + "%")

    plt.pause(0.001)

    plt.plot(x,y,'bo-')

    plt.xlim(-l*daSize-1, l*daSize+1)
    plt.ylim(-l*daSize-1, l*daSize+1)


    plt.text(-l*daSize, l*daSize*(1-3/10), "Is stable: " + str(not hasReport))
    plt.text(-l*daSize, l*daSize*(1+1/10), "Time: " + str(np.round(u*dt,5)) + " Seconds")
    plt.text(-l*daSize, l*daSize+0,"Initial Energy: " + str(EInitial) + " J")
    plt.text(-l*daSize, l*daSize*(1-1/10),"Current Energy: " + str(np.round(Deviation,4)) + " J")
    plt.text(-l*daSize, l*daSize*(1-2/10), "Error: " + str(np.round(Energy/EInitial * 100, 4)) + "%")

    plt.pause(0.001)

    if (not hasReport) and (np.abs(Energy/EInitial) > staThresh/100):
        print("Simulation destabilized at t=" + str(np.round(u*dt,4)) + " seconds with " + str(u) + " iterations")
        hasReport = True



plt.show()