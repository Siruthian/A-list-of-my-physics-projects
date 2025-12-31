import numpy as np
import matplotlib.pyplot as plt



# simulation parameters / controls
constant = 1
definition = 1000 # Magnetic field accuracy
dTime = 0.0002
PInitial = np.array([3.2,0,1.0,0,0.5,5.1]) # Cartesian Cordinates, arranged Position, Velocity
daMass = 0.1
charge = 1

# Visual Control
radiusNum = 3
phiNum = 8
zNum = 3
graphSize = 5

# Poloidal field controls
PCurrent = 5
radiusRing = 3

# toroidal field controls
Tcurrent = 0

# verticle field controls:
vField = 0.0

#setups
xpath, ypath, zpath = [], [], []
fig = plt.figure()
numtime = 1

# finding the magnetic vector at a specific point
def vectorMag(pointR, pointPhi, pointZ, current, currentRadius,numSection):
    daField = [0,0,0]

    for x in range(numSection):
        # poloidal field controls
        zeSize = np.power(
            np.square(pointZ) + np.square(pointR) + np.square(currentRadius) - 2 * pointR * currentRadius * np.cos(
                2 * np.pi * x / numSection - pointPhi), 3 / 2)
        if not (zeSize == 0):
            daField[0] += current * constant * 2 * np.pi / numSection * pointZ * currentRadius * np.cos(
                2 * np.pi * x / numSection) / zeSize

            daField[1] += current * constant * 2 * np.pi / numSection * pointZ * currentRadius * np.sin(
                2 * np.pi * x / numSection) / zeSize

            daField[2] += current * constant * 2 * np.pi / numSection * currentRadius * (
                        currentRadius - pointR * np.cos(2 * np.pi * x / numSection - pointPhi)) / zeSize
        else:
            pass

    # toroidal field controls
    if pointR == 0:
        pass
    else:
        Tmag = 2 * constant * Tcurrent / pointR
        daField[0] += -np.sin(pointPhi) * Tmag
        daField[1] += np.cos(pointPhi) * Tmag

    # Verticle field controls

    daField[2] += vField
    return np.array(daField)

# Defining Particle Motion

print(vectorMag(4.0,0,1,1,2.5,1000))

def particleMotion(PpointX, PpointY, PpointZ, VpointX, VpointY, VpointZ, mass, dT):
    zeField = vectorMag(np.sqrt(np.power(PpointX,2) + np.power(PpointY,2)),np.arctan(PpointY / PpointX), PpointZ , PCurrent, radiusRing, definition)
    Accaleration = charge / mass * np.cross(np.array([VpointX,VpointY,VpointZ]),zeField)

    return np.array([PpointX + VpointX * dT +  1/2 * Accaleration[0] * dT ** 2, PpointY + VpointY * dT +  1/2 * Accaleration[1] * dT ** 2, PpointZ + VpointZ * dT + 1/2 * Accaleration[2] * dT ** 2, VpointX + Accaleration[0]* dT, VpointY + Accaleration[1]* dT, VpointZ + Accaleration[2]* dT])

# Generating Magnetic field for Visual effects

vectorfieldDimensions = []
for x in range(1,radiusNum + 1):
    for y in range(phiNum):
        for z in range(-zNum, zNum):
            daVector = vectorMag(graphSize * x/radiusNum, np.pi * 2 * y / phiNum, graphSize * z/ zNum, PCurrent , radiusRing, definition)
            vectorfieldDimensions.append([graphSize * x/radiusNum * np.cos(np.pi * 2 * y / phiNum), graphSize * x/radiusNum * np.sin(np.pi * 2 * y / phiNum), graphSize * z/zNum, daVector[0], daVector[1], daVector[2]])

vectorfieldDimensions = np.array(vectorfieldDimensions)

print("Generated Magnetic field")

ax = plt.axes(projection ='3d')

gx = np.cos(np.linspace(0,definition) / (definition) * 2*np.pi) * radiusRing
gy = np.sin(np.linspace(0,definition) / (definition) * 2*np.pi) * radiusRing
gz = 0

ay = fig.gca(projection='3d')

ax.set_xlim(-7,7)
ax.set_ylim(-7,7)
ax.set_zlim(-7,7)

ax.plot3D(gx, gy, gz, 'red')
ay.quiver(vectorfieldDimensions[:,0], vectorfieldDimensions[:,1], vectorfieldDimensions[:,2], vectorfieldDimensions[:,3], vectorfieldDimensions[:,4], vectorfieldDimensions[:,5], length=0.1)

# Updating particle motion

while (True):
    ax.cla()
    xpath.append(PInitial[0])
    ypath.append(PInitial[1])
    zpath.append(PInitial[2])
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.plot3D(gx, gy, gz, 'red')
    ax.plot(xpath, ypath, zpath, 'b')
    ay.quiver(vectorfieldDimensions[:, 0], vectorfieldDimensions[:, 1], vectorfieldDimensions[:, 2],
              vectorfieldDimensions[:, 3], vectorfieldDimensions[:, 4], vectorfieldDimensions[:, 5], length=0.2)

    ax.scatter(PInitial[0], PInitial[1], PInitial[2], c='r')
    plt.pause(0.001)
    PInitial = particleMotion(PInitial[0],PInitial[1],PInitial[2],PInitial[3],PInitial[4],PInitial[5], daMass, dTime)
    numtime += 1
    if (np.mod(numtime,500) == 1):
        print(str((numtime - 1)/100) + "deci seconds elapse")


