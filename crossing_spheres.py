"""
With this code I'm trying to evaluate the actual portion of the projectile that intersect the target.
"""

Rt = 1.
Rp = 0.6
theta = 35     # [deg]



import numpy as np
from numpy.random import random_sample as rn
import matplotlib.pyplot as plt
from PySam import get_colours_for_plot

b = np.sin(theta*np.pi/180)

plt.rcParams["figure.figsize"] = [10, 10]
plt.rcParams["figure.autolayout"] = True
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

u, v = np.mgrid[0:2 * np.pi:200j, 0:np.pi:200j]

N = 10000


#Target
xt = np.cos(theta*np.pi/180)*(Rt+Rp)
yt = 0.
zt = b*(Rt+Rp)
x = Rt*np.cos(u) * np.sin(v) - xt
y = Rt*np.sin(u) * np.sin(v) - yt
z = Rt*np.cos(v) - zt
ax.plot_surface(x, y, z,alpha=0.4,color='k')#, cmap=plt.cm.YlGnBu_r)

#Projectile
x = Rp*np.cos(u) * np.sin(v)
y = Rp*np.sin(u) * np.sin(v)
z = Rp*np.cos(v)
ax.plot_surface(x, y, z,alpha=0.1,color='b')

pin, pout = [],[]
for r in rn((N,3)):
    r = 2.*r-1
    if np.sqrt(sum(r*r))>1.: continue
    r=r*Rp
    rrt = np.sqrt(np.power(r[2]+zt,2.)+r[1]*r[1])
    if rrt>Rt: pout.append(r)
    else: pin.append(r)

if pin!=[]:
    pin = np.asarray(pin)
    ax.scatter(pin[:,0],pin[:,1],pin[:,2],c='b',s=1)
if pout!=[]:
    pout = np.asarray(pout)
    ax.scatter(pout[:,0],pout[:,1],pout[:,2],c='r',s=1)

ax.plot([0,-Rp-Rt],[0,0],[0,0],lw=2,c='k')
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)
ax.set_zlim(-2,2)
plt.show()


def get_minter(Rp,Rt,b,N=100,test=20):
    if b<(Rt-Rp)/(Rt+Rp): return 1,0
    res = []
    for i in range(test):
        nin,nout = 0,0
        for r in rn((N,3)):
            r = 2.*r-1
            if np.sqrt(sum(r*r))>1.: continue
            r=r*Rp
            rrt = np.sqrt(np.power(r[2]+b*(Rt+Rp),2.)+r[1]*r[1])
            if rrt<Rt: nin+=1
            else: nout+=1
        res.append(nin/(nin+nout))
    return np.mean(res),np.std(res)

Rps = np.linspace(0.1,0.9,5)
c = get_colours_for_plot(len(Rps))
for i,Rp in enumerate(Rps):
    b0 = (Rt-Rp)/(Rt+Rp)
    X,Y,Yerr=[],[],[]
    for b in np.linspace(b0,1,20):
        y,yerr=get_minter(Rp,Rt,b,N=10000,test=20)
        X.append((b-b0)/(1.-b0))
        Y.append(y)
        Yerr.append(yerr)
    plt.errorbar(X,Y,yerr=Yerr,c=c[i],label="Rp = {:.1f} Rt".format(Rp),zorder=0)

xmod = np.linspace(b0,1,100)
lr = (Rp+Rt)*(1.-xmod)/Rp
ymod = 0.25*(3.-lr)*lr*lr
for j in range(len(ymod)):
    if xmod[j]<b0: ymod[j]=1
plt.plot((xmod-b0)/(1.-b0),ymod,ls='-',c='k',zorder=1,label="LS12")

plt.xlabel('$(b-b_0)/(1-b_0)$')
plt.ylabel('$\\alpha$')
plt.legend()
plt.savefig("alpha.png")
plt.show()