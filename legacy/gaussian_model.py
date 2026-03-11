import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# units field
np.random.seed(666); nx=100; ny=100
nz=101# 1 layer for shale barrier
nz1=50; nz2=50; nz_sh=30; nz22=nz1+nz_sh

top=18000

x_len=15000; y_len=15000; x_size=x_len/nx

pay_thickness=500 # 2 pay zones
shale_thickness=350 # 1 shale 
dip=20

por1=0.17 ;   por2=0.19 

ntg1=0.7;     ntg2=0.62

por_std=0.03;   perm_std=0.5

perm1=np.log10(10);   perm2=np.log10(20)

sand_filt=[1.5,2.5,1.5]
facies_filt=[2.5,5,2.5]
sand_nug=0.05
# important surface calculation
x=np.linspace(0, x_len,nx)
y=np.linspace(0, y_len,ny)
X,Y=np.meshgrid(x,y,indexing='ij')
z1=Y*np.tan(dip/180*np.pi)+top
z2=z1+pay_thickness+gaussian_filter(np.random.normal(0,500,z1.shape),(11,11))
z3=z2+shale_thickness+gaussian_filter(np.random.normal(0,500,z1.shape),(11,11))
z4=z3+pay_thickness+gaussian_filter(np.random.normal(0,500,z1.shape),(11,11))

surf_name=['sand1_top','sand1_bot','sand2_top','sand2_bot']
zz=[z1,z2,z3,z4]

for ii in range(4):
    with open('surface/'+surf_name[ii], 'w') as f:
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):             
                f.write('%.6f %.6f %.6f\n' % (X[i,j],Y[i,j],zz[ii][i,j]))

np.random.seed(234)
# facies modeling
facies=gaussian_filter(np.random.normal(0,1,(3*nx,3*ny,3*nz)),facies_filt,mode='wrap')
facies=facies[nx:2*nx,ny:2*ny,nz:2*nz]
facies[:,:,:nz1]=facies[:,:,:nz1]<np.percentile(facies[:,:,:nz1].flatten(),ntg1*100)
facies[:,:,nz1:nz1+nz_sh]=False
facies[:,:,nz1+nz_sh:]=facies[:,:,nz1+nz_sh:]<np.percentile(facies[:,:,nz1+nz_sh:].flatten(),ntg2*100)

# property modeling
poro=np.random.normal(0,1,(3*nx,3*ny,3*nz))
poro_nug=np.random.normal(0,1,(3*nx,3*ny,3*nz))
# sand poro
poro=gaussian_filter(poro,sand_filt,mode='wrap')+sand_nug*poro_nug
poro=poro[nx:2*nx,ny:2*ny,nz:2*nz]
perm=poro.copy()

# for uppepr unit
# percentile transform
mean=[0,0]
var=[[1,0.6],[0.6,1]]
mm= multivariate_normal(mean,var)
sand1=mm.rvs(int(poro[:,:,:nz1].size*1.2))
sand1[:,0]=sand1[:,0]*por_std+por1
sand1[:,1]=sand1[:,1]*perm_std+perm1

sand1=sand1[sand1[:,0]<0.3]
sand1=sand1[sand1[:,0]>0.05]
sand1=sand1[sand1[:,1]<3.5]
sand1=sand1[:poro[:,:,:nz1].size]

# quantile transformation to force it to be gaussian
por_zone1=poro[:,:,:nz1].flatten()
por_zone1_order=np.argsort(por_zone1)
sand1=sand1[np.argsort(sand1[:poro[:,:,:nz1].size,0])]
por_zone1[por_zone1_order]=sand1[:,0]
perm_zone1=por_zone1.copy()
perm_zone1[por_zone1_order]=sand1[:,1]
poro[:,:,:nz1]=por_zone1.reshape(poro[:,:,:nz1].shape)
perm[:,:,:nz1]=perm_zone1.reshape(poro[:,:,:nz1].shape)

# for lower unit
mean=[0,0]
var=[[1,0.6],[0.6,1]]
mm= multivariate_normal(mean,var)
sand2=mm.rvs(int(poro[:,:,nz22:].size*1.2))
sand2[:,0]=sand2[:,0]*por_std+por2
sand2[:,1]=sand2[:,1]*perm_std+perm2
sand2=sand2[sand2[:,0]<0.3]
sand2=sand2[sand2[:,0]>0.05]
sand2=sand2[sand2[:,1]<3.5]
sand2=sand2[:poro[:,:,nz22:].size]

por_zone2=poro[:,:,nz22:].flatten()
por_zone2_order=np.argsort(por_zone2)
sand2=sand2[np.argsort(sand2[:poro[:,:,nz22:].size,0])]
por_zone2[por_zone2_order]=sand2[:,0]
perm_zone2=por_zone2.copy()
perm_zone2[por_zone2_order]=sand2[:,1]
poro[:,:,nz22:]=por_zone2.reshape(poro[:,:,nz22:].shape)
perm[:,:,nz22:]=perm_zone2.reshape(poro[:,:,nz22:].shape)

active=facies.astype(int)
poro_out=poro*facies
perm_out=(10**perm)*facies

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract x, y, and z coordinates from the numpy array
x, y, z = np.indices(poro_out.shape)

# Flatten the numpy array and use it as the values for the 3D plot
ax.scatter(x.flatten(), y.flatten(), z.flatten(), c=poro_out.flatten())
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
plt.show()