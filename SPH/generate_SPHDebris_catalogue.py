from glob import glob
import os
import numpy as np
import shutil

#***************************   CONSTANTS   ****************************

# For coding the collisions
v0s=['v1.0','v1.5','v2.0','v3.0','v5.0']
alphas=['a0','a20','a40','a60']
mtots=['m21','m23','m24','m25']
gammas=['g0.1','g0.5','g1.0']
wts=['wt10.0','wt20.0']
wps=['wp10.0','wp20.0']
allpar=[v0s,alphas,mtots,gammas,wts,wps]
bases='544322'
Ncol=1
for i in bases: Ncol*=int(i)

MEARTH = 3.003489616e-06	# Earth mass [MSUN]
MMARS = 3.227154546e-07		# Mars mass [MSUN]
MMOON = 3.6951390215e-8		# Moon mass [MSUN]
MCERES = 4.7189547743e-10	# Ceres mass [MSUN]
MSUN = 1.98847e30			# Solar mass [kg]
AU = 1.495978707e11			# Astronomical Unit [m]
YEAR = 31556952.			# year [s]


#***************************   FUNCTIONS   ****************************


def assign_code(val):
	code=''
	for i in range(len(val)): code+=str(allpar[i].index(val[i]))
	return code

def get_index_from_code(code):
	val=0
	for i in range(len(bases)-1): val=(val+int(code[i]))*(int(bases[i+1]))
	return val+int(code[5])+1

#----------------------------------------------------------------------
#----------------------------------------------------------------------	


fldrs=sorted(glob('SPH_catalogue/id*'))
for f in fldrs[:]:
	SPHpcode=f.split('/')[-1].split('_')[1:]
	
	code=assign_code(SPHpcode)
	cid=str(get_index_from_code(code))
	while len(cid)<3: cid='0'+cid
	
	data=np.loadtxt(f+'/all_aggregates.txt',skiprows=1)
	
	if data.ndim==1:
		for i in range(3): data[i]=data[i]/AU
		for i in range(3,6): data[i]=data[i]*YEAR/2./np.pi/AU
		data[6]=data[6]/MSUN
		data[8]=data[8]*100
		f=open('SPHDebris_catalogue/{}_{}.dat'.format(cid,code),'w+')
		f.write('# x[AU] y[AU] z[AU] vx[AU/yr/2pi] vy[AU/yr/2pi] vz[AU/yr/2pi] m[MSUN] m[mtot] wf[%]\n')
		f.write('{}'.format(data[0]))
		for d in data[1:]: f.write('\t{}'.format(d))
		f.write('\n')
		f.close()
		print ' - {}  done  {}'.format(cid,data[-2])			
	else:
		for i in range(3): data[:,i]=data[:,i]/AU
		for i in range(3,6): data[:,i]=data[:,i]*YEAR/2./np.pi/AU
		data[:,6]=data[:,6]/MSUN
		data[:,8]=data[:,8]*100
		f=open('SPHDebris_catalogue/{}_{}.dat'.format(cid,code),'w+')
		f.write('# x[AU] y[AU] z[AU] vx[AU/yr/2pi] vy[AU/yr/2pi] vz[AU/yr/2pi] m[MSUN] m[mtot] wf[%]\n')
		for dat in data:
			f.write('{}'.format(dat[0]))
			for d in dat[1:]: f.write('\t{}'.format(d))
			f.write('\n')
		f.close()
		print ' - {}  done'.format(cid)
		
	if code[3:]=='201':
		newcode=code[:4]+'10'
		newcid=str(get_index_from_code(newcode))
		while len(newcid)<3: newcid='0'+newcid
	
		shutil.copyfile('SPHDebris_catalogue/{}_{}.dat'.format(cid,code),'SPHDebris_catalogue/{}_{}.dat'.format(newcid,newcode))
	
		print ' - {}  copied'.format(newcid)

