#----------------------------------------------------------------------
#----------------------------------------------------------------------
#
# generate_SPH_table.py
# By S. Crespi, Nov 2021
# Version 6.0
#
# This algorithm reads the 'aggregates.txt' file of each SPH simulation
#  and produces the look-up table
# This table is essential for the EFP_rebound code that assumes single
#  ring.
#
# VERSION NOTE
# 5.0 In this version the output file cointains the orbital elements, in
#  spherical (sperical=True) or cartesian coordinates, of the 2 largest
#  remnants.
# 6.0 In this version the fraction of water with respect the total of water
#  is provided instead of the water fraction
#
#----------------------------------------------------------------------
#----------------------------------------------------------------------

#************************   BOOLEAN OPTIONS   *************************

spherical=True

#***************************   LIBRARIES   ****************************

import numpy as np
import os
from glob import glob

#***************************   CONSTANTS   ****************************

otput_file='SPH.table'

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

#----------------------------------------------------------------------
#----------------------------------------------------------------------

#****************************   CLASSES   *****************************

class collision:
	"""Collision event"""
	def __init__(self,line):
		self.oldid=line[16:20]
		self.SPHpcode=line.split('_')[2:]
		self.SPHpval=[]
		for val in self.SPHpcode:
			if val[0]=='w': self.SPHpval.append(float(val[2:]))
			elif val[0]=='m': self.SPHpval.append(get_mval(int(val[1:])))
			else: self.SPHpval.append(float(val[1:]))
		
		# Getting the data
		f=open(line+'/aggregates.txt','r')
		lines=f.readlines()
		f.close()
		
		l1m=np.asarray(lines[2].split(),dtype=float)
		l1x=np.asarray(lines[15].split(),dtype=float)
		l2m=np.asarray(lines[5].split(),dtype=float)
		l2x=np.asarray(lines[14].split(),dtype=float)
		mtot,_,wtot=np.asarray(lines[8].split(),dtype=float)

		l1=np.concatenate((l1x,np.delete(l1m,1)),axis=None)
		l2=np.concatenate((l2x,np.delete(l2m,1)),axis=None)
		
		if l2[6]==-1.:
			self.largest=[units(l1,mtot,wtot),l2]
			self.Nbig=1
			self.mfr=0.
			
		else:
			self.largest=[units(l1,mtot,wtot),units(l2,mtot,wtot)]
			m1,m2=self.largest[0][6],self.largest[1][6]
			if m2>m1:
				self.largest=[self.largest[1],self.largest[0]]
				m1,m2=self.largest[0][6],self.largest[1][6]
			self.Nbig=get_Nbig(m1,m2,self.SPHpval[3])
			if self.Nbig==0: self.mfr=1.
			elif self.Nbig==1: self.mfr=1.-m1
			else: self.mfr=1.-m1-m2
		
		self.code=assign_code(self.SPHpcode)
		self.id=get_index_from_code(self.code)
		self.crashed=False


class empty_col:
	"""Empty collision"""
	def __init__(self,num):
		self.oldid='-1'
		self.id=num
		
		self.code=get_code_from_index(num)
		self.SPHpcode=[]
		for i in range(len(allpar)):
			par=allpar[i]
			self.SPHpcode.append(par[int(self.code[i])])
		
		self.SPHpval=[]
		for val in self.SPHpcode:
			if val[0]=='w': self.SPHpval.append(float(val[2:]))
			elif val[0]=='m': self.SPHpval.append(get_mval(int(val[1:])))
			else: self.SPHpval.append(float(val[1:]))

		self.largest=[np.zeros(8)-1,np.zeros(8)-1]

		self.Nbig=-1
		self.mfr=-1

		self.crashed=True

#***************************   FUNCTIONS   ****************************

def get_mval(m):
	if m==21: return 2*MCERES
	elif m==23: return 2*MMOON
	elif m==24: return 2*MMARS
	else: return 2*MEARTH

def get_Nbig(m1,m2,g):
	s1=1/(1.+g)
	s2=g*s1
	# CC
	if m1<s1*0.1: return 0		# Catastrophic Collision
	elif m2<s2*0.1: return 1	# Projectile accretion or destruction
	else: return 2				# Hit-and-run (or erosive HAR)
	
def assign_code(val):
	code=''
	for i in range(len(val)): code+=str(allpar[i].index(val[i]))
	return code

def get_index_from_code(code):
	val=0
	for i in range(len(bases)-1): val=(val+int(code[i]))*(int(bases[i+1]))
	return val+int(code[5])+1

def get_code_from_index(ind):
	code=''
	val,den=ind-1,Ncol*1
	for i in range(len(bases)):
		den=den/int(bases[i])
		code+=str(val/den)
		val=val-int(code[i])*den
	return code

def get_col(i):
	for col in raw_collisions:
		if col.id==i: return col
	return empty_col(i)

def units(l,mtot,wtot):
	l[:3]=l[:3]/AU
	l[3:6]=l[3:6]/(AU/(YEAR/2./np.pi))
	l[6]=l[6]/mtot
	l[7]=l[6]*l[7]/wtot
	return l
	
def go_to_spherical():
	for col in collisions:
		if col.crashed: continue
		col.largest[0][:6]=cart_to_sph(col.largest[0][:6])
		if col.largest[1][6]!=-1.: col.largest[1][:6]=cart_to_sph(col.largest[1][:6])

def cart_to_sph(xv):
	r=np.sqrt(sum(np.power(xv[:3],2.)))
	v=np.sqrt(sum(np.power(xv[3:],2.)))
	thr=np.arccos(xv[2]/r)
	thv=np.arccos(xv[5]/v)
	phir=np.arctan2(xv[1],xv[0])
	phiv=np.arctan2(xv[4],xv[3])
	return np.asarray([r,phir,thr,v,phiv,thv])

def identical_particles():
	for i in range(int(bases[0])):
		for j in range(int(bases[1])):
			for k in range(int(bases[2])):
				i1=get_index_from_code('{}{}{}201'.format(i,j,k))
				i2=get_index_from_code('{}{}{}210'.format(i,j,k))
				c1,c2=collisions[i1-1],collisions[i2-1]
				if not c2.crashed:
					c1.largest=c2.largest
					c1.Nbig=c2.Nbig
					c1.mfr=c2.mfr
					c1.crashed=False
					continue
				if not c1.crashed:
					c2.largest=c1.largest
					c2.Nbig=c1.Nbig
					c2.mfr=c1.mfr
					c2.crashed=False

# Table writing functions

def initialise_SPH_table(filename,spherical):
	f=open(filename,'w+')
	f.write('col_num code')
	f.write(' v[vesc] alpha[deg] mtot[MSUN] gamma[1] wft[%] wfp[%]')
	f.write(' Nbig[1] mfr[mtot]')
	if spherical:
		f.write(' r1[AU] phi_r1[rad] th_r1[rad]')
		f.write(' v1[AU/yr/2pi] phi_v1[rad] th_v1[rad]')
		f.write(' m1[mtot] wf1[1]')	
		f.write(' r2[AU] phi_r2[rad] th_r2[rad]')
		f.write(' v2[AU/yr/2pi] phi_v2[rad] th_v2[rad]')
		f.write(' m2[mtot] w1[wtot]\n')	
	else:
		f.write(' x1[AU] y1[AU] z1[AU]')
		f.write(' vx1[AU/yr/2pi] vy1[AU/yr/2pi] vz1[AU/yr/2pi]')
		f.write(' m1[mtot] wf1[1]')
		f.write(' x2[AU] y2[AU] z2[AU]')
		f.write(' vx2[AU/yr/2pi] vy2[AU/yr/2pi] vz2[AU/yr/2pi]')
		f.write(' m2[mtot] w2[wtot]\n')
	return f
	
def write_SPH_table(col,f):
	f.write('{} {} '.format(col.id,col.code))
	for val in col.SPHpval: f.write('{} '.format(val))
	f.write('{} {} '.format(col.Nbig,col.mfr))
	for val in col.largest:
		for num in val: f.write('{} '.format(num))
	f.write('\n')

#----------------------------------------------------------------------
#----------------------------------------------------------------------

#*****************************   INPUT   ******************************

# Load all the SPH collisions
raw_collisions = [collision(line) for line in np.sort(glob('SPH_catalogue/id*'))]

# Ordinate the collisions and create the crasehd one
collisions = [get_col(num) for num in range(1,Ncol+1)]

# Spherical coordinates
if spherical: go_to_spherical()

# Reproduce simulations with identical particles (gamma=1,wt=wp)
identical_particles()

#----------------------------------------------------------------------
#----------------------------------------------------------------------		

#*****************************   OUTPUT   *****************************

# Write SPH table
f=initialise_SPH_table(otput_file,spherical)
for collision in collisions: write_SPH_table(collision,f)

for col in collisions:
	if col.crashed: print ' - collision {} ({}) -> crashed'.format(col.id,col.code)
