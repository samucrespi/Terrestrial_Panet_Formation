#----------------------------------------------------------------------
#----------------------------------------------------------------------
#
# add_frag_file_to_catalog.py
# By S. Crespi, Mar 2022
#
# This algorithm generates filter the fragments in the SPH output by
#  calculationg the aggregates. Than it writes the file
#  'all_aggregates.txt' that cointains informations about all the
#  independent bodies.
# The aggregates is defined by all the particles which have relative
#  velocity smnaller than the escape velocity.
#
#----------------------------------------------------------------------
#----------------------------------------------------------------------

import glob
import numpy as np

#----------------------------------------------------------------------
#----------------------------------------------------------------------

G = 6.67408e-11	#[SI]
MEAR = 3.003489616e-06		# Earth mass [MSUN]
MSUN = 1.98847e33		# Solar mass [g]

#----------------------------------------------------------------------
#----------------------------------------------------------------------

def aggregate(b1,b2):
	if get_v0(b1,b2)<1: return True
	else: return False

def get_v0(b1,b2):
	x1,x2=b1[:3],b2[:3]
	v1,v2=b1[3:6],b2[3:6]
	m1,m2=b1[6],b2[6]
	dx,dv = vec_dif(x1,x2),vec_dif(v1,v2)
	vesc = np.sqrt(2.*G*(m1+m2)/dx)
	return dv/vesc

def vec_dif(vec1,vec2): return np.sqrt(sum(np.power(vec1-vec2,2.)))

def merge(b1,b2):
	res = np.zeros(9) #(x1,x2,x3,v1,v2,v3,m,mrel,wfrac)
	res[6:8] = b1[6:8]+b2[6:8]
	res[:6] = (b1[:6]*b1[6]+b2[:6]*b2[6])/res[6]
	res[8] = (b1[8]*b1[6]+b2[8]*b2[6])/res[6]
	return res

#----------------------------------------------------------------------
#----------------------------------------------------------------------

path = 'SPH_catalogue'
all_dir = sorted(glob.glob(path+'/id*'))

for sim in all_dir:
	if int(sim.split('/')[1][2:6])<498 or int(sim.split('/')[1][2:6])>509: continue
	print '\nStudying simulation ', sim.split('/')[1][2:6]

	f = open(sim+'/impact.frag.0300','r')
	lines = f.readlines()
	f.close()
	lines.pop(0)
	
	# Get fragment data
	Nbod = len(lines)
	bodies = []				# N X (x1,x2,x3,v1,v2,v3,m,mrel)
	for i in range(Nbod):
		l=np.asarray(lines[i].split(),dtype=np.float64)
		bodies.append(np.asarray([l[0],l[1],l[2],l[3],l[4],l[5],l[6],l[7],l[9]]))

	# Aggregate
	f = open(sim+'/aggregates_info.txt','w+')
	i,tot_counter = 0,0
	while i<len(bodies):
		new_aggregate = True
		counter = 0
		while new_aggregate:
			new_aggregate = False
			b1 = bodies[i]
			for j in range(i+1,len(bodies)):
				b2 = bodies[j]
				if aggregate(b1,b2):
					counter+=1
					tot_counter+=1
					#merge
					bodies[i] = merge(b1,b2)
					#pop
					bodies.pop(j)
					#repeat
					new_aggregate = True
					break
		if counter!=0: f.write('Body {} aggregates {} smaller bodies\n'.format(i+1,counter))
		i+=1
	f.write('\nTotal bodies merged: {} ({}%)'.format(tot_counter,np.round(100.*tot_counter/(len(bodies)+tot_counter),decimals=2)))
	f.close()
	
	# Write file
	f = open(sim+'/all_aggregates.txt','w+')
	f.write('x1\tx2\tx3\tv1\tv2\tv3\tmass\trel_mass\twater_fraction\n')
	for body in bodies:
		for val in body: f.write(str(val)+' ')
		f.write('\n')
	f.close
