#----------------------------------------------------------------------
#----------------------------------------------------------------------
#
# PM_Rebound.py
# By S. Crespi, Feb 2022
# Version 1.0
#
# This algorithm integrate s forming planetary system by assuming 
#  inelastic collisions only.
# It is based on SPH_Rebound.py v4
#
# NB: all the units are in "rebound" units (G=1) except where otherwise
#     specified
#
# Version note:
#	v1: this version is simply SPH_Rebound.py (v4) with perfect merging
#  
#
#----------------------------------------------------------------------
#----------------------------------------------------------------------

#************************   SIMULATION SETUP   ************************

# starting time    ( 'start' or 't1000', for example, t0=1000yr)
t0='start'
t0='t10000000'

# integration time in years
Dt = 90000000
dt=1.e-2    # less than 1 day in [yr/2pi]

# checkpoints
Nsaves=900

# scenario
scenario='0_rnBurger20_eJS1'

# number of Gas Giants
NGG=2

#************************   BOOLEAN OPTIONS   *************************

save_checkpoints=True		# generates a snapshot file at each checkpoint
new_sim_warning=False		# requires user input to delete old files
save_progess=True			# save the progress of the simulation on a log file

#***************************   LIBRARIES   ****************************

import rebound
import reboundx
import numpy as np
import glob

#***************************   CONSTANTS   ****************************

from CONSTANTS import *

#----------------------------------------------------------------------
#----------------------------------------------------------------------

#***************************   FUNCTIONS   ****************************

#-------------------
# Collisions solver
#-------------------

def collision_solver(sim_pointer, collision):
	sim = sim_pointer.contents
	
	# get collision parameters
	p1,p2=ps[collision.p1],ps[collision.p2]
	
	# collision with Sun
	if collision.p1==0:
		ps[0].m+=p2.m
		write_rmv_file(sim.t,'SUN_col',p2.m,p2.params['wf'])
		return 2
	if collision.p2==0:
		ps[0].m+=p1.m
		write_rmv_file(sim.t,'SUN_col',p1.m,p1.params['wf'])
		return 1

	# collision with Gas Giants
	if collision.p1<=NGG:
		merge(p1,p2)
		write_rmv_file(sim.t,'GG{}_col'.format(collision.p1),p2.m,p2.params['wf'])
		return 2
	if collision.p2<=NGG:
		merge(p2,p1)
		write_rmv_file(sim.t,'GG{}_col'.format(collision.p2),p1.m,p1.params['wf'])
		return 1
	
	#  !!! bug !!!
	xrel=np.asarray([p1.x-p2.x,p1.y-p2.y,p1.z-p2.z])
	if np.sqrt(xrel.dot(xrel))>2.*(p1.r+p2.r): return 0
	#  !!!!!!!!!!
	
	indeces=[collision.p1,collision.p2]	
	coll_p=get_coll_params(p1,p2)
	
	# save snapshot at the collision time
	collision_snapshot(ps,indeces)
	
	#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
	print('\n{}'.format('-'*80))
	print('Collision parameters: ')
	print('  - v0 = {:.2f}'.format(coll_p[0]))
	print('  - alpha = {:.1f}'.format(coll_p[1]))
	print('  - mtot = {:.2f} [MEAR]'.format(coll_p[2]/MEAR))
	print('  - m1/m2 = {:.2f}'.format(coll_p[3]))
	print('  - wf1 = {:.1f} [%]'.format(coll_p[4]))
	print('  - wf2 = {:.2f} [%]'.format(coll_p[5]))
	#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^	
	
	# collision CoM position and velocity
	x1,v1,m1,wf1,R1=np.asarray([p1.x,p1.y,p1.z]),np.asarray([p1.vx,p1.vy,p1.vz]),p1.m,p1.params['wf'],p1.r
	x2,v2,m2,wf2,R2=np.asarray([p2.x,p2.y,p2.z]),np.asarray([p2.vx,p2.vy,p2.vz]),p2.m,p2.params['wf'],p2.r
	
	mtot=m1+m2
	xCoM=(x1*m1+x2*m2)/mtot
	vCoM=(v1*m1+v2*m2)/mtot
	wf=(wf1*m1+wf2*m2)/mtot
	
	# update parameters
	sim.particles[indeces[0]].x=xCoM[0]
	sim.particles[indeces[0]].y=xCoM[1]
	sim.particles[indeces[0]].z=xCoM[2]
	sim.particles[indeces[0]].vx=vCoM[0]
	sim.particles[indeces[0]].vy=vCoM[1]
	sim.particles[indeces[0]].vz=vCoM[2]
	sim.particles[indeces[0]].m=mtot
	sim.particles[indeces[0]].params['wf']=wf
	sim.particles[indeces[0]].r=get_radius(mtot,wf)	
	
	
	#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
	mwtot=mtot*wf/100
	print('\n Mass and Water check:')
	print('--- before collision ---')
	print(' TOT: m={:.4f} MEAR'.format(mtot/MEAR))
	print('      w={:.5f} MEAR (wf:{:.2f}%)'.format(mwtot/MEAR,wf))
	print(' - 1: m={:.4f} MEAR ({:.2f}%)'.format(m1/MEAR,100.*m1/mtot))
	print('      w={:.5f} MEAR ({:.2f}% from this body with wf:{:.3f}%)'.format(wf1*m1/MEAR/100,100.*wf1/wf,wf1))
	print(' - 2: m={:.5f} MEAR ({:.3f}%)'.format(m2/MEAR,100.*m2/mtot))
	print('      w={:.6f} MEAR ({:.3f}% from this body with wf:{:.3f}%)'.format(wf2*m1/MEAR/100,100.*wf2/wf,wf2))
	print('--- after collision ---')
	print(' - 1: m={:.4f} MEAR'.format(mtot/MEAR))
	print('      w={:.5f} MEAR (wf:{:.2f}%)'.format(mwtot/MEAR,wf))
	print('\n\n Survivor ',i+1)
	print('   m: {:.4f} [MEAR]'.format(mtot/MEAR))
	print('   wf: {:.2f} [%]'.format(wf))
	print('   r: ',np.sqrt(xCoM.dot(xCoM)),' [AU]')
	print('   v: ',np.sqrt(vCoM.dot(vCoM)),' [AU/yr/2pi]')
	print('   a: {:.3f} [AU]'.format(sim.particles[indeces[0]].a))
	print('   e: {:.4f} '.format(sim.particles[indeces[0]].e))
	print('   inc: {:.4f} [pi]'.format(sim.particles[indeces[0]].inc/np.pi))
	print()
	#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^	
	
	# save collisions on file
	survivors=[[xCoM,vCoM,mtot,wf]]
	chi,psi,h=angle_SPH_Rebound([x1,v1,m1],[x2,v2,m2],R1,R2)
	save_collision(coll_p,xCoM,vCoM,chi,psi,h,survivors,0,0)
	
	return 2


def get_coll_params(p1,p2):
	m1,m2=p1.m,p2.m
	xrel=np.asarray([p1.x-p2.x,p1.y-p2.y,p1.z-p2.z])
	vrel=np.asarray([p1.vx-p2.vx,p1.vy-p2.vy,p1.vz-p2.vz])
	xmod=np.sqrt(xrel.dot(xrel))
	vmod=np.sqrt(vrel.dot(vrel))
	
	mtot=m1+m2
	gamma=min(m1/m2,m2/m1)
	vesc=np.sqrt(2.*mtot/xmod)
	v0=vmod/vesc
	alpha=np.arccos(abs(np.dot(xrel/xmod,vrel/vmod)))*180/np.pi #[deg]
	if m1>m2: wft,wfp=p1.params['wf'],p2.params['wf']
	else: wft,wfp=p2.params['wf'],p1.params['wf']

	return v0,alpha,mtot,gamma,wft,wfp


def merge(pa,pb):
	# merge the second body into the first one
	mab=pa.m+pb.m
	pa.x=(pa.m*pa.x+pb.m*pb.x)/mab
	pa.y=(pa.m*pa.y+pb.m*pb.y)/mab
	pa.z=(pa.m*pa.z+pb.m*pb.z)/mab
	pa.vx=(pa.m*pa.vx+pb.m*pb.vx)/mab
	pa.vy=(pa.m*pa.vy+pb.m*pb.vy)/mab
	pa.vz=(pa.m*pa.vz+pb.m*pb.vz)/mab
	pa.m=mab
		
	
def angle_SPH_Rebound(l1,l2,R1,R2):

	# move to SoC of smaller body
	if l1[2]>l2[2]:
		r0=l1[0]-l2[0]
		v0=l1[1]-l2[1]
		c=['b','r']
		R=[R2,R1]
	else:
		r0=l2[0]-l1[0]
		v0=l2[1]-l1[1]	
	
	# rotate the system so that h=[0,0,+h]
	h=np.cross(r0,v0)
	phi=np.arccos(h[0]/np.sqrt(h[0]*h[0]+h[1]*h[1]))
	th=np.arccos(h[2]/np.sqrt(h.dot(h)))
	
	r0=Ry(Rz(r0,-phi),-th)
	v0=Ry(Rz(v0,-phi),-th)

	# orbital elements
	r0_mod=np.sqrt(r0.dot(r0))
	v0_mod=np.sqrt(v0.dot(v0))
	mtot=l1[2]+l2[2]
	a=1./(v0_mod*v0_mod/mtot-2./r0_mod)
	e=np.sqrt(1.+h.dot(h)/mtot/a)
	f0=-np.arccos(((a*(e*e-1)/r0_mod)-1.)/e)
	
	# rotating so that f=0 for y=0 (omega=0)
	th0=np.arctan2(r0[1],r0[0])
	omega=th0-f0
	R0=Rz(r0,-omega)
	V0=Rz(v0,-omega)
	
	# back-tracing the colliding bodies to d=5(R1+R2) and get the velocity angle wrt x-axis (delta)
	rmin=5.*(R1+R2)
	fd=-np.arccos(((a/rmin)*(e*e-1.)-1.)/e)
	cfd,sfd=np.cos(fd),np.sin(fd)
	phid=np.arctan2(1.+e*cfd,e*sfd)

	# getting the y_versor of the SPH frame (velocity versor of the bigger body)
	SPHy=np.asarray([np.cos(fd+phid),np.sin(fd+phid),0.])
	SPHy=Rz(Ry(Rz(SPHy,omega),th),phi)
	chi=np.arccos(SPHy[2])
	psi=np.arctan2(-SPHy[0],SPHy[1])
	
	return chi,psi,h


def Ry(v,t):	#rotates v around the y-axis through the angle t
	ct,st=np.cos(t),np.sin(t)
	return np.asarray([ct*v[0]+st*v[2],v[1],-st*v[0]+ct*v[2]])
	
def Rz(v,t):	#rotates v around the z-axis through the angle t
	ct,st=np.cos(t),np.sin(t)
	return np.asarray([ct*v[0]-st*v[1],st*v[0]+ct*v[1],v[2]])


def get_radius(m,wf):	# m in Solar masses
	# independent from wf
	# from Chen & Kipping 2017
	C=1.008*REAR
	S=0.279
	R=C*np.power(m/MEAR,S)
	# depending on wf
	# ... still to be done		<<------------
	return R	# radius [AU]


#-------------------
# remove particles
#-------------------

def remove_ps():
	sim.move_to_hel()
	for i in reversed(range(1,sim.N)):
		# unbound particles
		if ps[i].e>=1.:
			write_rmv_file(sim.t,'HYPER',ps[i].m,ps[i].params['wf'])
			sim.remove(i)
			continue
		# too far
		if ps[i].a>100.:
			write_rmv_file(sim.t,'Agt100',ps[i].m,ps[i].params['wf'])
			sim.remove(i)
			continue
		if ps[i].a*(1.-ps[i].e)>12.:
			write_rmv_file(sim.t,'PERIgt12',ps[i].m,ps[i].params['wf'])
			sim.remove(i)
			continue
		# too close to the Sun
		if ps[i].a<0.2:
			write_rmv_file(sim.t,'Alt02',ps[i].m,ps[i].params['wf'])
			ps[0].m+=ps[i].m
			sim.remove(i)
			continue
		if ps[i].a*(1.-ps[i].e)<0.03:
			write_rmv_file(sim.t,'PERIlt003',ps[i].m,ps[i].params['wf'])
			ps[0].m+=ps[i].m
			sim.remove(i)
			continue
	sim.move_to_com()


#-------------------
# output files
#-------------------

def initialize_collisions_file():
	f=open(coll_file,'w+')
	f.write('# time [yr]\tv0 [v_esc]\talpha [degrees]\tmtot [MEAR]\tgamma [1]\twft [%]\twfp [%]')
	f.write('\tx_CoM [AU]\ty_CoM [AU]\tz_CoM [AU]\tvx_CoM [AU/yr/2pi]\tvy_CoM [AU/yr/2pi]\tvz_CoM [AU/yr/2pi]')
	f.write('\tchi [rad]\tpsi [rad]\thz [AU^2/yr/2pi]')
	f.write('\tNbig [1]\tm1 [MEAR]\twf1 [%]\tm2 [MEAR]\twf2 [%]\tmfr [MEAR]\twffr [%]\n')
	f.close() 

def initialize_removed_particles_file():
	f=open(rmv_file,'w+')
	f.write('# time [yr]\tTYPE\tm [MEAR]\twf [%]\n')
	f.close()

def save_collision(params,xCoM,vCoM,chi,psi,h,survs,mfr,wffr):
	# position and velocity of the collision's CoM are given in the CoM system of coordinate of the simulation
	v0,alpha,mtot,gamma,wft,wfp=params
	f=open(coll_file,'a')
	f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(sim.t/2./np.pi,v0,alpha,mtot/MEAR,gamma,wft,wfp))
	for val in xCoM: f.write('\t{}'.format(val))
	for val in vCoM: f.write('\t{}'.format(val))
	f.write('\t{}\t{}\t{}'.format(chi,psi,h[2]))
	Nbig=len(survs)
	if Nbig==0: f.write('\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(Nbig,0,0,0,0,mfr/MEAR,wffr))
	if Nbig==1: f.write('\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(Nbig,survs[0][2]/MEAR,survs[0][3],0,0,mfr/MEAR,wffr))
	if Nbig==2: f.write('\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(Nbig,survs[0][2]/MEAR,survs[0][3],survs[1][2]/MEAR,survs[1][3],mfr/MEAR,wffr))
	f.close()

def save_data(t,ps,path):
	sim.move_to_hel()
	f=open('{}/{}.t{}'.format(path,path,int(np.around(t))),'w+')
	f.write('m [MSUN]\tR [AU]\ta [AU]\te [1]\tinc [rad]\tOmega [rad]\tomega [rad]\tM [rad]\twf [%]\n')
	line='{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'
	f.write(line.format(ps[0].m,ps[0].r,0.,0.,0.,0.,0.,0.,0.))
	for p in ps[1:]: f.write(line.format(p.m,p.r,p.a,p.e,p.inc,p.Omega,p.omega,p.M,p.params['wf']))
	sim.move_to_com()
	return f
	
def write_rmv_file(t,ty,m,wf):
	f=open(rmv_file,'a')
	f.write('{}\t{}\t{}\t{}\n'.format(t/2./np.pi,ty,m,wf))
	f.close()
	
def collision_snapshot(ps,indeces):
	previous_files=glob.glob(scenario+'/*.snapshot')
	collN=len(previous_files)+1
	new_snap='{}/collision_{}.snapshot'.format(scenario,collN)
	sim.move_to_hel()
	
	f=open(new_snap,'w+')
	f.write('# m [MSUN]\tR [AU]\ta [AU]\te [1]\tinc [rad]\tOmega [rad]\tomega [rad]\tM [rad]\twf [%]')
	f.write(' x [AU]\ty [AU]\tz [AU]\tvx [AU/yr/2pi]\tvy [AU/yr/2pi]\tvz [AU/yr/2pi]\n')
	f.write('# Colliding bodies\n')
	linea='{}'+'\t{}'*8
	lineb='\t{}'*6+'\n'
	for i in indeces:
		f.write(linea.format(ps[i].m,ps[i].r,ps[i].a,ps[i].e,ps[i].inc,ps[i].Omega,ps[i].omega,ps[i].M,ps[i].params['wf']))
		f.write(lineb.format(ps[i].x,ps[i].y,ps[i].z,ps[i].vx,ps[i].vy,ps[i].vz))
	f.write('# Other bodies\n')
	f.write(linea.format(ps[0].m,ps[0].r,0,0,0,0,0,0,0))
	f.write(lineb.format(0,0,0,0,0,0))
	for i in range(1,len(ps)-1):
		if i==0 or i==indeces[0] or i==indeces[1]: continue
		f.write(linea.format(ps[i].m,ps[i].r,ps[i].a,ps[i].e,ps[i].inc,ps[i].Omega,ps[i].omega,ps[i].M,ps[i].params['wf']))
		f.write(lineb.format(ps[i].x,ps[i].y,ps[i].z,ps[i].vx,ps[i].vy,ps[i].vz))
	f.close()	
	sim.move_to_com()
	
def save_progr(t,Np,t1,t0,dE):
	f=open(sp_file,'a+')
	f.write(' - t={:.0f} yr   N={}   partial: {:.2f} s   running: {:.2f} s   err_rel={:.2e}\n'.format(t,Np,t1,t0,dE))
	f.close()


#----------------------------------------------------------------------
#----------------------------------------------------------------------


#***************   INITIALIZE SIMULATION OUTPUT FILES   ***************

# old file warning
if new_sim_warning:
	import os
	import glob
	old_files=glob.glob(scenario+'/*.dat')+glob.glob(scenario+'/*.snapshot')
	if old_files!=[]:
		print('\n\n {}\n WARNING!!!\n \n {}\n'.format('~'*50,'~'*50))
		print(' Previos simulation file detected!\n')
		for f in old_files: print(f)
		user=str(input('\nRemove old files? (y,n) '))
		if user=='y':
			for f in old_files: os.remove(f)

# collisions list file
coll_file='{}/coll.dat'.format(scenario)
if glob.glob(coll_file)==[]: initialize_collisions_file()

# removed particles file
rmv_file='{}/rmv.dat'.format(scenario)
if glob.glob(rmv_file)==[]: initialize_removed_particles_file()

# save progress file
sp_file='{}/progress.{}'.format(scenario,scenario)


#***********************   SIMULATION SETUP   *************************
											     
sim = rebound.Simulation()

#integrator options
sim.integrator = "mercurius"
sim.ri_mercurius.hillfac = 5.
sim.dt=dt
sim.ri_ias15.min_dt = 1e-4 * sim.dt	# minimum timestep for when the close encounter is detected

#collision and boundary options
sim.collision = "direct"				# The way it looks for collisions
sim.collision_resolve_keep_sorted = 1
sim.boundary = "none"
sim.track_energy_offset = 1

#collision solver
sim.collision_resolve = collision_solver	# Custom collision solver

#reboundx
rebx = reboundx.Extras(sim)		# add the extra parameter water fraction 'wf'

#removing particles
rem_freq=1000				# checks for particles to be romeved every rem_freq time steps

#----------------------------------------------------------------------
#----------------------------------------------------------------------

#*****************************   INPUT   ******************************

start=np.loadtxt('{}/{}.{}'.format(scenario,scenario,t0),skiprows=1)

# --- Sun
sim.add(m=start[0,0],r=start[0,1],hash='Sun')
sim.particles[0].params['wf']=0.
# --- Planets/Embryos/Planetesimals
for i in range(1,len(start)):
	sim.add(m=start[i,0],r=start[i,1],a=start[i,2],e=start[i,3],inc=start[i,4],Omega=start[i,5],omega=start[i,6],M=start[i,7])
	sim.particles[i].params['wf']=start[i,8]

ps = sim.particles
sim.move_to_com()
E0 = sim.calculate_energy() # Calculate initial energy 

#----------------------------------------------------------------------
#----------------------------------------------------------------------		

#*****************************   OUTPUT   *****************************

if t0=='start': tmin=0.
if t0[0]=='t': tmin=int(t0[1:])*np.pi*2.

sim.t=tmin
tend=tmin+Dt*2.*np.pi
times = np.linspace(tmin,tend,Nsaves+1)[1:]

import time
clock0=time.time()

print(' - t={:.0f} yr   N={}   partial: {:.2f} s   running: {:.2f} s'.format(np.round(sim.t/2./np.pi),len(ps),0,0))
if save_progess: save_progr(np.round(sim.t/2./np.pi),len(ps),0,0,0)

for t in times:
	clock1=time.time()
	
	# removing particles stepping 
	tnext=sim.t+rem_freq*sim.dt
	while tnext<t:
		sim.integrate(tnext)
		remove_ps() 	# remove particles
		tnext+=rem_freq*sim.dt
	
	# saving data stepping 
	sim.integrate(t)
	remove_ps() 	# remove particles

	if save_checkpoints: save_data(sim.t/np.pi/2.,ps,scenario)
	
	clock2=time.time()
	E1 = sim.calculate_energy()
	
	print(' - t={:.0f} yr   N={}   partial: {:.2f} s   running: {:.2f} s   err_rel={:.2e}'.format(np.round(sim.t/2./np.pi),len(ps),clock2-clock1,clock2-clock0,(E1-E0)/E0))
	if save_progess: save_progr(np.round(sim.t/2./np.pi),len(ps),clock2-clock1,clock2-clock0,(E1-E0)/E0)
	
