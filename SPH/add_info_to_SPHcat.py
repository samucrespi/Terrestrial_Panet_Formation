# I used this routine to add the mtot to the output_info.txt file

import glob
for f in glob.glob('SPH_catalogue/*'): open(glob.glob(f+'/output_info.txt')[0],'a').write('Total_mass: {}'.format(open(glob.glob(f+'/spheres_ini_log')[0]).readlines()[62].split()[3]))
