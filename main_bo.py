import argparse
import os
import numpy as np
from utils import *
import pandas as pd
import shutil
import glob 
import subprocess
import time
#from run_dexof import *
import sys 
from cfd_sim.run_dexof import *
from cfd_sim.dexof_reader_class import parse_dex_file
import GPyOpt
from subprocess import PIPE, run
import random
from numpy.random import seed
from single_myring import myring_hull_ds


# parameter edit for running the file 
a=555; b=2664;c=500;r=513; a_ext=0;c_ext=0
data_file_name='./data_bo/bo_LCB_design1'
##







sys.dont_write_bytecode = True
cad_storage_name= './cad_sim/design_points.csv'
cfd_storage_name= './cfd_sim/design_points.csv'


src= './cad_sim/stl_repo'
dst='./cfd_sim/stl_cfd'


hull_ds= myring_hull_ds()
drag_storage=[73.631122]  
    


def delete_dir(loc):
    print('*Deleted directory:',loc)
    shutil.rmtree(loc)

def copy_dir(src,dst):
	print('*Copied directory from',src,'to destination:',dst)
	shutil.copytree(src, dst)

def deletefiles(loc):
	print('Deleted files from location:',loc)
	file_loc= loc+'/*'
	files = glob.glob(file_loc)
	for f in files:
		os.remove(f)

def copy_file(src,dst):
	print('*Copied file from',src,'to destination:',dst)
	shutil.copy(src, dst)

def save_design_points(x):
    np.savetxt(cad_storage_name,x,  delimiter=',')
    np.savetxt(cfd_storage_name,x,  delimiter=',')

def run_cad_cfd(x):
	print('shape of x:',x.shape)
	feasible=hull_ds.check_feasibility_design(a=a,b=b,c=c,r=r,n=x[0][0],theta=x[0][1],a_ext=x[0][2],c_ext=x[0][3],pieces=10)
	if feasible==0:
		    return max(drag_storage)

	save_design_points(np.array([x[0][0],x[0][1],a,b,c,r,x[0][2],x[0][3]]))
	
	delete_dir(dst)
	subprocess.call('./cad_sim/run_cad.sh')
	copy_dir(src,dst)
	deletefiles(src)
      
	prev = os.path.abspath(os.getcwd()) # Save the real cwd
	print('prev is',prev)
	cfd_sim_path= prev+'/cfd_sim'
	print('func path is:',cfd_sim_path)
	os.chdir(cfd_sim_path)
	result = main_run()
	drag_storage.append(result)
	print('****Drag drag_storage:',drag_storage)
	os.chdir(prev)
	return result



def run_bo(run_id=0,aquistion='EI',seeds=0):
	
	################################################
	deletefiles('./cad_sim/fig_hull')
	bounds = [{'name': 'n', 'type': 'continuous', 'domain': (1,50)},
            {'name': 'theta', 'type': 'continuous', 'domain': (1,50)},
            {'name': 'a_ext', 'type': 'continuous', 'domain': (0,0)},
            {'name': 'c_ext', 'type': 'continuous', 'domain': (0,0)}]    
	print('Bound is:',bounds)
	max_time  = None 
	max_iter  = 50
	num_iter=10
	batch= int(max_iter/num_iter)
	#tolerance = 1e-8     # distance between two consecutive observations 
	  
	#################################################
	already_run = len(glob.glob(data_file_name))
	print('file exist?:',already_run)

	print('Batch is:',batch)



	seed(seeds)
	for i in range(num_iter): 
	
	 if already_run==1:
	   evals = pd.read_csv(data_file_name, index_col=0, delimiter="\t")
	   Y = np.array([[x] for x in evals["Y"]])
	   X = np.array(evals.filter(regex="var*"))
	   myBopt2D = GPyOpt.methods.BayesianOptimization(run_cad_cfd, bounds,model_type = 'GP',X=X, Y=Y,
                                              acquisition_type=aquistion, normalize_Y=False, 
                                              exact_feval = True) 

	   print('In other runs run')
	 else: 
	   myBopt2D = GPyOpt.methods.BayesianOptimization(f=run_cad_cfd,
                                              domain=bounds,
                                              model_type = 'GP',
                                              acquisition_type=aquistion,  normalize_Y=False, 
                                              exact_feval = True) 
	   already_run=1
	   print('In 1st run')
	 print('------Running batch is:',i) 
   
 # --- Run the optimization 
	 try:
	  myBopt2D.run_optimization(batch,verbosity=True)
	  pass   
	 except KeyboardInterrupt:
	  pass
	 sim_data_x= myBopt2D.X;
	 myBopt2D.save_evaluations(data_file_name)
	myBopt2D.plot_acquisition()  
	myBopt2D.plot_convergence()





if __name__=='__main__':
	#run=[1,2,3,4,5]; seeds=[11,13,17,19,21]
	run=[1]; seeds=[17]	
	aqu1='EI'; aqu2='LCB'

	

	for i in range(len(run)):
		#run_bo(run[i],aqu1,seeds[i])
		run_bo(run[i],aqu2,seeds[i])  
		#print('In BO run')
	
	#myBopt2D.plot_acquisition()  
	#myBopt2D.plot_convergence()

	
	
