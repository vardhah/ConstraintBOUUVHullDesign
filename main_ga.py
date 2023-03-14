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
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.factory import get_termination
from pymoo.algorithms.soo.nonconvex.nelder_mead import NelderMead
from pymoo.core.problem import ElementwiseProblem

# parameter edit for running the file 
a=555; b=2664;c=500;r=513;                    # fixed constrained for optimization. 
data_file_name='./data/GA_exp1'               # change data file name for each experiment.
external_optim= False                         # change if external optimization is allowed.
##
drag_storage=[97.137896]                      # drag force realted to intial feasible design.

sys.dont_write_bytecode = True
cad_storage_name= './cad_sim/design_points.csv'
cfd_storage_name= './cfd_sim/design_points.csv'


src= './cad_sim/stl_repo'
dst='./cfd_sim/stl_cfd'


hull_ds= myring_hull_ds()

    


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




class CFDProblem(ElementwiseProblem):
    def __init__(self,filename,ext):
        self.flag=0; self.data=None; self.filename=filename; self.external_optim=ext
        if self.external_optim==False: 
         super().__init__(n_var=2,
                         n_obj=1,
                         n_constr=1,
                         xl=np.array([1,1]),
                         xu=np.array([50,50]))
        else: 
         super().__init__(n_var=4,
                         n_obj=1,
                         n_constr=1,
                         xl=np.array([1,1,0,0]),
                         xu=np.array([50,50,2500,2500]))
  
    def _evaluate(self, x, out, *args, **kwargs):
        print('---> x is:',x)
        if self.external_optim==False:
          a_ext=0; c_ext=0
          feasible=hull_ds.check_feasibility_design(a=a,b=b,c=c,r=r,n=x[0],theta=x[1],a_ext=a_ext,c_ext=c_ext,pieces=10)
        else: 
          feasible=hull_ds.check_feasibility_design(a=a,b=b,c=c,r=r,n=x[0],theta=x[1],a_ext=x[2],c_ext=x[3],pieces=10)  
        if feasible==0:
        	   feasibility_val= 1 
        else:  
                   feasibility_val= -1  
       
        if feasible==0:
            out["F"] = max(drag_storage)
        else:     
            save_design_points(np.array([x[0],x[1],a,b,c,r,a_ext,c_ext]))
            delete_dir(dst)
            subprocess.call('./cad_sim/run_cad.sh')
            copy_file(cad_storage_name,cfd_storage_name)
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
            out["F"] = [result]
        out["G"] = feasibility_val
     
        
        if self.flag==0:
        	self._data= np.append(x,out["F"] ).reshape(1,-1) ; self.flag=1
        else: 
        	self._data= np.concatenate((self._data,np.append(x,out["F"]).reshape(1,-1)),axis=0)
        	np.savetxt(self.filename,self._data,  delimiter=',')




class DummyProblem(ElementwiseProblem):

    def __init__(self,d,tl,filename):
        self.d=d;self.tl=tl; self.flag=0; self.data=None; self.filename=filename
        
        super().__init__(n_var=4,
                         n_obj=1,
                         n_constr=0,
                         xl=np.array([10,10,10,1]),
                         xu=np.array([573,573,50,50]))

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = [x[0]+x[1]+x[2]+x[3]+self.d+self.tl]
        #out["G"] = [g1]
        if self.flag==0:
        	self._data= np.append(x,out["F"][0]).reshape(1,-1) ; self.flag=1
        else: 
        	self._data= np.concatenate((self._data,np.append(x,out["F"][0]).reshape(1,-1)),axis=0)
        	np.savetxt(self.filename,self._data,  delimiter=',')





def run_pymoo(run_id=0,optimiser='GA',seeds=0):
	###########################
	n=100;
	pop_size=10
	#################################################
	#define CFD problem as pymoo problem instance 
	problem = CFDProblem(data_file_name,external_optim) 
	#problem = DummyProblem(D,tl,data_file_name)     
	already_run = len(glob.glob(data_file_name))
	print('file exist?:',already_run)

	###########Genetic algorithm ######################
	if optimiser=='GA':
		algorithm = GA(pop_size=pop_size,eliminate_duplicates=True)
		termination = get_termination("n_eval", n)
		res = minimize(problem,algorithm,termination,seed=seeds,verbose=True, save_history=True)
	##################################
	else:
		print('****No valid optimiser****')
		

if __name__=='__main__':
	aqu1='GA';
	run=[1]; seeds=[17]
	for i in range(len(run)):
		run_pymoo(run[i],aqu1,seeds[i])
		
	

