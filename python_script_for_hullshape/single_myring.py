# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 12:10:56 2022

@author: Vardhan Harsh (Vanderbilt University)
"""
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt
import math 

#########
class myring_hull_ds(): 
   def __init__(self):
       pass
   
        
   def estimate_nose(self,a,r,x,n):
    d=2*r
    #print('n shape:',n.shape)
    exp_power=np.transpose(np.divide(1,n))
    #print(exp_power)
    #print('exp_power shape:',exp_power.shape)
    base=1-np.power(np.divide(np.subtract(x,a),a),2).T
    #print('base shape:',base.shape)
    return 0.5*d*np.power(base,exp_power).T

   def estimate_tail(self,a,b,c,r,x,theta):
    d=2*r
    theta=theta*math.pi/180
    #print('Shape of theta:',theta.shape,'x.shape:',x.shape)
    y1=0.5*d
    z= np.tan(theta)/c
    #print('Shape of z:',z.shape)
    zz= ((3*d)/(2*c*c)-z)
    #print('zz shape:',zz.shape)
    y2= np.multiply(np.power((x-a-b),2).T,zz).T
    #print('y2:',y2.shape)
    zzz= ((d/(c*c*c))-(np.tan(theta)/(c*c)))
    y3= np.multiply(np.power((x-a-b),3).T,zzz).T
    #print('y3:',y3.shape)
    return (y1-y2+y3)

   def run_single_design(self,a = 300, b=500, c=1000, r =1000, n=0.5, theta=1): 
    x_n = np.atleast_2d(np.array([0, a / 5, 2 * a / 5, 3 * a / 5, 4 * a / 5, a]));
    x_t = np.atleast_2d(np.array([a + b, a + b + c / 5, a + b + 2 * c / 5, a + b + 3 * c / 5, a + b + 4 * c / 5, a + b + c]));
    x_b =np.array([a, a + b * 1 / 6, a + b * 2 / 6, a + b * 3 / 6, a + b * 4 / 6, a + b * 5 / 6, a + b])
    y_b = np.array([r, r, r, r, r, r, r])
    y_b = y_b.astype('float64')
    
    x_ref=np.array([0,a,a+b,a+b+c])
    y_ref=np.array([0,r,r,0])
    y = self.estimate_nose(a, r, x_n, n)
    z = self.estimate_tail(a, b, c, r, x_t, theta)

    #figname = './fig_hull/GA_a' + str(round(a, 2)) + 'c_' + str(round(c, 2)) + 'n_' + str(
    #    round(n, 2)) + '_theta_' + str(round(theta, 2)) + '.png'
    plt.figure(figsize=(10, 3))
    print('x_n',x_n.shape,'y:',y.shape)
    plt.plot(x_n[0,:], y[0,:])
    plt.scatter(x_n,y)
    plt.plot(x_t[0,:], z[0,:])
    plt.scatter(x_t,z)
    plt.plot(x_b, y_b)
    plt.plot(x_n[0,:], -1 * y[0,:])
    plt.plot(x_t[0,:], -1 * z[0,:])
    plt.plot(x_b, -1 * y_b)
  
    
    #plt.savefig(figname)
    plt.plot(x_ref, y_ref)
    plt.plot(x_ref, -1*y_ref)

    plt.vlines(a, -1*r, r)
    plt.vlines(a+b, -1*r, r)
    plt.show()
    plt.close()


   def get_pieces(self,a,pieces): 
    a_n=np.repeat(a/pieces,pieces+1,axis=1)
    index=np.arange(pieces+1)
    #print('Index is:',index)
    _loc_= np.multiply(a_n,index)
    return _loc_

   def run_multiple_designs(self,grid,pieces):
    print('grid is:',grid,'pieces are:',pieces)
    a=self.a; b=self.b; c=self.c; r=self.r
    ref=True; plot_sketch=False
    
    fix_param= np.array([a,b,c,r])
    bounds=[[0.1,10],[1,50]]
    
    
    ref_design_Y=np.arange(pieces+1)* r/pieces
    print('ref_design Y:', ref_design_Y)
    effective_ref_design_Y= ref_design_Y[1:-1].reshape(1,-1)
    
    x_ref=np.array([0,a,a+b,a+b+c])
    y_ref=np.array([0,r,r,0])
    
    x_body=np.array([a,a+b])
    y_body=np.array([r,r])
    
    print('bound:',bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1])
 
    X1 = np.linspace(bounds[0][0], bounds[0][1], grid)
    X2 = np.linspace(bounds[1][0], bounds[1][1], grid)
    
    x1, x2 = np.meshgrid(X1, X2)
    X = np.hstack((x1.reshape(grid*grid,1),x2.reshape(grid*grid,1)))
    Y= np.multiply(np.ones((X.shape[0],4)),fix_param)
    ds= np.concatenate((Y,X),axis=1)
    print(ds)
    #print(d_p)
    print('DS:',ds.shape)
    print('X:',X.shape)
    
    nose_x= self.get_pieces(ds[:,0].reshape(-1,1), pieces)
    _t_p_=  self.get_pieces(ds[:,2].reshape(-1,1), pieces)
    tail_x= a+b+_t_p_
    print('Nose_X:',nose_x.shape)
    print('Tail_X',tail_x.shape)
    
    #ref= [r/5,2*r/5, 3*r/5,4*r/5]
    nose_y= self.estimate_nose(a,r,nose_x,ds[:,4])
    tail_y= self.estimate_tail(a,b,c,r,tail_x,ds[:,5]) 
    
    
    print(nose_y[:,1:-1].shape,'tail:',tail_y[:,1:-1])
    nose_interference_matrix=np.greater_equal(nose_y[:,1:-1],effective_ref_design_Y)
    tail_interference_matrix=np.greater_equal(tail_y[:,1:-1],np.flip(effective_ref_design_Y))
    
    print('nose_interference:',nose_interference_matrix)
    print('tail_interference:',tail_interference_matrix)
    
    nose_interference=np.all(nose_interference_matrix,axis=1)
    tail_interference=np.all(tail_interference_matrix,axis=1)
    feasible_design= np.logical_and(nose_interference,tail_interference)
    print('N I:',nose_interference,'T I:',tail_interference,'feasible design:',feasible_design)
    
    
    print('nose_x shape:',nose_x.shape,'Nose_y:',nose_y.shape)
    print('tail_x shape:',tail_x.shape,'Tail_y:',tail_y.shape)
    if plot_sketch==True:
     for i in range(nose_x.shape[0]): 
        plt.figure(figsize=(10, 3))
        plt.plot(nose_x[i,:], nose_y[i,:])
        plt.scatter(nose_x[i,:], nose_y[i,:])
        plt.plot(nose_x[i,:], -1*nose_y[i,:])
        
        plt.plot(x_body, y_body)
        plt.plot(x_body, -1*y_body)
        
        plt.plot(tail_x[i,:], tail_y[i,:])
        plt.scatter(tail_x[i,:], tail_y[i,:])
        plt.plot(tail_x[i,:], -1*tail_y[i,:])
        if feasible_design[i]==True: 
            plt.title('Feasible design')
        else: 
            plt.title('Infeasible design')
        
        if ref==True: 
            plt.plot(x_ref, y_ref)
            plt.plot(x_ref, -1*y_ref)
            plt.vlines(a, -1*r, r)
            plt.vlines(a+b, -1*r, r)
        plt.show()
        plt.close()
    feasible_ds= ds[feasible_design]
    infeasible_ds=ds[np.logical_not(feasible_design)]
    print('DS:',ds.shape, 'Feasible ds:',feasible_ds.shape,'Infeasible DS:',infeasible_ds.shape)
    
    return ds,feasible_ds, infeasible_ds 


   def check_feasibility_design(self,a,b,c,r,n,theta,a_ext,c_ext,pieces):
    #print('pieces are:',pieces)
    
    ref=True; plot_sketch=True

    ref_design_Y=np.arange(pieces+1)* r/pieces
    #print('ref_design Y:', ref_design_Y)
    effective_ref_design_Y= ref_design_Y[1:-1].reshape(1,-1)
    #print('Effective ref_design Y:', effective_ref_design_Y)
    x_ref=np.array([a_ext,a+a_ext,a+a_ext+b,a+a_ext+b+c])
    y_ref=np.array([0,r,r,0])
    
    
    
    ds= np.atleast_2d(np.array([a,b,c,r,n,theta]))

    
    _n_p_= self.get_pieces(ds[:,0].reshape(-1,1), pieces)
    _t_p_=  self.get_pieces(ds[:,2].reshape(-1,1), pieces)
    
    nose_x= a_ext+_n_p_
    tail_x= a+a_ext+b+_t_p_

    a_eff= a+a_ext
    c_eff= c+c_ext
    nose_y= self.estimate_nose(a_eff,r,nose_x,n)
    tail_y= self.estimate_tail(a_eff,b,c_eff,r,tail_x,theta) 
    
    nose_interference_matrix=np.greater_equal(nose_y[:,1:-1],effective_ref_design_Y)
    tail_interference_matrix=np.greater_equal(tail_y[:,1:-1],np.flip(effective_ref_design_Y))

    
    nose_interference=np.all(nose_interference_matrix,axis=1)
    tail_interference=np.all(tail_interference_matrix,axis=1)
    feasible_design= np.logical_and(nose_interference,tail_interference)

    if plot_sketch==True:
        
     _aext_p_= self.get_pieces(np.atleast_2d(np.array(a_ext)), pieces)     
     ext_nose_y= self.estimate_nose(a_eff,r,_aext_p_,ds[:,4])
     
     _c_ext_p_=  self.get_pieces(np.atleast_2d(np.array(c_ext)), pieces) 
     ext_tail_y= self.estimate_tail(a_eff,b,c_eff,r,a_eff+b+c+_c_ext_p_,ds[:,5]) 
        
     for i in range(nose_x.shape[0]): 
        plt.figure(figsize=(10, 3))
        plt.plot(nose_x[i,:], nose_y[i,:])
        plt.scatter(nose_x[i,:], nose_y[i,:])
        plt.plot(nose_x[i,:], -1*nose_y[i,:])
        
        plt.plot(_aext_p_[0], ext_nose_y[i,:])
        plt.plot(_aext_p_[0], -1*ext_nose_y[i,:])
        
        plt.plot(a_eff+b+c+_c_ext_p_[0], ext_tail_y[i,:])
        plt.plot(a_eff+b+c+_c_ext_p_[0], -1*ext_tail_y[i,:])
        #plt.scatter(nose_x[i,:], nose_y[i,:])
        
        #plt.plot(x_body, y_body)
        #plt.plot(x_body, -1*y_body)
        
        plt.plot(tail_x[i,:], tail_y[i,:])
        plt.scatter(tail_x[i,:], tail_y[i,:])
        plt.plot(tail_x[i,:], -1*tail_y[i,:])
        if feasible_design[i]==True: 
            plt.title('Feasible design')
        else: 
            plt.title('Infeasible design')
        
        if ref==True: 
            plt.plot(x_ref, y_ref)
            plt.plot(x_ref, -1*y_ref)
            #plt.vlines(a_ext+a, -1*r, r)
            #plt.vlines(a_ext+a+b, -1*r, r)
        plt.show()
        plt.close()
    feasible_ds= ds[feasible_design]

    return feasible_ds.shape[0] 






   def run_extened_nosetail_designs(self,grid,pieces):
    print('grid is:',grid,'pieces are:',pieces)
    a=self.a; b=self.b; c=self.c; r=self.r;a_ext=self.a_ext;c_ext=self.c_ext
    ref=True; plot_sketch=True
    
    fix_param= np.array([a,b,c,r])
    bounds=[[0.1,10],[1,50]]
    
    
    ref_design_Y=np.arange(pieces+1)* r/pieces
    print('ref_design Y:', ref_design_Y)
    effective_ref_design_Y= ref_design_Y[1:-1].reshape(1,-1)
    print('Effective ref_design Y:', effective_ref_design_Y)
    x_ref=np.array([a_ext,a+a_ext,a+a_ext+b,a+a_ext+b+c])
    y_ref=np.array([0,r,r,0])
    
    #x_body=np.array([a,a+b])
    #y_body=np.array([r,r])
    
    print('bound:',bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1])
 
    X1 = np.linspace(bounds[0][0], bounds[0][1], grid)
    X2 = np.linspace(bounds[1][0], bounds[1][1], grid)
    
    x1, x2 = np.meshgrid(X1, X2)
    X = np.hstack((x1.reshape(grid*grid,1),x2.reshape(grid*grid,1)))
    Y= np.multiply(np.ones((X.shape[0],4)),fix_param)
    ds= np.concatenate((Y,X),axis=1)
    print(ds)
    #print(d_p)
    print('DS:',ds.shape)
    print('X:',X.shape)
    
    _n_p_= self.get_pieces(ds[:,0].reshape(-1,1), pieces)
    _t_p_=  self.get_pieces(ds[:,2].reshape(-1,1), pieces)
    
    nose_x= a_ext+_n_p_
    tail_x= a+a_ext+b+_t_p_
    print('Nose_X:',nose_x.shape)
    print('Tail_X',tail_x.shape)
    a_eff= a+a_ext
    c_eff= c+c_ext
    #ref= [r/5,2*r/5, 3*r/5,4*r/5]
    nose_y= self.estimate_nose(a_eff,r,nose_x,ds[:,4])
    tail_y= self.estimate_tail(a_eff,b,c_eff,r,tail_x,ds[:,5]) 
    
    
    
    print(nose_y[:,1:-1].shape,'tail:',tail_y[:,1:-1])
    nose_interference_matrix=np.greater_equal(nose_y[:,1:-1],effective_ref_design_Y)
    tail_interference_matrix=np.greater_equal(tail_y[:,1:-1],np.flip(effective_ref_design_Y))
    
    print('nose_interference:',nose_interference_matrix)
    print('tail_interference:',tail_interference_matrix)
    
    nose_interference=np.all(nose_interference_matrix,axis=1)
    tail_interference=np.all(tail_interference_matrix,axis=1)
    feasible_design= np.logical_and(nose_interference,tail_interference)
    print('N I:',nose_interference,'T I:',tail_interference,'feasible design:',feasible_design)
    
    
    
    
    print('nose_x shape:',nose_x.shape,'Nose_y:',nose_y.shape)
    print('tail_x shape:',tail_x.shape,'Tail_y:',tail_y.shape)
    if plot_sketch==True:
        
     _aext_p_= self.get_pieces(np.atleast_2d(np.array(a_ext)), pieces)     
     ext_nose_y= self.estimate_nose(a_eff,r,_aext_p_,ds[:,4])
     
     _c_ext_p_=  self.get_pieces(np.atleast_2d(np.array(c_ext)), pieces) 
     ext_tail_y= self.estimate_tail(a_eff,b,c_eff,r,a_eff+b+c+_c_ext_p_,ds[:,5]) 
     print('ext_nose_y:',ext_nose_y.shape,'_a_ext_p_:',_aext_p_.shape)
     print('ext_tail_y:',ext_nose_y.shape,'_c_ext_p_:',_c_ext_p_.shape)
        
     for i in range(nose_x.shape[0]): 
        plt.figure(figsize=(10, 3))
        plt.plot(nose_x[i,:], nose_y[i,:])
        plt.scatter(nose_x[i,:], nose_y[i,:])
        plt.plot(nose_x[i,:], -1*nose_y[i,:])
        
        plt.plot(_aext_p_[0], ext_nose_y[i,:])
        plt.plot(_aext_p_[0], -1*ext_nose_y[i,:])
        
        plt.plot(a_eff+b+c+_c_ext_p_[0], ext_tail_y[i,:])
        plt.plot(a_eff+b+c+_c_ext_p_[0], -1*ext_tail_y[i,:])
        #plt.scatter(nose_x[i,:], nose_y[i,:])
        
        #plt.plot(x_body, y_body)
        #plt.plot(x_body, -1*y_body)
        
        plt.plot(tail_x[i,:], tail_y[i,:])
        plt.scatter(tail_x[i,:], tail_y[i,:])
        plt.plot(tail_x[i,:], -1*tail_y[i,:])
        if feasible_design[i]==True: 
            plt.title('Feasible design')
        else: 
            plt.title('Infeasible design')
        
        if ref==True: 
            plt.plot(x_ref, y_ref)
            plt.plot(x_ref, -1*y_ref)
            #plt.vlines(a_ext+a, -1*r, r)
            #plt.vlines(a_ext+a+b, -1*r, r)
        plt.show()
        plt.close()
    feasible_ds= ds[feasible_design]
    infeasible_ds=ds[np.logical_not(feasible_design)]
    print('DS:',ds.shape, 'Feasible ds:',feasible_ds.shape,'Infeasible DS:',infeasible_ds.shape)
    
    return ds,feasible_ds, infeasible_ds 











# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    a=[10]
    print(max(a))
    hull_ds= myring_hull_ds()
    
    
    feasible=hull_ds.check_feasibility_design(a=555,b=2664,c=500,r=513,n=1,theta=15,a_ext=0,c_ext=0,pieces=10)
    
    print('feasible:',feasible)
    
    
    
    """
    hull_ds= myring_hull_ds(a=1000,b=200,c=1000,r=50)
    
    ds,feasible_ds,_=hull_ds.run_multiple_designs(200,10)
    if feasible_ds.shape[0]>1:
     n_max,n_min=np.max(feasible_ds[:,4]),np.min(feasible_ds[:,4])
     theta_max,theta_min=np.max(feasible_ds[:,5]),np.min(feasible_ds[:,5])
    
     plt.figure(figsize=(10, 3))
     plt.scatter(ds[:,4],ds[:,5],label='infeasible')
     plt.scatter(feasible_ds[:,4], feasible_ds[:,5],label='feasible')
     plt.legend(loc='upper center')
     plt.xlabel('nose(n)')
     plt.ylabel('tail(theta)')
     plt.hlines(theta_min, 0.1, 10)
     plt.hlines(theta_max, 0.1, 10)
     plt.vlines(n_min, 1, 50)
     plt.vlines(n_max, 1, 50)
     plt.show()
     plt.close()
    
    if feasible_ds.shape[0]>1:
     for i in range(1):
      idx=np.random.randint(0,feasible_ds.shape[0])
      print('idx is:',idx)
      hull_ds.run_single_design(feasible_ds[idx][0],feasible_ds[idx][1],feasible_ds[idx][2],feasible_ds[idx][3],feasible_ds[idx][4],feasible_ds[idx][5])
    
    """