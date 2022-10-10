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
import random 

#########
def estimate_nose(a,r,x,n):
    d=2*r
    print('n shape:',n.shape)
    exp_power=np.transpose(np.divide(1,n))
    print(exp_power)
    print('exp_power shape:',exp_power.shape)
    base=1-np.power(np.divide(np.subtract(x,a),a),2).T
    print('base shape:',base.shape)
    return 0.5*d*np.power(base,exp_power).T

def estimate_tail(a,b,c,r,x,theta):
    d=2*r
    theta=theta*math.pi/180
    y1=0.5*d
    y2= np.power((x-a-b),2)*((3*d)/(2*c*c) - math.tan(theta)/c)
    y3= ((d/(c*c*c)) - math.tan(theta)/(c*c))*np.power((x-a-b),3)
    return (y1-y2+y3)

def check_nose_validity(myring_nose_loc,a,r):
    return True
def check_tail_validity(myring_tai_loc,c,r):
    return True


def print_design(): 
    plt.plot(x_n, y)
    plt.scatter(x_n,y)
    return 0





def run_single_design(): 
    a = 300; b=500; c=1000; r =100; n=5; theta=50
    d=2*r
    x_n = np.array([0, a / 5, 2 * a / 5, 3 * a / 5, 4 * a / 5, a]);
    x_t = np.array([a + b, a + b + c / 5, a + b + 2 * c / 5, a + b + 3 * c / 5, a + b + 4 * c / 5, a + b + c]);
    x_b = np.array([a, a + b * 1 / 6, a + b * 2 / 6, a + b * 3 / 6, a + b * 4 / 6, a + b * 5 / 6, a + b])
    y_b = np.array([r, r, r, r, r, r, r])
    y_b = y_b.astype('float64')
    
    x_ref=np.array([0,a,a+b,a+b+c])
    y_ref=np.array([0,r,r,0])
    y = estimate_nose(a, d, x_n, n)
    z = estimate_tail(a, b, c, d, x_t, theta)

    figname = './fig_hull/GA_a' + str(round(a, 2)) + 'c_' + str(round(c, 2)) + 'n_' + str(
        round(n, 2)) + '_theta_' + str(round(theta, 2)) + '.png'
    plt.figure(figsize=(10, 3))
    plt.plot(x_n, y)
    plt.scatter(x_n,y)
    plt.plot(x_t, z)
    plt.scatter(x_t,z)
    plt.plot(x_b, y_b)
    plt.plot(x_n, -1 * y)
    plt.plot(x_t, -1 * z)
    plt.plot(x_b, -1 * y_b)
    plt.plot(x_n, y)
    plt.plot(x_t, z)
    plt.plot(x_b, y_b)
    
    #plt.savefig(figname)
    plt.plot(x_ref, y_ref)
    plt.plot(x_ref, -1*y_ref)

    plt.vlines(a, -1*r, r)
    plt.vlines(a+b, -1*r, r)
    plt.show()
    plt.close()


def get_pieces(a,pieces):
    
    a_n=np.repeat(a/pieces,pieces,axis=1)
    index=np.arange(pieces)+1
    #print('Index is:',index)
    _loc_= np.multiply(a_n,index)
    
    return _loc_



def run_multiple_designs(a=100,b=50,c=200,r=25): 
       
    fix_param= np.array([a,b,c,r])
    bounds=[[0.1,5],[1,50]]
    grid = 3
    pieces=5
    
    print('bound:',bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1])
 
    X1 = np.linspace(bounds[0][0], bounds[0][1], grid)
    X2 = np.linspace(bounds[1][0], bounds[1][1], grid)
    
    x1, x2 = np.meshgrid(X1, X2)
    X = np.hstack((x1.reshape(grid*grid,1),x2.reshape(grid*grid,1)))
    Y= np.multiply(np.ones((X.shape[0],4)),fix_param)
    ds= np.concatenate((Y,X),axis=1)
    print(ds)
    #print(d_p)
    print(ds.shape)
    print(X.shape)
    
    nose_x= get_pieces(ds[:,0].reshape(-1,1), pieces)
    _t_p_=  get_pieces(ds[:,2].reshape(-1,1), pieces)
    tail_x= a+b+_t_p_
    print(nose_x)
    print(_t_p_)
    print(tail_x)
    
    ref= [r/5,2*r/5, 3*r/5,4*r/5,r]
    nose_y= estimate_nose(a,r,nose_x,ds[:,4])
    print('nose_x shape:',nose_x.shape,'Nose_y:',nose_y.shape)
    for i in range(nose_x.shape[0]): 
        print()
    
    
    
    
    
    
    return ds




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #run_single_design()
    run_multiple_designs()
    """
    power=np.array([3,2])
    base=np.array([[1,2,3],[4,5,6]]).T
    print(base.T)
    print('power:',power.shape,'base:',base.shape)
    d= np.power(base,power)
    print(d.T)
    """ 
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
