# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 12:33:15 2022

@author: Vardhan Harsh (Vanderbilt University)
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import random 




def estimate_nose(a,d,x,n):
    
    y=0.5*d*np.power((1-np.power(((x-a)/a),2)),(1/n))
    return y

def estimate_tail(a,b,c,d,x,theta):
    
    y1=0.5*d 
    y2= np.power((x-a-b),2)*((3*d)/(2*c*c) - math.tan(theta)/c)
    y3= ((d/(c*c*c)) - math.tan(theta)/(c*c))*np.power((x-a-b),3)
    return (y1-y2+y3)    

def degree_radian(theta):
    
    return theta*math.pi/180
        

def reynolds(vel,dia): 
    rho= 1027
    vel= vel
    mu= 0.00089
    cl=dia
    reyn=(rho*vel*cl)/mu
    return reyn
    
def hoernerfactor(length,dia):
     return 1+1.5*np.power((dia/length),1.5) + 7* np.power((dia/length),3)

def correlationline(reyn):
    cf= 0.075/(math.log((reyn-2)*(reyn-2)))
    return cf

def surfarea(length,dia):
    return 2*(math.pi)*dia*0.5*length
 
def calculate_drag(cf,surfarea,vel):
    rho=1027
    return 0.5*rho*cf*surfarea*vel*vel
    

#a=340.93;c=456.48;n=10;theta=1      #BO_EI 
#a=573;c=527.19;n=10;theta=1         #BO_LCB
#a=342.3;c=494.01;n=11.33;theta=3.90 #GA 
#a=432.66;c=51.15;n=4.19;theta=12.52 #VMC
#a=545.65;c=549.52;n=1.5;theta=14.46 #LHC
a=555;c=500;n=1.98;theta=46.36 #NM
b=2664;r=513
print('b is:',b)
#b=1369.7788
d=2*r  




#n=n/10   
dp=np.array([n,theta])


col='k'

x_n=np.array([0,a/5,2*a/5,3*a/5,4*a/5,a]);
x_t=np.array([a+b,a+b+c/5,a+b+2*c/5,a+b+3*c/5,a+b+4*c/5,a+b+5*c/5]);

x_b=np.array([a,a+b]); y_b=np.array([r,r]);y_b_n=[-1*r,-1*r]
y= estimate_nose(a,d,x_n,n)
z= estimate_tail(a,b,c,d,x_t,degree_radian(theta))

plt.plot(x_n,y, color=col,linewidth=1)
plt.plot(x_t,z,color=col,linewidth=1)
plt.plot(x_b,y_b,color=col,linewidth=1)
plt.plot(x_n,-1*y,color=col,linewidth=1)
plt.plot(x_t,-1*z,color=col,linewidth=1)

plt.plot(x_b,-1*y_b,color=col,linewidth=1)
#plt.xlim(-20,2005)
#plt.ylim(-105,105)
plt.show()