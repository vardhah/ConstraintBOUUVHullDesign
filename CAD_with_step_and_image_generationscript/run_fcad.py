# use: "exec(open('run_fcad.py').read())" to run this in FreeCAD console to see effects
#all units in mm
import numpy as np 
import math
import PartDesignGui

##Just this part neeeds editing
a=300; b=400;c=20;r=50;n=10; theta=30;

step_file_name= "design_1" 
image_file_name= "design_1"

##Dont edit below it

step_file="C:/Users/HPP/Desktop/sketch/designs/CFD/STEP/"+step_file_name+".step"
image_file= "C:/Users/HPP/Desktop/sketch/designs/CFD/PNG/"+image_file_name+".png"

###



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



d=2*r

x_n=np.array([a/5,2*a/5,3*a/5,4*a/5]);
x_t=np.array([a+b+c/5,a+b+2*c/5,a+b+3*c/5,a+b+4*c/5]);

y= estimate_nose(a,d,x_n,n)
z= estimate_tail(a,b,c,d,x_t,degree_radian(theta))
#print('y is:',y)
#print('z is:',z)

_a= str(a)+'mm'
_b= str(b)+'mm'
_c= str(c)+'mm'
_r= str(r)+'mm'
_n1= str(y[0])+'mm'
_n2= str(y[1])+'mm'
_n3= str(y[2])+'mm'
_n4= str(y[3])+'mm'

_t1= str(z[0])+'mm'
_t2= str(z[1])+'mm'
_t3= str(z[2])+'mm'
_t4= str(z[3])+'mm'


document = App.openDocument("C:/Users/HPP/Desktop/sketch/designs/CFD/Myring_hull_spreadsheet.FCStd")

#Update parameters
sheet = document.getObjectsByLabel('Parameters')[0]
sheet.set("myring_a", _a)
sheet.set("myring_b", _b)
sheet.set("myring_c", _c)
sheet.set("myring_r", _r)
sheet.set("nose1_y", _n1)
sheet.set("nose2_y", _n2)
sheet.set("nose3_y", _n3)
sheet.set("nose4_y", _n4)

sheet.set("tail1_y", _t1)
sheet.set("tail2_y", _t2)
sheet.set("tail3_y", _t3)
sheet.set("tail4_y", _t4)




sheet.recompute()
document.recompute()


Gui.Selection.clearSelection()
Gui.activeDocument().activeView().viewFront()
Gui.activeDocument().activeView().saveImage(image_file,1465,1354,'Transparent')

__objs__=[]
__objs__.append(FreeCAD.getDocument("Myring_hull_spreadsheet").getObject("Body"))
import ImportGui
ImportGui.export(__objs__,step_file)
del __objs__
