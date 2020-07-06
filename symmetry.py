from ase import Atoms,Atom
from ase.io import read,write
from scipy.spatial import Delaunay
import numpy as np
from scipy import io
from collections import Counter
import math

###Data input###
p=read('pt.xyz')
Ppt=p.positions
x=[i[0] for i in Ppt]
y=[i[1] for i in Ppt]
z=[i[2] for i in Ppt]

###Central positions of clusters###
X_center=0
Y_center=0
Z_center=0
P_number=len(Ppt)
for pos in Ppt:
    X_center=pos[0]+X_center
    Y_center=pos[1]+Y_center
    Z_center=pos[2]+Z_center
P_center=[X_center/P_number,Y_center/P_number,Z_center/P_number]
print('Central point coordinate:',P_center)

###Euclidean distance between two points###
def Euclidean_distance(point1,point2):
    point1=np.asarray(point1)
    point2=np.asarray(point2)
    AB=np.asmatrix(point2-point1)
    distance=np.linalg.norm(AB)
    return distance

# Smooth a list.
def smooth(d):
    sd=[]
    for i in d:
        for j in sd:
            if abs(i-j)<0.1:
                i=j
                break
        sd.append(i)
    return sd

# Find point's positions in points list.
def find_in_points(points,point,atol=0.1):
    if len(points)==0:
        return []
    diff=np.array(points)-np.array(point)[None,:]
    return np.where(np.all(np.abs(diff)<atol,axis=1))[0]

# Judge point is or not in points_list.
def is_in_points(points,point):
    return len(find_in_points(points,point))>0

# Return the positions (x',y') when (x1,y1) revolve thtea(0<thrat<360) around (x2,y2)
def revolved(p1,p2,theta):
    x1=p1[0]
    y1=p1[1]
    x2=p2[0]
    y2=p2[1]
    x=(x1-x2)*math.cos(theta)-(y1-y2)*math.sin(theta)+x2
    y=(x1-x2)*math.sin(theta)+(y1-y2)*math.cos(theta)+y2
    return [x,y]

# Judge whether two points are same or not.
def Issame(p1,p2,threshold=0.5):
    detap=[(abs(i)-threshold>0) for i in np.array(p1)-np.array(p2)]
    zeros=[0 for i in p1]
    return detap==zeros

# Judge the x direction is plane symmertry or not.
Xmean=np.mean(x)
atoms_x_top=[]
atoms_x_bottom=[]
Flag_sym_x=1
for pos in Ppt:
    if pos[0] - Xmean > 0.1:
        atoms_x_top.append(pos)
    if pos[0] - Xmean < -0.1:
        atoms_x_bottom.append(pos)
for pos_t in atoms_x_top:
    pos_tr=[2*Xmean-pos_t[0],pos_t[1],pos_t[2]]
    flag=1
    for index,pos_b in enumerate(atoms_x_bottom):
        if Issame(pos_tr,pos_b):
            atoms_x_bottom.pop(index)
            flag=0
            break
    if flag:
        Flag_sym_x = 0
        break
print('X plane symmetry:',Flag_sym_x==1)

# Judge the y direction is symmertry or not.
Ymean=np.mean(y)
atoms_y_top=[]
atoms_y_bottom=[]
Flag_sym_y=1
for pos in Ppt:
    if pos[0] - Ymean > 0.1:
        atoms_y_top.append(pos)
    if pos[0] - Ymean < -0.1:
        atoms_y_bottom.append(pos)
for pos_t in atoms_y_top:
    pos_tr=[2*Ymean-pos_t[0],pos_t[1],pos_t[2]]
    flag=1
    for index,pos_b in enumerate(atoms_y_bottom):
        if Issame(pos_tr,pos_b):
            atoms_y_bottom.pop(index)
            flag=0
            break
    if flag:
        Flag_sym_y = 0
        break
print('Y plane symmetry:',Flag_sym_y==1)

# Judge the z direction is symmetry or not.
Zmean=np.mean(z)
atoms_z_top=[]
atoms_z_bottom=[]
Flag_sym_z=1
for pos in Ppt:
    if pos[0] - Zmean > 0.1:
        atoms_z_top.append(pos)
    if pos[0] - Zmean < -0.1:
        atoms_z_bottom.append(pos)
for pos_t in atoms_z_top:
    pos_tr=[2*Zmean-pos_t[0],pos_t[1],pos_t[2]]
    flag=1
    for index,pos_b in enumerate(atoms_z_bottom):
        if Issame(pos_tr,pos_b):
            atoms_z_bottom.pop(index)
            flag=0
            break
    if flag:
        Flag_sym_z = 0
        break
print('Z plane symmetry:',Flag_sym_z==1)

###Judge the x direction is rational symmetry or not.
#Judge the layers in x direaction.
xlayer=[]
for i in x:
    flag = 1
    for j in xlayer:
        if abs(i-j)<0.1:
            flag=0
            break
    if flag==1:
        xlayer.append(i)

#Divide the atoms into different x layers.
atoms_layer_x=[]
for x in xlayer:
    single_layer_x=[]
    for pos in Ppt:
        if abs(pos[0]-x)<0.1:
            single_layer_x.append(pos)
    atoms_layer_x.append(single_layer_x)
layers_number_x=len(atoms_layer_x)

#Judge each layer is rational symmertry or not.
layers_symmetry_flag_x=np.zeros(layers_number_x) #Store the revolve symmetry information for each layers
layers_symmetry_dimension_x=np.zeros(layers_number_x) #Store the dimension of symmetry axis for each layers
for i in range(layers_number_x):
    distance=[]
    for pos in atoms_layer_x[i]:
        x_axis=[xlayer[i],Ymean,Zmean]
        d = Euclidean_distance(x_axis, pos)
        if d>0.1:
            distance.append(d)
    distance=smooth(distance)
    D=Counter(distance)
    n=min(D.values())
    theta=math.pi/n*2
    pos_yz=[p[1:3] for p in atoms_layer_x[i]]
    flag=1
    for po in pos_yz:
        pos_re=revolved(po,[Ymean,Zmean],theta)
        if (not is_in_points(pos_yz,pos_re)):
            flag=0
    layers_symmetry_flag_x[i]=flag
    layers_symmetry_dimension_x[i]=n
if not is_in_points(layers_symmetry_flag_x,[0]):
    n_sym_x=min(layers_symmetry_dimension_x)
    flag=1
    for n in layers_symmetry_dimension_x:
        if n%n_sym_x==0:
            flag=0
    print('X revolving symmetry:',int(n_sym_x),'dimension')
else:
    print('X revolving symmetry: None')

###Judge the y direction is rational symmertry or not.
#Judge the layers in x direaction.
ylayer=[]
for i in y:
    flag = 1
    for j in ylayer:
        if abs(i-j)<0.1:
            flag=0
            break
    if flag==1:
        ylayer.append(i)

#Divide the atoms into different y layers.
atoms_layer_y=[]
for x in ylayer:
    single_layer_y=[]
    for pos in Ppt:
        if abs(pos[1]-x)<0.1:
            single_layer_y.append(pos)
    atoms_layer_y.append(single_layer_y)
layers_number_y=len(atoms_layer_y)

#Judge each layer is rational symmertry or not.
layers_symmetry_flag_y=np.zeros(layers_number_y) #Store the revolve symmetry information for each layers
layers_symmetry_dimension_y=np.zeros(layers_number_y) #Store the dimension of symmetry axis for each layers
for i in range(layers_number_y):
    distance=[]
    for pos in atoms_layer_y[i]:
        y_axis=[Xmean,ylayer[i],Zmean]
        d = Euclidean_distance(y_axis, pos)
        if d>0.1:
            distance.append(d)
    distance=smooth(distance)
    D=Counter(distance)
    n=min(D.values())
    theta=math.pi/n*2
    pos_xz=[[p[0],p[2]] for p in atoms_layer_y[i]]
    flag=1
    for po in pos_xz:
        pos_re=revolved(po,[Xmean,Zmean],theta)
        if (not is_in_points(pos_xz,pos_re)):
            flag=0
    layers_symmetry_flag_y[i]=flag
    layers_symmetry_dimension_y[i]=n
if not is_in_points(layers_symmetry_flag_y,[0]):
    n_sym_y=min(layers_symmetry_dimension_y)
    flag=1
    for n in layers_symmetry_dimension_y:
        if n%n_sym_y==0:
            flag=0
    print('Y revolving symmetry:',int(n_sym_y),'dimension')
else:
    print('Y revolving symmetry: None')


###Judge the z direction is rational symmertry or not.
#Judge the layers in z direaction.
zlayer=[]
for i in z:
    flag = 1
    for j in zlayer:
        if abs(i-j)<0.1:
            flag=0
            break
    if flag==1:
        zlayer.append(i)

#Divide the atoms into different y layers.
atoms_layer_z=[]
for x in zlayer:
    single_layer_z=[]
    for pos in Ppt:
        if abs(pos[2]-x)<0.1:
            single_layer_z.append(pos)
    atoms_layer_z.append(single_layer_z)
layers_number_z=len(atoms_layer_z)

#Judge each layer is rational symmertry or not.
layers_symmetry_flag_z=np.zeros(layers_number_z) #Store the revolve symmetry information for each layers
layers_symmetry_dimension_z=np.zeros(layers_number_z) #Store the dimension of symmetry axis for each layers
for i in range(layers_number_z):
    distance=[]
    for pos in atoms_layer_z[i]:
        z_axis=[Xmean,Ymean,zlayer[i]]
        d = Euclidean_distance(z_axis, pos)
        if d>0.1:
            distance.append(d)
    distance=smooth(distance)
    D=Counter(distance)
    n=min(D.values())
    theta=math.pi/n*2
    pos_xy=[[p[0],p[1]] for p in atoms_layer_z[i]]
    flag=1
    for po in pos_xy:
        pos_re=revolved(po,[Xmean,Ymean],theta)
        if (not is_in_points(pos_xy,pos_re)):
            flag=0
    layers_symmetry_flag_z[i]=flag
    layers_symmetry_dimension_z[i]=n

if not is_in_points(layers_symmetry_flag_z,[0]):
    n_sym_z=min(layers_symmetry_dimension_z)
    flag=1
    for n in layers_symmetry_dimension_z:
        if n%n_sym_z==0:
            flag=0
    print('Z revolving symmetry:',int(n_sym_z),'dimension')
else:
    print('Z revolving symmetry: None')
