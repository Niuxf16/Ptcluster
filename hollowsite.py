
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
#Delaunay 
tri=Delaunay(np.array([x,y,z]).T)


###Choose surfaces in bundary###
surfaces=dict()
surfaces_num=len(tri.simplices)
for i in range(surfaces_num):
    tetrahedron=np.sort(tri.simplices[i])
    s1=str([tetrahedron[0],tetrahedron[1],tetrahedron[2]])
    s2=str([tetrahedron[0],tetrahedron[1],tetrahedron[3]])
    s3=str([tetrahedron[0],tetrahedron[2],tetrahedron[3]])
    s4=str([tetrahedron[1],tetrahedron[2],tetrahedron[3]])
    surfaces[s1]=surfaces.setdefault(s1,0)+1
    surfaces[s2]=surfaces.setdefault(s2,0)+1
    surfaces[s3]=surfaces.setdefault(s3,0)+1
    surfaces[s4]=surfaces.setdefault(s4,0)+1
print('totol surfaces number:',len(surfaces))

surfaces_bud=[]
for key,value in surfaces.items():
    if value==1:
        key=key[1:-1] #delete '[' and ']'
        key=key.split(',') #delete ','
        key=[int(i) for i in key]
        surfaces_bud.append(key)
print('boundary surfaces number:',len(surfaces_bud))

###Choose edges in bundary###
edges=dict()
for triangle in surfaces_bud:
    triangle=np.sort(triangle)
    l1=str([triangle[0],triangle[1]])
    l2=str([triangle[0],triangle[2]]) 
    l3=str([triangle[1],triangle[2]])
    edges[l1] = edges.setdefault(l1,0) + 1
    edges[l2] = edges.setdefault(l2,0) + 1
    edges[l3] = edges.setdefault(l3,0) + 1

edges_bud=[]
for key,value in edges.items():
    if value==2:
        key=key[1:-1] #delete '[' and ']'
        key=key.split(',') #delete ','
        key=[int(i) for i in key]
        edges_bud.append(key)
print('boundary edges number:',len(edges_bud))

###Choose points in bundary###
points=dict()
for i in edges_bud:
    line=np.sort(i)
    p1=str([line[0]])
    p2=str([line[1]])
    points[p1] = points.setdefault(p1,0) + 1
    points[p2] = points.setdefault(p2,0) + 1

points_bud=[]
for key,value in points.items():
    
    key=key[1:-1] #delete '[' and ']'
    key=key.split(',') #delete ','
    key=[int(i) for i in key]
    points_bud.append(key)
print('boundary points number:',len(points_bud))


###Central positions of clusters###
X_center=0
Y_center=0
Z_center=0
P_number=len(points_bud)
for i in points_bud:
    X_center=x[i[0]]+X_center
    Y_center=y[i[0]]+Y_center
    Z_center=z[i[0]]+Z_center
P_center=[X_center/P_number,Y_center/P_number,Z_center/P_number]


###Euclidean distance between two points###
def Euclidean_distance(point1,point2):
    point1=np.asarray(point1)
    point2=np.asarray(point2)
    AB=np.asmatrix(point2-point1)
    distance=np.linalg.norm(AB)
    return distance

###Normal vector for triangle plane###
def Normalvector_by_3points(point1,point2,point3):
    point1=np.asarray(point1)
    point2=np.asarray(point2)
    point3=np.asarray(point3)

    AB=np.asmatrix(point2-point1)
    AC=np.asmatrix(point3-point1)
    N=np.cross(AB,AC)
    L=np.linalg.norm(N)
    return N[0]/L

###Hollow sites###
Hollowsites=[]
for triangle in surfaces_bud:
    point1=np.array([x[triangle[0]],y[triangle[0]],z[triangle[0]]])
    point2=np.array([x[triangle[1]],y[triangle[1]],z[triangle[1]]])
    point3=np.array([x[triangle[2]],y[triangle[2]],z[triangle[2]]])
    point_mean=(point1+point2+point3)/3
    n=Normalvector_by_3points(point1,point2,point3)
    site1=point_mean+1.5*n
    site2=point_mean-1.5*n
    d1=Euclidean_distance(P_center,site1)
    d2=Euclidean_distance(P_center,site2)
    if d1>d2:
        Hollowsites.append(site1)
    else:
        Hollowsites.append(site2)

#Write file
p=read('pt.xyz')
for i in range(len(Hollowsites)):
    p.append(Atom('H',position=Hollowsites[i]))
write('Pt_H_ads_hollow.xyz',p)


