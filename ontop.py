from ase import Atoms,Atom
from ase.io import read,write
from scipy.spatial import Delaunay
import numpy as np
from scipy import io
from collections import Counter
import math
import heapq

###Data input###
p=read('pt.xyz')
Ppt=p.positions

x=[i[0] for i in Ppt]
y=[i[1] for i in Ppt]
z=[i[2] for i in Ppt]

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
edges_neighbor=dict()
for triangle in surfaces_bud:
    triangle=np.sort(triangle)
    triangle=[triangle[0], triangle[1], triangle[2]]
    l1=str([triangle[0],triangle[1]])
    l2=str([triangle[0],triangle[2]])
    l3=str([triangle[1],triangle[2]])
    edges[l1] = edges.setdefault(l1,0) + 1
    edges[l2] = edges.setdefault(l2,0) + 1
    edges[l3] = edges.setdefault(l3,0) + 1
    edges_neighbor[l1]=edges_neighbor.get(l1,[])+triangle
    edges_neighbor[l2] = edges_neighbor.get(l2,[])+triangle
    edges_neighbor[l3] = edges_neighbor.get(l3,[])+triangle

edges_bud=[]
for key,value in edges.items():
    if value==2:
        key=key[1:-1] #delete '[' and ']'
        key=key.split(',') #delete ','
        key=[int(i) for i in key]
        edges_bud.append(key)
print('boundary edges number:',len(edges))

###Choose points in bundary###
points = dict()
points_neighbor=dict()
for i in edges_bud:
    line = np.sort(i)
    p1 = str([line[0]])
    p2 = str([line[1]])
    points[p1] = points.setdefault(p1, 0) + 1
    points[p2] = points.setdefault(p2, 0) + 1
    if p2 in points_neighbor.get(p1, ''):
        pass
    else:
        points_neighbor[p1] = points_neighbor.get(p1, '') + p2

    if p1 in points_neighbor.get(p2, ''):
        pass
    else:
        points_neighbor[p2] = points_neighbor.get(p2, '') + p1
print(points_neighbor)

points_bud = []
for key, value in points.items():
    key = key[1:-1]  # delete '[' and ']'
    key = key.split(',')  # delete ','
    key = [int(i) for i in key]
    points_bud.append(key)
print('boundary points number:', len(points_bud))

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

###Find point's positions in points list.
def find_in_points(points,point,atol=0.1):
    if len(points)==0:
        return []
    diff=np.array(points)-np.array(point)[None,:]
    return np.where(np.all(np.abs(diff)<atol,axis=1))[0]

###Judge point is or not in points_list.
def is_in_points(points,point):
    return len(find_in_points(points,point))>0

###Return the positions (x',y') when (x1,y1) revolve thtea(0<thrat<360) around (x2,y2)
def revolved(p1,p2,theta):
    x1=p1[0]
    y1=p1[1]
    x2=p2[0]
    y2=p2[1]
    x=(x1-x2)*math.cos(theta)-(y1-y2)*math.sin(theta)+x2
    y=(x1-x2)*math.sin(theta)+(y1-y2)*math.cos(theta)+y2
    return [x,y]

###Ontop sites###
Ontopsites=[]
for point in points_neighbor.keys():
    point_neis=points_neighbor[point]
    point=point[1:-1]
    point=int(point)
    x1=x[point]
    y1=y[point]
    z1=z[point]

    point_neis=point_neis.replace('[','')
    point_neis=point_neis.split(']')
    point_neis.pop()
    point_neis=[int(i) for i in point_neis]
    distance=[]
    #Get the distance for neighborhood
    for i in point_neis:
        x1_neb=x[i]
        y1_neb=y[i]
        z1_neb=z[i]
        d=Euclidean_distance([x1,y1,z1],[x1_neb,y1_neb,z1_neb])
        distance.append(d)

    #Find three point nearest the point
    min_number=heapq.nsmallest(3,distance)
    min_index=[]
    for t in min_number:
        index=distance.index(t)
        min_index.append(index)
        distance[index]=float('inf')
    p1_neb = [x[point_neis[min_index[0]]],y[point_neis[min_index[0]]],z[point_neis[min_index[0]]]]
    p2_neb = [x[point_neis[min_index[1]]], y[point_neis[min_index[1]]], z[point_neis[min_index[1]]]]
    p3_neb = [x[point_neis[min_index[2]]], y[point_neis[min_index[2]]], z[point_neis[min_index[2]]]]
    n=Normalvector_by_3points(p1_neb,p2_neb,p3_neb)

    #Judge diraction of nomal vector and get ontop site.
    site1=np.array([x1,y1,z1])+1.5*n
    site2=np.array([x1,y1,z1])-1.5*n
    d1=Euclidean_distance(P_center,site1)
    d2=Euclidean_distance(P_center,site2)
    if d1>d2:
        ontopsite=site1
    else:
        ontopsite=site2
    Ontopsites.append(ontopsite)

p=read('pt.xyz')
for i in range(len(Ontopsites)):
    p.append(Atom('H',position=Ontopsites[i]))
write('Pt_H_ads_top_x.xyz',p)