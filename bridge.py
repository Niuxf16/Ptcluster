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

###Bridge sites###
Bridgesites=[]
for edge in edges_neighbor.keys():
    edge_neighbor=edges_neighbor[edge]
    edge=edge[1:-1]
    edge=edge.split(',')
    edge=[int(i) for i in edge]
    point1 = np.array([x[edge[0]],y[edge[0]],z[edge[0]]])
    point2 = np.array([x[edge[1]],y[edge[1]],z[edge[1]]])
    point_mean_edge=(point1+point2)/2
    triangel_neighbor1 = edge_neighbor[0:3]
    triangel_neighbor2 = edge_neighbor[3:]
    triangle1_p1 = np.array([x[triangel_neighbor1[0]], y[triangel_neighbor1[0]], z[triangel_neighbor1[0]]])
    triangle1_p2 = np.array([x[triangel_neighbor1[1]], y[triangel_neighbor1[1]], z[triangel_neighbor1[1]]])
    triangle1_p3 = np.array([x[triangel_neighbor1[2]], y[triangel_neighbor1[2]], z[triangel_neighbor1[2]]])
    triangle2_p1 = np.array([x[triangel_neighbor2[0]], y[triangel_neighbor2[0]], z[triangel_neighbor2[0]]])
    triangle2_p2 = np.array([x[triangel_neighbor2[1]], y[triangel_neighbor2[1]], z[triangel_neighbor2[1]]])
    triangle2_p3 = np.array([x[triangel_neighbor2[2]], y[triangel_neighbor2[2]], z[triangel_neighbor2[2]]])

    point_tri1_mean=(triangle1_p1+triangle1_p2+triangle1_p3)/3
    point_tri2_mean = (triangle2_p1 + triangle2_p2 + triangle2_p3) / 3
    n1=Normalvector_by_3points(triangle1_p1,triangle1_p2,triangle1_p3)
    n2 = Normalvector_by_3points(triangle2_p1, triangle2_p2, triangle2_p3)

    site1=point_tri1_mean+1.5*n1
    site2=point_tri1_mean-1.5*n1
    d1=Euclidean_distance(P_center,site1)
    d2=Euclidean_distance(P_center,site2)
    if d1>d2:
        n1=n1
    else:
        n1=-n1

    site3=point_tri2_mean+1.5*n2
    site4=point_tri2_mean-1.5*n2
    d1=Euclidean_distance(P_center,site3)
    d2=Euclidean_distance(P_center,site4)
    if d1>d2:
        n2=n2
    else:
        n2=-n2
    n=(n1+n2)/np.linalg.norm(n1+n2)
    bridge_site=point_mean_edge+1.5*n
    Bridgesites.append(bridge_site)

#Write file
p=read('pt.xyz')
for i in range(len(Bridgesites)):
    p.append(Atom('H',position=Bridgesites[i]))

write('Pt_H_ads_bri_x.xyz',p)