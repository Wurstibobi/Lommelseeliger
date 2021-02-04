import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from mpl_toolkits import mplot3d

from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull


#Define the shape of the ellipsoid

Coeffs = (1, 1, 1)
rx, ry, rz = 1/np.sqrt(Coeffs)
point = []
theta = []
phi= []




#We create the points of our ellipsoids surface
num_pts = 1000
indices = np.arange(0, num_pts, dtype=float) + 0.5

phi = np.arccos(1 - 2*indices/num_pts)

theta = np.pi * (1 + 5**0.5) * indices

#Mesh_vectors creates a matrix that contains the vertices of the ellipsoids surface triangles
def mesh_vectors(V,F):
   """
   Creates a vector set of the mesh data for nice plotting.
   """
   msh = np.zeros((np.shape(F)[0],3,3))
   for i, face in enumerate(F):
      for j in range(3):
         msh[i][j] = V[face[j],:]
   return msh


#We change the spherical coordinates to cartesian coordinates for later use.
x, y, z = rx*np.cos(theta) * ry*np.sin(phi), np.sin(theta) * rz*np.sin(phi), np.cos(phi);
points = np.column_stack((x,y,z))
hull = ConvexHull(points)

triangles = hull.simplices



#We create the Lommel function that uses Lommel-seeliger method to calculate the brithness of each triangle.
def Lommel(Kolmiot, norms,a):

    I = (1,0,0)


    l = 0
    for i in range (0,len(Kolmiot)):
        e = [np.cos(a), np.sin(a),0]
        evalo = np.dot(I,norms[i])
        ehav = np.dot(e,norms[i])

        if (evalo > 0 ) and (ehav > 0):
            norm = norms[i]
            kolmio = Kolmiot[i]
            Ala = np.sqrt(norm[0]**2+norm[1]**2+norm[2]**2)
            n = norm/np.sqrt(norm[0]**2+norm[1]**2+norm[2]**2)
            u = np.dot(I,n)

            eh = e/np.sqrt(e[0]**2+e[1]**2)
            u0 = np.dot(eh,n)

            l = (l+(Ala*u*u0/(u+u0)/np.pi))
    return l


#Plot_mesh function plots our ellipsoid and the integrated brighness.
def plot_mesh(points, tris):
   fig = plt.figure(figsize=(8,8))

   #Comment this line if you don't want to plot the ellipsoid
   ax = mplot3d.Axes3D(fig)

   norms = []
   meshvectors = mesh_vectors(points, tris)
   centers = []
   for meshvector in meshvectors:


       vec1 = meshvector[1]-meshvector[0]
       vec2 = meshvector[2]-meshvector[0]

       cent = 1/3*(meshvector[0]+meshvector[1]+meshvector[2])
       rkesk = np.sqrt(cent[0]**2+cent[1]**2+cent[2]**2)
       norm = np.cross(vec1,vec2)
       d = cent + norm
       rd = np.sqrt(d[0]**2+d[1]**2+d[2]**2)
       centers.append(cent)
       if((rd < rkesk)):
           norm = np.cross(vec2,vec1)

       norms.append(norm)

        #Comment this line if you don't want to plot the ellipsoid
       ax.quiver(cent[0],cent[1],cent[2],norm[0],norm[1],norm[2],length=0.4,normalize=True)

   #Comment this line if you don't want to plot the ellipsoid
   ax = mplot3d.Axes3D(fig)


   Brightness2 = []

   #Comment this line if you don't want to plot the ellipsoid
   ax.add_collection3d(mplot3d.art3d.Poly3DCollection(meshvectors, facecolor=[0.5,0.5,0.5], lw=0.5, edgecolor=[0,0,0], alpha=1, antialiaseds=True))
   scale = points.flatten('F')
   ax.auto_scale_xyz(scale, scale, scale)

   for x in range (0,180,1):
       a = x*np.pi/180
       Analytical =(1-np.sin(a/2)*np.tan(a/2)*np.log(np.tan(a/4)**-1))
       Phase_angle.append(x)
       Brightness.append(Lommel(meshvectors,norms,a))
       Brightness2.append(Analytical)


   #Remove these comments if you want to plot the Numerican and analytical brightness.
   #plt.plot(Phase_angle,Brightness, label = "Numerical")
   #plt.plot(Phase_angle,Brightness2,label = "Analytical")
   #plt.title("Integroitu Brightness - Fibonacci")
   plt.legend()
   plt.show()
   return
Brightness = []
Phase_angle = []
plot_mesh(points,triangles)
