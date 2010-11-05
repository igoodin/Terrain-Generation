from math import floor
import random

class Noise:
		"""
			Class to generate different types of noise
		"""

        def __init__(self):
                pass
				
        def simplex3d(self,x,y,z):
				"""
					Generates 3d simplex noise
				"""

                #gradients in 3d are the midpoints of a cube
                grad = ((1,1,0),(-1,1,0),(1,-1,0),(-1,-1,0),(1,0,1),(-1,0,1),(1,0,-1),
                        (-1,0,-1),(0,1,1),(0,-1,1),(0,1,-1),(0,-1,-1))
                
                #permutation table from simplex noise demystified
                perm = (151,160,137,91,90,15, 
                131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23, 
                190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33, 
                88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,134,139,48,27,166, 
                77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244, 
                102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,200,196, 
                135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123, 
                5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42, 
                223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9, 
                129,22,39,253,9,98,108,110,79,113,224,232,178,185,112,104,218,246,97,228, 
                251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107, 
                49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254, 
                138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180)
                
                perm *= 2
                
                #skew factor
                s = (x+y+z)/3.0
                
                i=floor(x+s)
                j=floor(y+s)
                k=floor(z+s)
                
                G3 = 1/6.0
                #unskew factor
                t = (i+j+k)*G3
                
                #distances from the cell's orgin
                x0 = x-(i-t)
                y0 = y-(j-t)
                z0 = z-(k-t)
                
                if(x0>=y0):
                        if(y0>=z0):
                                i1,j1,k1,i2,j2,k2=1,0,0,1,1,0
                        elif(x0>=z0):
                                i1,j1,k1,i2,j2,k2=1,0,0,1,0,1
                        else:
                                i1,j1,k1,i2,j2,k2=0,0,1,1,0,1
                else:
                        if(y0<z0):
                                i1,j1,k1,i2,j2,k2=0,0,1,0,1,1
                        elif(x0<z0):
                                i1,j1,k1,i2,j2,k2=0,1,0,0,1,1
                        else:
                                i1,j1,k1,i2,j2,k2=0,1,0,1,1,0
        
                x1 = x0-i1+G3
                y1 = y0-j1+G3
                z1 = z0-k1+G3
                x2 = x0-i2+2.0*G3
                y2 = y0-j2+2.0*G3
                z2 = z0-k2+2.0*G3
                x3 = x0-1.0+3.0*G3
                y3 = y0-1.0+3.0*G3
                z3 = z0-1.0+3.0*G3

                #gets hashed gradient indices of the 4 simplex corners
                ii = int(i)%256
                jj = int(j)%256
                kk = int(k)%256

                gi0 = perm[ii + perm[jj + perm[kk]]] % 12
                gi1 = perm[ii + i1 + perm[jj + j1 + perm[kk + k1]]] % 12
                gi2 = perm[ii + i2 + perm[jj + j2 + perm[kk + k2]]] % 12
                gi3 = perm[ii + 1 + perm[jj + 1 + perm[kk + 1]]] % 12

                t0 = 0.6-x0**2-y0**2-z0**2

                if(t0<0):
                        n0=0.0
                else:
                        g=grad[gi0]
                        n0=t0**4*(g[0]*x0+g[1]*y0+g[2]*z0)
                        
                t1 = 0.6-x1**2-y1**2-z1**2
                if(t1<0):
                        n1=0.0
                else:
                        g=grad[gi1]
                        n1=t1**4*(g[0]*x1+g[1]*y1+g[2]*z1)
                        
                t2 = 0.6-x2**2-y2**2-z2**2
                if(t2<0):
                        n2=0.0
                else:
                        g=grad[gi2]
                        n2=t2**4*(g[0]*x2+g[1]*y2+g[2]*z2)
                        
                t3 = 0.6-x3**2-y3**2-z3**2
                if(t3<0):
                        n3=0.0
                else:
                        g=grad[gi3]
                        n3=t3**4*(g[0]*x3+g[1]*y3+g[2]*z3)
                return 32.0*(n0+n1+n2+n3)