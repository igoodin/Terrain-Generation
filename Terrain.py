from Noise import *
import numpy
import math
import Image, ImageDraw
import matplotlib
import random
import model
from matplotlib import pyplot

class Terrain:
	"""
	Class to generate terrain heightmaps
	"""
	def __init__(self,width=512):
		self.width=width
		self.array = numpy.zeros((width,width),'float32')
		self.river = []

	def simplex(self,octaves=4,freq=0.00388,amp=95.0):
		#makes a heightmap using multiple layers of simplex noise

		self.octaves = octaves
		self.freq = freq
		self.amp = amp

		#load simplex noise object
		N=Noise()

		#get noise for each octave and sum them
		for i in range(self.width):
			for j in range(self.width):
				for t in range(octaves):
					f = freq*(t+1.0)**2
					a= amp/((t+1.0)**3.4)
					self.array[i][j]+= N.simplex3d(i*f,j*f,5.0)*a+30.0

	def diamond_square(self,iterates=1,r=2.0,resume=0):
		#Generates a heightmap using the diamond square recursive method
		t=0
		if(resume!=0):
			t=resume
		width=self.width -1

		#checking that the width is a power of two
		pow=width
		while(pow>=2.0):
			pow/=2.0
		if(int(pow)!=pow):
			raise NameError("Width must be a power of 2 plus one")

		#setting the values of the four corners
		self.array[0][0] = 50.0
		self.array[width][width] = 50.0
		self.array[0][width] = 50.0
		self.array[width][0] = 50.0

		while(t<=iterates):
			t+=1
			f=1
			numfaces = 4**(t-1)

			while(f<=numfaces):
				units = math.sqrt(4.0**(t-1.0))

				#calculating x and y for faces and adjust for boundaries
				if(f==units):
					y = 1
					x = f
				elif(f==units**2):
					y = int(units)
					x = int(units)
				elif(int(f/units)==f/units):
					y = int(f/units)
					x = int(units)
				else:
					y = int(f/units)
					x = f - int(y*units)
					y+=1

				offset =width/units
				x=offset*(x)/(units)*(2**(t-1))
				y=offset*(y)/(units)*(2**(t-1))

				#getting values for the corners
				c1 = self.array[x-width/units][y-width/units]
				c2 = self.array[x][y-width/units]
				c3 = self.array[x-width/units][y]
				c4 = self.array[x][y]

				#setting value for the middlepoint
				Mx = x-width/units*0.5
				My = y-width/units*0.5
				self.array[Mx][My] = (c1+c2+c3+c4)/4.0+random.uniform(-r,r)

				#calculating averages for the midpoints
				m1 = (c1+c2)/2.0+random.uniform(-r,r)
				m2 = (c2+c4)/2.0+random.uniform(-r,r)
				m3 = (c3+c4)/2.0+random.uniform(-r,r)
				m4 = (c1+c3)/2.0+random.uniform(-r,r)

				#setting values of the midpoints
				self.array[x-width/units*0.5][y-width/units] = m1
				self.array[x][y-width/units*0.5] = m2
				self.array[x-width/units*0.5][y] = m3
				self.array[x-width/units][y-width/units*0.5] = m4

				f+=1

	def minima(self,show=False):
		#find minima of the image using a masking technique
		self.minima = self.find_extrema(False)
		
		#Display minima on image by painting dots on the points
		if(show==True):
			img = Image.fromarray(self.array,mode='F')
			draw = ImageDraw.Draw(img)
			for point in self.minima:
					draw.point((point[1],point[0]),fill=255)
			img.show()

	def maxima(self,show=False):
		#find maxima of the image using a masking technique
		self.maxima = self.find_extrema()
	
		#Display maxima on image by painting dots on the points
		if(show==True):
			img = Image.fromarray(self.array,mode='F')
			draw = ImageDraw.Draw(img)
			for point in self.maxima:
					draw.point((point[1],point[0]),fill=2)
			img.show()

	def find_extrema(self,option=True):
		"""
			Finds the extrema of the image using a masking technique. The function
			returns minima when it starts form the bottom and maxima when it starts 
			from the top.
		"""

		extrema = []
		dz=100 #step size
		std = self.array.std()
		max=self.array.max()
		min=self.array.min()

		diff=max-abs(min)
		if(option==False):
			z=min
			step=-float(diff/dz)
		else:
			z=max
			step=float(diff/dz)

		endz=int(dz*.6)
		zper = 1.0

		#Take the image and slices it on x-y plane and loops through loo
		for k in range(dz):
			if(k<=endz):
				for i in range(self.width):
					for j in range(self.width):
						abovez=False
						if(option==False):
							if(self.array[i][j]<=z):
								abovez = True
						else:
							if(self.array[i][j]>=z):
								abovez =True
						if(abovez==True):
							lmax= (i,j,z)
							#checking if point is already in the list of extrema
							in_list = lmax in set(extrema)
							if(in_list==False):
								in_mask =self.check_mask(lmax,extrema,std,zper)
								if(in_mask ==False):
									extrema.append(lmax)

				z-=step
				zper=1.0/dz*100.0
		return extrema

	def check_mask(self,point,list,std,percent):
		radius = std*8.0/percent

		for i in list:
			x=point[0]-i[0]
			y=point[1]-i[1]
			r=math.sqrt(x**2+y**2)
			if(r<=radius):
				return True
		return False

	def contour(self,sections=100):
		contours =[]
		max=self.array.max()
		min=self.array.min()
		height= max-abs(min)
		steps = height/sections
		threshold=steps/10.0
		slice=min

		for z in range(sections):
			loop = []
			for x in range(self.width):
				for y in range(self.width):
					if(self.array[x][y]<slice+threshold):
						if(self.array[x][y]>slice-threshold):
							loop.append([x,y])
			slice+=steps
			contours.append(loop)
		self.contours = contours

	def rivers(self,num_rivers=2,phi=150.0,dr=7.0):
		n=0
		l=False
		
		for minima in self.minima:
			if(n<=num_rivers):
				if(l==False):
					if(minima[0]!=0):
						if(minima[1]!=0):
							r1 = minima
							l=True
							n+=1
				else:
					if(minima[0]!=0):
						if(minima[1]!=0):
							r2 = minima
							n+=1
		
		#make sure the start point is at a lower height than the end point
		if(r1[2]>r2[2]):
			temp = r1
			r1 = r2
			r2 = temp		
		
		path_finished=False

		dx = r2[0]-r1[0]
		dy = r2[1]-r1[1]
		x = r2[0]
		y = r2[1]
		previousx = -100
		previousy = -100
		path = []
		path.append((y,x))

		while(path_finished==False):

			r = math.sqrt(dx**2+dy**2)
			

			#@todo fix and clean this section
			if(abs(dx)>abs(dy)):
				offset = math.degrees(math.atan(dy/(dx+0.000000000000001)))
				
				if(dx>=0.0 and dy>=0.0):
					oo =offset
					offset = offset-90.0

				elif(dx<0.0 and dy>0.0):
					oo =offset
					if(offset<0.0):
						offset = 90.0 - abs(offset)
					else:
						offset = 90.0 - abs(offset)
			
				elif(dx<=0.0 and dy<=0.0):
					oo =offset
					if(offset<0.0):
						offset = 180.0-offset
					else: 
						offset = 90.0 + abs(offset)

				else: 
					oo =offset
					if(offset<=0.0):
						offset = 270.0-abs(offset)
					else:
						offset = offset+90.0

			else:
				offset = math.degrees(math.atan(dx/(dy+0.00000000000000001)))

				if(dx>=0.0 and dy>=0.0):
					oo =offset
					offset = -offset
				elif(dx<=0.0 and dy>=0.0):
					oo =offset
					if(offset<0.0):
						offset = -offset
					else:
						offset = abs(offset-90.0)
			
				elif(dx<=0.0 and dy<=0.0):
					oo =offset
					if(offset<0.0):
						offset = 180-offset
					else:
						offset = 180-offset
	
				else: 
					oo =offset
					if(offset<0.0):
						offset = -(offset+180.0)
					else:
						offset =offset +180.0


			offset = math.radians(offset)
			
			points = []
			dphi=int((dr**1.2)*(phi))

			#find the points on the line thatt we are extrapolating to and put them in an array
			for i in range(dphi+1):
				
				slice = math.radians((phi/dphi)*i-phi/2.0)
				ynew = -dr*(math.cos(slice))
				xnew = -dr*(math.sin(slice)) 

				#hit xpoints an y points with rotational matrix
	
				xp = (xnew*math.cos(offset)-ynew*math.sin(offset))
				yp = (xnew*math.sin(offset)+ynew*math.cos(offset))

				xnew = round(xp)
				ynew = round(yp)
				
				#translate the coordinates to user (x,y) as their orgin
				xnew +=x
				ynew +=y
				
				check = set(points)
				if((xnew,ynew) in check):
					pass
				else:
					checkpath = set(path)
					if((xnew,ynew) in checkpath):
						pass
					else:
						if(self.inbounds(xnew,ynew)==False):
							points.append((xnew,ynew))
							
			#find point with the smallest value
			min=1000000000.0
			img = self.array
			
			
			checkpath = set(path)
			for p in points:
				height = img[math.ceil(p[0])][math.ceil(p[1])]

				if((p[0],p[1]) in checkpath):
					print("already used")
				else:
					if(height<min):
						if(p[0]==previousx and p[1]==previousy):
							pass
						else:
							min = height
							x = math.ceil(p[0])
							y = math.ceil(p[1])
			
			dx = x-r1[0]
			dy = y-r1[1]
			r = math.sqrt(dx**2+dy**2)
			
			#make sure the point isnt the same one as before
			if(previousx==x):
				if(previousy==y):
					path_finished=True
					
			path.append((y,x))
			
			#check if path is oscillating
			
			length = len(path)
			if(length>=3):
				t = path[length-3]
				t3 = path[length-1]
				if(t3 ==t):
					path_finished=True
			
			previousx=x
			previousy=y

			#check if the point is close to the endpoint
			if(r<=1.2*dr):
				path_finished=True
		
		self.river = path
		img = Image.fromarray(self.array,mode='F')
		draw = ImageDraw.Draw(img)
		
		for point in path:
			draw.point((point[0],point[1]),fill=500)
			print(point[0],point[1])
			self.array[point[1]][point[0]]= 0.0
		draw.point((r1[1],r1[0]),fill=2000)
		draw.point((r2[1],r2[0]),fill=2000)
		img.show()
		
	def contour_plot(self,make=False,sections=100):
		if(make==True):
			self.contour(sections)

		img = numpy.zeros((self.width,self.width),'float32')
		step=255.0/sections
		for section in self.contours:
			brightness = 220.0
			for point in section:
				img[point[0],point[1]]=brightness
			brightness+=step
		pyplot.imshow(img)
		self.contour_plot=img

	def hydraulic_erode(self):
			wk=10.0
			ks=.75
			w = numpy.zeros((self.width,self.width),'float32')
			s = numpy.zeros((self.width,self.width),'float32')

			steps =1
			for t in range(steps):

				for i in range(self.width):
					for j in range(self.width):
						w[i][j]+=wk
						self.array[i][j]-=ks*w[i][j]
						s[i][j]+=ks*w[i][j]

				for i in range(self.width):
					for j in range(self.width):
						ht = self.array[i][j]+w[i][j]

						inmatrix = []
						#check if all neighbors are within the matrix indices
						if(self.inbounds(i-1,j-1)==False):
							inmatrix.append((i-1,j-1))
						if(self.inbounds(i-1,j+1)==False):
							inmatrix.append((i-1,j+1))
						if(self.inbounds(i+1,j-1)==False):
							inmatrix.append((i+1,j-1))
						if(self.inbounds(i+1,j+1)==False):
							inmatrix.append((i+1,j+1))

						#get the average for the values of the neighbors
						avg=0.0
						for n in inmatrix:
							avg+=self.array[n[0],n[1]]
						avg/=size(inmatrix)

						di=0.0
						for n in inmatrix:
							hn=self.array[n[0]][n[1]]+w[n[0]][n[1]]
							if(ht>hn):
								dt+=ht-hn

						for n in inmatrix:
							hn=self.array[n[0]][n[1]]+w[n[0]][n[1]]
							if(ht>hn):
								da=ht-avg
								di=ht-hn
								dw=w[i][j]-da*di/dt
								w[n[0]][n[1]]=dw

						w[i][j]+=wk
						self.array[i][j]-=ks*w[i][j]
						s[i][j]+=ks*w[i][j]

	def inbounds(self,i,j):
		i+=1
		j+=1
		within = False
		if(i>self.width):
			within =True
		if(i<1):
			within =True
		if(j>self.width):
			within =True
		if(j<1):
			within =True
		return within

	def show(self,array=None):
		if(array==None):
			img = Image.fromarray(self.array,mode='F')
			img.show()
			pyplot.imshow(self.array)
			pyplot.show()
		else:
			img = Image.fromarray(array,mode='F')
			img.show()
			pyplot.imshow(array)
			pyplot.show()

	def export(self):
		model.export(self.array,self.width)

	def mesh(self):
		surf(self.array)

		if(len(self.river)!=0):
			x = []
			y = []
			z = []
			
			for point in self.river:
				x.append(point[1])
				y.append(point[0])				
				z.append(self.array[point[0]][point[1]])


			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
