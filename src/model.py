def export(d,w):
		"""
			Exports a grid array as an wavefront obj file
		"""
        file = open('terrain.obj','w')
        rows = w-1
        faces = rows**2
        points = w**2
        for u in range(w):
                for v in range(w):
                        file.write(str('v ')+str(u)+' '+str(v)+' '+str(d[u][v])+'\n')
        file.write('\n')

        for x in range(rows):
                for y in range(rows):
                        p1=x*w+y+1
                        p2=x*w+y+2
                        p3=(x+1)*w+y+1
                        p4=(x+1)*w+y+2
                        file.write(str('f ')+str(p1)+' '+str(p2)+' '+str(p3)+'\n')
			file.write(str('f ')+str(p2)+' '+str(p3)+' '+str(p4)+'\n')
        file.close()
        return
