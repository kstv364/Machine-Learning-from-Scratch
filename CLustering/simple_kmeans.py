
import random 
points = [(1,1),  (100, 100), (1,0), (98,100), (1,2), (100, 98), (2,1), (2,2), (98,98), (0,1), (-200, 98), (-198,100),   (-200, 100),  (198, 98)]

class cluster:
    def __init__(self,point,id):
        self.id = id
        self.center = point
        self.points = set()
        #self.points.add(point)

    def add(self,point):
        self.points.add(tuple(point))

    def calculate_center(self):
        old_center = self.center
        cx = 0
        cy = 0
        for (x,y) in list(self.points):
            cx += x
            cy += y
        cx = cx/len(self.points); cy = cy/len(self.points)
        cx = round(cx,2)
        cy = round(cy,2)
        self.center = (cx,cy)

        if old_center==self.center:
            return False
        return True

    def reinit_clusters(self):
        self.points = set()

    def __str__(self):
       
        return "id : {} \n Center : {} \n Points {}".format(self.id, self.center,self.points)


def distance(c1,c2):
    x1,y1 = c1
    x2, y2 = c2
    return ((x1-x2)**2 + (y1-y2)**2)**(1/2)


def kMeans(k,points,n=None):
    clusters = []
    choices = random.sample(points,k)
    for i in range(k):
        clusters.append(cluster(choices[i],i+1))
        #print(clusters[i])
    j = 0
    while True:
        print("================== Iteration -",j+1,"===================")
        j += 1
        for c in clusters:
            c.reinit_clusters()

        for point in points:
            distances = [None]*k
            for i,c in enumerate(clusters):
                distances[i] = distance(c.center,point)
            
            idx = distances.index(min(distances))
            clusters[idx].add(point)

        changed = [False]*k

        print("End of Iteration : ",i+1)
        for i,c in enumerate(clusters):
            changed[i] = c.calculate_center()
            print(c)
        if True not in changed:
            break
        if n and j==n:
            break

    return clusters


clusters = kMeans(3,points)
print()
print("================= End of algorithm ==================")
for c in clusters:
    print(c)
