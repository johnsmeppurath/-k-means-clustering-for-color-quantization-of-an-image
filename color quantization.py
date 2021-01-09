

#No proper elbow was found but after k=7 the graph shows very littile inertia change so i am selecting k=7 as the cluster value

import numpy as np
from skimage import data
from skimage import io
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


#To find the natural quantization of the image graph is drawn on Plot of Inertia vs. K where k is the number of clusters. The function is used from Kmeans.py example and used SKlearn Kmeans method. The graph is run 3 times to make sure that you are not reporting the result of the algorithm getting stuck in a bad configuration
def kMeansRange(data):
    ks = list(range(2,11))
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k)
        c = km.fit(data)
        a = km.cluster_centers_
        inertia = km.inertia_
        for i in range(3):
            km = KMeans(n_clusters=k)
            c=km.fit(data)
            a = km.cluster_centers_
            new_inertia = km.inertia_
            if new_inertia < inertia:
                inertia = new_inertia
        print("k =",k,", inertia =",inertia)
        inertias.append(inertia)
    plt.figure()
    plt.title("Plot of Inertia vs. K")
    plt.xlabel("k")
    plt.ylabel("inertia")
    plt.plot(ks, inertias)
    plt.show()



# Outputing the orginal bird image
bird = io.imread("bird.jpg")
plt.figure()
plt.title("bird.jpg")
plt.imshow(bird)
plt.axis('off')
plt.show()

#making the brid image to a 2d array for calculation
data = bird.reshape(368*529,3)

#drawing the graph to find an elbow
kMeansRange(data)


#USing sklearn calling Kmeans with cluster number 7
km = KMeans(n_clusters=7)
km.fit(data)

#seting the colors for the new image by replacing the colors with the closeset clustor centor colors
newimage = km.cluster_centers_[km.predict(data)]

print("-----------")

#converting the new image back to image format and priting the new image
newimage=np.array(newimage).astype(int)
newimage=newimage.reshape(368, 529, 3)

#Showing the new image
plt.figure()
plt.imshow(newimage)
plt.axis('off')
plt.show()




#Showing the second image
rat = io.imread("rat.jpg")
plt.figure()
plt.title("rat.jpg")
plt.imshow(rat)
plt.axis('off')
plt.show()

#calulating the new color values for the second image using the perviosuly calculated values
data = rat.reshape(368*529,3)
#Assigning the calulated color values to the new image and converting it back to image format arrray
centroids = km.cluster_centers_[km.predict(data)]
centroids=np.array(centroids).astype(int)
centroids=centroids.reshape(368, 529, 3)

#Show the newsecond image with previously calulated color values
plt.figure()
plt.imshow(centroids)
plt.axis('off')
plt.show()




