
#----------------------------------------------
# Written by: Sadra Avestan
# Date: July 29, 2019
# Description: clustering of protein configurations
#----------------------------------------------
import matplotlib.pyplot as plt

import numpy as np

import math

from mpl_toolkits.mplot3d import Axes3D

from scipy.stats import gaussian_kde

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from operator import itemgetter as ig

from sadra import pubfig as pf

from pylab import rcParams

from copy import deepcopy

import scipy.stats as st

from sklearn.cluster import KMeans

from sadra import pubfig as pf

pf.setup(28, 18)

pf.setup(label_font=28, tick_font=28, axis_width=2, tick_major_width=2, tick_minor_width=1.5, tick_major_size=4, tick_minor_size=4, showminorticks=True)

termini = ["c", "n"]

for ter in termini:

    def kdeplot(X):

        #----- Plot the results -----

        wcss = [] # #within cluster sum of squares

        for i in range(1, 11):

            kmeans = KMeans(n_clusters=i, random_state=5) # init='k-means++', max_iter=300, n_init=10, random_state=0)

            kmeans.fit(X)

            wcss.append(kmeans.inertia_)

                      #Plotting the results onto a line graph, allowing us to observe 'The elbow'

        plt.plot(range(1,11), wcss)

        plt.scatter(range(1,11), wcss, c='r')

        # plt.title('The elbow method')

        plt.xlabel('Numbe of clusters')

        plt.ylabel('wcss')

        plt.xticks([i for i in range(11)])

        # plt.savefig('scaled-Elbowmethod_on_nter-theta_phi-cv.png', bbox_inches='tight', dpi=200)

        # plt.show()

        img_name = "elbow-%s-terminal-qn-theta-scaled.png"%ter

        plt.savefig(img_name, format='png', dpi=300, bbox_inches='tight', transparent=False)

        plt.close()

                # ##Number of clusters

        n_cl = 4 if ter=='n' else 3

        kmeans = KMeans(n_clusters=n_cl, random_state=5)

        # Fitting the input data

        kmeans = kmeans.fit(X)

        # Getting the cluster labels

        labels = kmeans.predict(X)

        # Centroid values

        centroids = kmeans.cluster_centers_

        # Comparing with scikit-learn centroids

        print("Centroid values")

        print("Scratch")

        #print(C) # From Scratch

        print("sklearn")

        print(centroids) # From sci-kit learn

        ##plot the cluster centers

        with open('kmeans-cter-nc%d.csv'%n_cl, 'w+') as file:

            file.write('%s'%centroids)

        plt.scatter(X[:,0], X[:,1], c=labels, cmap='rainbow' )

        ##plot the cluster centers

        plt.scatter(centroids[:,0], centroids[:,1], c='black', s=300, alpha=0.8, marker='*', label='centroid' )

        # plt.ylim(0, 360)

        plt.xlim(0,1)

        plt.ylabel(r'$Q_N (^{\circ}$)')

        plt.xlabel(r'$\theta (^{\circ})$') # ^{*90}

        # plt.tick_params(top='off', bottom='off', labelleft='on', labelbottom='off')

        plt.xticks([0, 0.28, 0.56, 0.84, 1], [0, 25, 50, 75, ' ']) # mix ticks and xticklabels

        plt.legend(loc=2)

        from collections import Counter, defaultdict

        print(Counter(kmeans.labels_))

        plt.savefig('kmeans-cv-qn-theta-%ster-nc%d.png'%(ter,n_cl), dpi=400, bbox_inches='tight')

        # plt.show()

        plt.close()

    ################################################################################################

    # theta_name = "theta-unified_with_cter_theta-rows.dat"

    qn_name = "208trajs-qn-unified_with_%ster_theta-rows_450head.dat"%ter

    theta_name = "208trajs-theta-unified_with_%ster_theta-rows_450head.dat"%ter

    qn_file = np.genfromtxt(qn_name)

    theta_file = np.genfromtxt(theta_name)

    qn_data = qn_file[:,1]

    theta_data = theta_file[:,1]

    y = qn_data

    x = theta_data

    x = x/90

    X = np.array(list(zip(x,y)))

    # print(theta_file.shape)

    kdeplot(X)
