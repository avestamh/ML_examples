#----------------------------
## Written by: Sadra Avestan
## Date: Dec 2019
#----------------------------

# Hierarchical (agglomerative) clustering 
# As an example of an unsupervised learning problem: 
# classify it in a number of clusters derived from the dendogram
# compare with K-means clustering results
import numpy as np
import pandas as pd
from numpy import savetxt
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sadra import pubfig as pf
pf.setup(label_font=24, tick_font=22, axis_width=2, tick_major_width=2,
         tick_minor_width=1.5, tick_major_size=4, tick_minor_size=4, showminorticks=True)
# plt.figure(figsize=(10,4))
qn_path = '/address_for_the_file'
# pf.setup(18,16)
pf.setup(label_font=28, tick_font=20, axis_width=2, tick_major_width=2,
         tick_minor_width=2, tick_major_size=4, tick_minor_size=3, showminorticks=True)
# from sadra import test
# test.bark()
path_names = ['a', 'b']
prot_names = ['ngfp', 'gfpc']
# beta_names = {'a':'b11', 'b':'b1'}
qn_columns = {'ngfp': 3, 'gfpc': 4}
angle_path = 'address/inertia-azimuth'
qn_path = 'address'
patha = [1, 2, 4, 5, 6, 8, 10, 12, 14, 15, 16, 18, 20, 23, 26, 28, 29, 31, 33, 34, 36, 38, 39, 40, 43, 44, 45, 46, 48, 49, 51, 55, 56, 57, 62, 63, 64, 66]
pathb = [3, 7, 9, 17, 19, 21, 22, 25, 27, 30, 32, 37, 41, 42, 47, 50, 52, 53, 54, 58, 59, 61, 65]
def get_qn_theta_data(trajs):
    qn_ngfp = []
    qn_gfpc = []
    theta_ngfp = []
    theta_gfpc = []
    for traj in trajs:
        # for traj in range(1,2):
        for prot in prot_names:
            qn_name = qn_path + \
                "/qn-rmsd-2gfp-26s-repetitive-l51-f100-traj%d/qn-rept-2gfp.dat.merged" % traj
            theta_name = angle_path + \
                '/%s-rept-inertia-T300-l51-f100/%s-repet-first_inertia-f100-traj%d/%s-rep-f100-mom_inertia.dat.merged' % (
                    prot, prot, traj, prot)
            theta_file = np.genfromtxt(theta_name, comments='#')
            qn_file = np.genfromtxt(qn_name, comments='#')
            ####___make new file to save every 10 frame____
            theta_row = np.shape(theta_file)[0]  # number of rows
            theta_col = np.shape(theta_file)[1]  # number of columns
            ####CHECK JUST 4 ns BEFORE UNFOLDING OF THE FIRST CHAIN, TO SEE THE DIFFERENCES IN THE CLUSTERIN
            qn_file_gfp1 = np.genfromtxt(
                qn_path+"/qn-rmsd-2gfp-26s-repetitive-l51-f100-traj%d/qn-gfp1-rept.dat.merged" % traj, comments="#")  # ngfp
            qn_file_gfp2 = np.genfromtxt(
                qn_path+"/qn-rmsd-2gfp-26s-repetitive-l51-f100-traj%d/qn-gfp2-rept.dat.merged" % traj, comments="#")
            no_rows = np.where(qn_file_gfp1[:, 14] < .2)[
                0][0] if trajs == patha else np.where(qn_file_gfp2[:, 3] < 0.2)[0][0]
            # no_rows = np.where(qn_file[:,qn_columns[prot]] < .92)[0][0]
            print('no_row-prot-%s-traj%d: %s' % (prot, traj, no_rows))
            no_rows = no_rows - 400
            # print(no_rows)
            theta_index = range(theta_row)[0::10] ## index of each 10 elements including the first element
            # to have data 4 ns b4 first beta sheet unfolds
            # theta_index = range(theta_row)[no_rows::10]
            new_theta_file = np.ndarray(
                shape=(len(theta_index), theta_col))  # my new matrix
            for i in range(len(theta_index)):
                # adding each 10 row from theta_file to the new_theta_file
                new_theta_file[i] = theta_file[theta_index[i]]
            qn_row = np.shape(qn_file)[0]
            qn_col = np.shape(qn_file)[1]
            qn_index    = range(qn_row)[0::10]
            # qn_index = range(qn_row)[no_rows::10]
            new_qn_file = np.ndarray(shape=(len(qn_index), qn_col))
            for i in range(len(qn_index)):
                new_qn_file[i] = qn_file[qn_index[i]]
            theta_data = new_theta_file[:, 3]
            # theta_data = np.genfromtxt(thet_file, comments="#")[:,3]
            len_theta = len(theta_data)
            qn_data = new_qn_file[:len_theta, qn_columns[prot]]
            # print('traj%d, prot:%s'%(traj, prot), len_theta, 'qn:',len(qn_data))
            theta_ngfp.extend(
                theta_data) if prot == "ngfp" else theta_gfpc.extend(theta_data)
            qn_ngfp.extend(
                qn_data) if prot == "ngfp" else qn_gfpc.extend(qn_data)
    qnn = np.array(qn_ngfp)
    qnc = np.array(qn_gfpc)
    tn = np.array(theta_ngfp)
    tc = np.array(theta_gfpc)
    return qnn, tn, qnc, tc
qn_na, t_na, qn_ca, t_ca = get_qn_theta_data(patha)
qn_nb, t_nb, qn_cb, t_cb = get_qn_theta_data(pathb)
### produce pair list of [qn, tn]
list_1 = [qn_na, t_na, qn_ca, t_ca, qn_nb, t_nb, qn_cb, t_cb]
list_2 = []
for item in range(0, len(list_1) - 1, 2):
    list_2.append([list_1[item], list_1[item+1]])
# print(list_2[0][1])
print(len(list_2))
# exit()
path_dict = {0: 'major_ngfp', 1: 'major_gfpc',
             2: 'minor_ngfp', 3: 'minor_gfpc'}
# for path_pro in range(len(list_2)):
for key, value in path_dict.items():
    plt.figure(figsize=(10, 7))
    x = list_2[key][1]
    y = list_2[key][0]
    x = x/90
    X = np.array(list(zip(x, y)))
    data = X
    #this is for complete linkage
    # dend = shc.dendrogram(shc.linkage(data, method='complete'))
    linkage_type = 'ward' #"complete"  # "ward"
    dend = shc.dendrogram(shc.linkage(data, method=linkage_type),truncate_mode='lastp',p=12, leaf_rotation=45., leaf_font_size=15., show_contracted=True)
    plt.xlabel('Cluster Size')
    plt.ylabel('Distance')
    plt.title("Dendrogram")
    plt.savefig('dendrogram-%s-alldata.png'%value)
    # plt.show()
    # # use the Calinski-Harabasz score to determine number of clusters
    pSF = []
    linkage_type = "complete" #"ward"
    for number_clust in range(2, 7):
        cluster = AgglomerativeClustering(n_clusters=number_clust, affinity='euclidean', linkage=linkage_type)
        y = cluster.fit_predict(data)
        labels = cluster.labels_
        pSF.append(metrics.calinski_harabasz_score(data, labels))
    plt.plot(range(2, 7), pSF)
    plt.plot(range(2, 7), pSF)
    plt.scatter(range(2, 7), pSF, c = 'blue')
    plt.title('Calinski Harabasz Score vs. clusters (%s Linkage)'%linkage_type)
    plt.xlabel('Number of clusters')
    plt.ylabel('Calinski-Harabasz')
    plt.savefig('CalinskiHarabasz_Agglom-%s_linkage-%s-alldata.png' %
                (linkage_type, value), dpi=300, bbox_inches="tight")
    # plt.show()
    plt.close()
    # use the silhouette score to determine the number of clusters
    silh = []
    for number_clust in range(2, 7):
        cluster = AgglomerativeClustering(n_clusters=number_clust, affinity='euclidean', linkage=linkage_type)
        y = cluster.fit_predict(data)
        labels = cluster.labels_
        silh.append(metrics.silhouette_score(data, labels, metric='euclidean'))
        message = "For n_clusters = {} The {} silhouette_score is: {}"
        print(message.format(number_clust,  number_clust,round(metrics.silhouette_score(data, labels, metric='euclidean'), 2)))    
    plt.plot(range(2, 7), silh)
    plt.plot(range(2, 7), silh)
    plt.scatter(range(2, 7), silh, c = 'blue')
    plt.title('Silhouette Score vs. clusters (%s linkage)'%linkage_type)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette') 
    plt.savefig('AvSilhouette_Agglom-%s_linkage-%s-alldata.png' %
                (linkage_type, value), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
    # calculate the Davies Bouldin score (DBI) for agglomerative clustering
    scores = []
    centers = list(range(2,7))
    for center in centers:
        cluster = AgglomerativeClustering(n_clusters=center, affinity='euclidean', linkage=linkage_type)
        y = cluster.fit_predict(data)
        labels = cluster.labels_
        scores.append(metrics.davies_bouldin_score(data, labels))
    plt.plot(centers, scores, linestyle='--', marker='o', color='b')
    plt.xlabel('clusters')
    plt.ylabel('Davies Bouldin score')
    #plt.title('Davies Bouldin score vs. clusters (Single linkage)')
    #plt.title('Davies Bouldin score vs. clusters (complete linkage)')
    plt.title('Davies Bouldin score vs. clusters (%s linkage)'%linkage_type)
    #plt.savefig('DBIscore_Agglomsingle_shopping.png')
    #plt.savefig('DBIscore_Agglomcomplete_shopping.png')
    plt.savefig('DBIscore_Agglom-%s_linkage-%s-alldata.png' %
                (linkage_type, value), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
    # we know the number of clusters for complete linkage for our dataset is 4 or 5 (from most scores it is 5)
    # the next step is to group the data points into these 5 clusters
    for n_cluster in range(2,7):
        cluster = AgglomerativeClustering(n_clusters=n_cluster, affinity='euclidean', linkage=linkage_type)
        cluster.fit_predict(data)
        #save cluster labels in a file
        savetxt('clusters.csv', cluster.labels_, fmt = '%d', delimiter=',')
        ### plot the centroids
        from sklearn.neighbors.nearest_centroid import NearestCentroid
        clf = NearestCentroid()
        clf.fit(data, cluster.fit_predict(data))
        centroids = clf.centroids_
        print(centroids)
        # plot the clusters in the 2D space
        # plt.figure(figsize=(10, 7))
        plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.ylabel(r'$Q_N$')
        plt.xlabel(r'$\theta (^{\circ}) $')
        # plt.tick_params(top='off', bottom='off', labelleft='on', labelbottom='off')
        # mix ticks and xticklabels
        plt.xticks([0, 0.28, 0.56, 0.84, 1], [0, 25, 50, 75, ' '])
        plt.yticks([i*.2 for i in range(0, 6)])
        plt.title("Agglomerative clustering with %s linkage with %d clusters"%(linkage_type, n_cluster))
        image_name = 'agglomerativeclusters%s-linkage-%d-clusters-%s-alldata.png' % (
            linkage_type, n_cluster, value)
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    c='black', s=300, alpha=0.8, marker='*')
        plt.savefig(image_name, dpi=300, bbox_inches='tight' )
        # plt.show()
        plt.close()
