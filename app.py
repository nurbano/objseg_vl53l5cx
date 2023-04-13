from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import Birch
import random
from sklearn import metrics
from PIL import Image


# Funciones

def ransac(point_cloud, max_iterations, distance_ratio_threshold):

    inliers_result = set()
    while max_iterations:
        max_iterations -= 1
        # Add 3 random indexes
        random.seed()
        inliers = []
        while len(inliers) < 3:
            random_index = random.randint(0, len(point_cloud.X)-1)
            inliers.append(random_index)
        # print(inliers)
        try:
            # In case of *.xyz data
            x1, y1, z1, _ = point_cloud.loc[inliers[0]]
            x2, y2, z2, _ = point_cloud.loc[inliers[1]]
            x3, y3, z3, _ = point_cloud.loc[inliers[2]]
        except:
            # In case of *.pcd data
            x1, y1, z1 = point_cloud.loc[inliers[0]]
            x2, y2, z2 = point_cloud.loc[inliers[1]]
            x3, y3, z3 = point_cloud.loc[inliers[2]]
        # Plane Equation --> ax + by + cz + d = 0
        # Value of Constants for inlier plane
        a = (y2 - y1)*(z3 - z1) - (z2 - z1)*(y3 - y1)
        b = (z2 - z1)*(x3 - x1) - (x2 - x1)*(z3 - z1)
        c = (x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1)
        d = -(a*x1 + b*y1 + c*z1)
        plane_lenght = max(0.1, math.sqrt(a*a + b*b + c*c))

        for point in point_cloud.iterrows():
            index = point[0]
            # Skip iteration if point matches the randomly generated inlier point
            if index in inliers:
                continue
            try:
                # In case of *.xyz data
                x, y, z, _ = point[1]
            except:
                # In case of *.pcd data
                x, y, z = point[1]

            # Calculate the distance of the point to the inlier plane
            distance = math.fabs(a*x + b*y + c*z + d)/plane_lenght
            # Add the point as inlier, if within the threshold distancec ratio
            if distance <= distance_ratio_threshold:
                inliers.append(index)
        # Update the set for retaining the maximum number of inlier points
        if len(inliers) > len(inliers_result):
            inliers_result.clear()
            inliers_result = inliers

    # Segregate inliers and outliers from the point cloud
    inlier_points = pd.DataFrame(columns=["X", "Y", "Z"])
    outlier_points = pd.DataFrame(columns=["X", "Y", "Z"])
    for point in point_cloud.iterrows():
        if point[0] in inliers_result:
            inlier_points = pd.concat([inlier_points, pd.DataFrame({"X": [point[1]["X"]],
                                                                    "Y": [point[1]["Y"]],
                                                                    "Z": [point[1]["Z"]]})], ignore_index=True)
            continue
        outlier_points = pd.concat([outlier_points, pd.DataFrame({"X": [point[1]["X"]],
                                                                  "Y": [point[1]["Y"]],
                                                                  "Z": [point[1]["Z"]]})], ignore_index=True)

    return inlier_points, outlier_points, a, b, c, d


# Pantalla Princial
st.title('Segmentación en dataset segobj_VL53L5CX')
st.text('Autores:' + "\n" + 'Nicolás Urbano Pintos (Grupo TAMA UTN FRH/DRL CITEDEF)'
        + "\n" + 'Héctor Alberto Lacomi (Grupo ASE UTN FRH/DRL CITEDEF)'
        + "\n" + 'Mario Blas Lavorato(Grupo TAMA UTN FRH)')

st.text('urbano.nicolas@gmail.com')

df_dataset = pd.DataFrame({
    'first column': ['escenario_1',
                     'escenario_2',
                     'escenario_3',
                     'escenario_4',
                     'escenario_5',
                     'escenario_6',
                     'escenario_7',
                     'escenario_8'],

})

df_clustering = pd.DataFrame({
    'first column': ['K-means',
                     'DBSCAN',
                     'MeanShift',
                     'BIRD'],

})

with st.sidebar:
    option = st.selectbox(
        'Dataset',
        df_dataset['first column'])

    #porc_puntos = st.slider('% puntos', 0, 100, 5)
    tam_punto= st.slider("Tamaño de puntos",0,10,2)
    #piso_preg = st.checkbox('Sacar Piso')
    ransac_th = st.slider('Umbral RANSAC', 1, 100, 9)
    option_clustering = st.selectbox('Metodo',
                                     df_clustering['first column'])


'Dataset: ', option


# Apertura de Dataset

df_in = pd.read_csv("./escenarios/"+option+ ".xyz", delimiter=";",
                    usecols=[0, 1, 2, 4], names=["X", "Y", "Z", "clase"], header=0)
df_in.Y= -df_in.Y
X_= np.column_stack([df_in.X, df_in.Y, df_in.Z])


outlier_points = []
while(len(outlier_points) == 0):
    inlier_points, outlier_points, a, b, c, d = ransac(df_in, ransac_th, 50)
    # print(len(outlier_points))
    X_= np.column_stack([outlier_points.X, outlier_points.Y, outlier_points.Z])

if option_clustering== "K-means":
    with st.sidebar:
        n_clus = st.slider('Cantidad de Cluster', 1, 20, 2)
    with st.spinner('Agrupando...'):
        kmeans = KMeans(n_clusters=n_clus).fit(X_)
    st.success('Listo!')
    df_out=pd.DataFrame(data={"cat": kmeans.labels_})
    cantidad_cluster= n_clus
    
    
if option_clustering== "DBSCAN":
    with st.sidebar:
        eps_ = st.slider('Epsilon', 1, 200, 80)
        min_samples_ = st.slider('Min samples', 1, 10, 8)
    with st.spinner('Agrupando...'):
        cluster_db = DBSCAN( min_samples= min_samples_, eps=eps_).fit(X_)
    st.success('Listo!')
    df_out=pd.DataFrame(data={"cat": cluster_db.labels_})
    cantidad_cluster= len(np.unique(cluster_db.labels_))



    

if option_clustering== "MeanShift":
    with st.sidebar:
        cuantile= st.slider("Cuantil del BW",0.1,1.0, 0.2 )
    with st.spinner('Agrupando...'):

        bandwidth = estimate_bandwidth(X_, quantile=cuantile, n_samples=500)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(X_)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        cantidad_cluster=n_clusters_
        df_out=pd.DataFrame(data={"cat": labels})
    st.success('Listo!')


if option_clustering== 'BIRD':
    with st.sidebar:
        n_clus = st.slider('Cantidad de Cluster', 1, 20, 2)
    clustering_birch = Birch(n_clusters=n_clus)
    clustering_birch.fit(X_)
    labels=clustering_birch.labels_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    cantidad_cluster=n_clusters_
    df_out=pd.DataFrame(data={"cat": labels})

image = Image.open("./escenarios/"+option+ ".jpg")

st.image(image)    
fig = px.scatter_3d(data_frame=df_in, x="X", y="Y", z="Z", color="clase",
                    title="Clases Reales",  color_continuous_scale=px.colors.sequential.Viridis)
fig.update_traces(marker_size=tam_punto)
camera = dict(
    eye=dict(x=-0.1, y=-2, z=-1)
)
fig.update_layout(scene_camera=camera)
fig.update_layout(showlegend=False)
st.plotly_chart(fig)

df_outlier = pd.DataFrame(data={"X": outlier_points.X, "Y": outlier_points.Y,
                          "Z": outlier_points.Z, "pred": np.zeros(len(outlier_points.X))})
df_inlier = pd.DataFrame(data={"X": inlier_points.X, "Y": inlier_points.Y,
                         "Z": inlier_points.Z, "pred": np.ones(len(inlier_points.X))+9})
fig2 = px.scatter_3d(data_frame=df_outlier, x="X", y="Y",
                     z="Z", color="pred", title= "Detección de planos con RANSAC")
trace_1 = px.scatter_3d(data_frame=df_inlier, x="X",
                        y="Y", z="Z", color="pred")
fig2.add_trace(trace_1.data[0])
fig2.update_traces(marker_size=tam_punto)
fig2.update_layout(scene_camera=camera)
fig2.update_layout(showlegend=False)
st.plotly_chart(fig2)

st.metric(label="Clases predichas", value=cantidad_cluster+1)

df = pd.DataFrame(data= {'x': X_[:, 0], 'y': X_[:,1], 'z': X_[:,2]})
fig3 = px.scatter_3d(data_frame=df, x="x", y="y",z="z", color= df_out["cat"].astype("category"),
                    title= option_clustering + " - Vista 3D" , 
                    color_discrete_sequence=px.colors.qualitative.G10)
trace_2 = px.scatter_3d(data_frame=df_inlier, x="X",
                        y="Y", z="Z", color="pred")
fig3.add_trace(trace_2.data[0])
fig3.update_traces(marker_size = tam_punto)
fig3.update_layout(legend_itemsizing ='trace')
fig3.update_layout(scene_camera=camera)
st.plotly_chart(fig3)
