#!/usr/bin/env python
# coding: utf-8

# # BikeMi Stalls *k*-Means Clustering

# In[1]:


# path manipulation
from pathlib import Path
from typing import Union

# data manipulation
import pandas as pd
import geopandas

# plotting
import matplotlib.pyplot as plt
import contextily as cx
import seaborn as sns

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# set settings for seaborn
sns.set_style(style="whitegrid", rc={"grid.color": ".9"})
sns.set_palette(palette="deep")

# customise matplotlib and sns plot dimensions
plt.rcParams["figure.figsize"] = [12, 6]
plt.rcParams["figure.dpi"] = 100
title_font: dict[str] = {"fontname": "DejaVu Sans Mono"}

# create paths
milan_data: Path = Path("../data/milan")

# load stalls data
bikemi_selected_stalls: pd.DataFrame = pd.read_csv(
    milan_data / "bikemi-selected_stalls-with_nils.csv",
    index_col="numero_stazione"
)

nils: geopandas.GeoDataFrame = (
    geopandas.read_file(milan_data / "administrative-nil.geo.json")
        .rename(str.lower, axis=1)
        .set_index("id_nil")
        .filter(["nil", "geometry"])
)


# ## *k*-Means Clustering Review

# After our first data selection to remove outliers and restrict the spatial area in which we are conducting our analysis,
# we are still left with more than 200 stations, spread across 25 neighbourhoods out of 88 - identified by the acronym NIL, i.e. *nuclei d'identità locale*. This figure might still be too high, especially as far as multivariate models are concerned: indeed, shrinkage will be necessary in order to avoid highly correlated features (*multicollinearity*).
# However, it is still in our interests to reduce the number of series to model even for the univariate forecasting: fitting twenty or two-hundred series is a different task. Even inspecting the correlation across series becomes a daunting task with such a great number of features.
# 
# To do so, we will use *k*-Means clustering - a popular method widely used in the sharing-services literature, especially to identify "virtual stations" in free-float services <cite id="sfr96">(Ma et al., 2018)</cite> or to "visualize the spatial distribution of DBS [Dockless Bike Sharing] and taxis around metro stations" <cite id="bh87a">(Li et al., 2019)</cite>. These "virtual stations" are also called *TAZ*, i.e. Traffic Analysis Zones, and are also used for docked-bike sharing services <cite id="6ozrx">(Cantelmo et al., 2020)</cite>.
# 
# ### *k*-Means Clusttering and its Shortcomings
# 
# In a few words, with *k*-Means clustering  we "want to partition the observations into $K$ clusters such that the total within-cluster variation [also known as  *inertia*], summed over all $K$ clusters, is as small as possible" <cite id="5wuoa">(Sohil et al., 2021)</cite>. The objective function to optimise is usually the squared Euclidean distance. Simply put, *k*-Means "aims to partition n observations into $K$ clusters, represented by their centres or means. The centre of each cluster is calculated as the mean of all the instances belonging to that cluster" <cite id="oc9xp">(Li et al., 2019)</cite> and "is extremely efficient and concise for the classification of equivalent multidimensional data" such as sharing services data <cite id="4rpc6">(Li et al., 2019)</cite>.
# 
# The algorithm begins with randomly choosing clusters centres and, with each iteration, the centres are re-calculated to reduce the partitioning error - which decreases monotonically, as $K$ increases. Basically, in this second step the algorithm "creates new centroids by taking the mean value of all of the samples assigned to each previous centroid [...] until the centroids do not move significantly" <cite id="nt8nm">(<i>Clustering</i>, n.d.)</cite>. However, despite similarity always increasing, values of $K$ that are too great defy the purpose of this classification algorithm. For this reasons, practitioners and researchers have come up with more or less sophisticated ways to assess the optimal number of clusters, the most common of which is the so-called "Elbow method". In a few words, a chosen performance metric is computed for each number of clusters and the optimum is represented by the point at which the performance improvements start to marginally decline.
# 
# *k*-Means clustering scales well with the number of samples $n$, but assumes convex shapes (i.e., has worse performances where the "true" clusters have elongated or irregular shapes) <cite id="xpmoo">(<i>Clustering</i>, n.d.)</cite>. Besides, since the initial position of the cluster is random, it might take some attempt for the algorithm to converge. More importantly, however, *k*-Means is sensitive to the scales of the variables in the data, so normalising the feature matrix is a crucial step. Finally, *k*-Means assumes continuous data and is not designed to handle categorical features: "[T]he *k*-Means algorithm only works on numerical data, i.e. variables that are measured on a ratio scale [...], because it minimises a cost function by changing the means of clusters. This prohibits it from being used in applications where categorical data are involved" <cite id="acr6r">(Huang, 1998)</cite>.
# 
# ### Alternative Approaches to *k*-Means Clustering
# 
# Some workarounds have long been known in the literature: as an example, by changing the distance metrics from the Euclidean distance to the Gower distance, which is a measure of similarity <cite id="1bw6w">(Gower, 1971)</cite>, *k*-Means can be adjusted to cluster categorical variables. However, similarity-based algorithms often display "quadratic computational costs", and their usage "hinders" the reputation of *k*-Means, which is know to be a pretty fast algorithm. As an alternative, new methods have been developed, such as the *k*-Modes <cite id="d9fnm">(Huang, 1998)</cite>, which scale well with the larger and larger datasets we have been getting accustomed to in the past ten years.
# 
# Given the data availability, this analysis should not benefit from more sophisticated clustering algorithms. While the chosen sample of stalls would even be small enough to justify an attempt with similarity scores, other meaningful regressor are absent from the data - especially weather data. The features that we could extract from the data, like the neighbourhood, are in fact highly correlated with the geographic coordinates themselves. Another avenue of research might be to exploit **spatial attributes**, such as the nature and number of building in an area, to provide quantitative measurements of the characterisation of one geographic area. For example, the number of train and metro stations (and, to a lesser extent, of bus stops), perhaps within a certain radius from the individual station, could be used. The share of residential buildings and firms, or the extension of green areas such as parks, might be another one. Some, if not most, of this data could be retrieved using open sources like Open Street Maps, but is out of the scope for this analysis.
# 
# The situation changes when we move from a static clustering approach to a dynamic one. Time series clustering <cite id="gxpvj">(Ali et al., 2019)</cite> seems a promising path of research, as it would allow to implement a novel procedure for feature selection, which could be layered on top of a geographic (or spatial) clustering. Time-series clustering could be paired with spatial cross-validation folds, whose importance has been underlined since at least the past ten years <cite id="v3vci">(Brenning, 2012)</cite> <cite id="5ch77">(Roberts et al., 2017)</cite>.
# 
# We know of only one peculiar approach, implemented by Chen and his coauthors <cite id="d003u">(Chen et al., 2016)</cite>, who implemented a weighted correlation network to account for "contextual factors". Indeed, according to the authors, "bike usage patterns are highly context dependent: time of the day, day of the week, weather condition, social events, and traffic conditions can all lead to different bike usage patterns". The authors identify two types of context-dependencies: "the *common contextual factors* that occur frequently and affect all the stations, such as time and weather" and "the *opportunistic contextual factors* that happen irregularly and only affect a subset of stations, such as social and traffic events" <cite id="vqgjn">(Chen et al., 2016)</cite>. The idea behind this is to "build a [set of] statistical clustering template[s] using historical records (e.g., a cluster template for sunny weekday rush hours)", that are automatically switched between according to the values of the other regressors.
# 
# To provide a more practical example, let us assume that there are four clusters, the results of the combination two independent variables: whether it is a weekday or weekends, and whether there is sunny or rainy weather. Each of these templates is used to fit a *k*-Means algorithm (actually, Chen and his colleagues build a "correlation network") and the resulting clustering labels are 'switched on and off' as the weather changes from sunny to rainy and the time from weekdays to weekends.

# ## *k*-Means Clustering Evaluation Metrics

# The main, if not only, hyperparameter to tune in *k*-Means is, indeed, the number of clusters.
# As most machine learning methods, *k*-Means is a quite old algorithm and has thus been widely studied.
# The first metric to look at is indeed the inertia, or *within-cluster sum-of-squares* ($WSS$), which is usually computed using the Euclidean norm:
# 
# \begin{align}
# & WSS = \sum_{i=0}^{n}\underset{\mu_{j}\in{C}}{min}(||x_{i} - \mu_{j}||^2)
# \end{align}
# 
# Other norms can be used; however, some other performance metrics can only be calculated with the Euclidean distance. While clustering is usually presented as an unsupervised machine-learning technique, a first set of metrics is actually computed comparing the clustering with the ground truths. This is necessary because the evaluation metrics should not take into account the absolute number of clusters, yet if this clustering defines "separations of the data similar to some ground truth set of classes" <cite id="pdryw">(<i>Clustering</i>, n.d.)</cite>.
# 
# Among these metrics there are the *Rand index*, mutual information-based scores, measures of completeness and homogeneity (or both, such as the *V-measure*) and Fowlkes-Mallows scores. There is, however, another category of metrics that does not require a comparison with the ground-truth: the *silhouette coefficients*, the *Calinski-Harabasz index*, computed as  and  the *Davies-Bouldin index*. The downside of these evaluation metrics is that they are not robust against "unconventional" shapes and tend to display "better" values when the clusters are convex.
# 
# These measures are computed for each set of data points and the labels assigned with the *k*-Means. The silhouette score is computed as $s = \frac{b - a}{max(a, b)}$, where $a$ is the mean distance between a sample and all points in the same class and $b$ is the average distance between the sample and the points in the *next nearest* cluster. The silhouette coefficients for all samples are averaged and a higher silhouette coefficient relates to a model with better defined clusters <cite id="4ixnl">(<i>Clustering</i>, n.d.)</cite>. For this reason, the silhouette score is defined within $[-1;1]$, and scores closer to $-1$ denote incorrect clustering and values near $0$ indicate overlapping data points.
# 
# The Calinski-Harabasz Index is defined as:
# 
# \begin{align}
# & s = \frac{\mathrm{tr}(B_k)}{\mathrm{tr}(W_k)} \times \frac{n_E - k}{k - 1}
# \end{align}
# 
# Where $B_k$ is the dispersion between clusters and $W_k$ is the one within clusters and $n$ is the number of observations in the dataset $E$. Because of this, the index is a raw number (i.e., it is not normalised between, say, $0$ and $1$). Since we want the clusters to be as far apart as possible and the points inside to be as close as possible, we will choose $K$ such as it maximises the Calinski-Harabasz index.
# 
# Finally, the Davies-Bouldin index indicates a better separation between clusters the lower it is, and is computed as the average similarity across the two closer classes $C_i$ and $C_k$. The similarity measure between $C_i$ and $C_k$ is called $R_{ij}$ and is defined as $R_{ij} = \frac{s_i + s_j}{d_{ij}}$, where $s$ is the average distance between each point of a cluster and its centre, and $d_{ij} is the distance between the two clusters' centroids. The Davies-Bouldin index is thus formally defined as such:
# 
# \begin{align}
# & DB = \frac{1}{k} \sum_{i=1}^k \max_{i \neq j} R_{ij}
# \end{align}

# In[2]:


from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline


def filter_coords(data: pd.DataFrame, cols=None) -> pd.DataFrame:
    if cols is None:
        cols = ["longitudine", "latitudine"]
    return data.filter(cols)


selected_stalls_lonlat: pd.DataFrame = bikemi_selected_stalls.pipe(filter_coords)


# ## Fitting *k*-Means on BikeMi Stalls Geographic Data

# We define two functions, using Python and its `scikit-learn` library. The first one fits one instance of *k*-Means up to an arbitrarily large number of clusters, and collects the chosen metrics to evaluate their performance. As a graphic aid, the second function plots this data in a grid, to ease graphic analysis. It is worth nothing that `scikit-learn` is widely used in the literature <cite id="rkywj">(Cantelmo et al., 2020)</cite> <cite id="pgl48">(Li et al., 2019)</cite>.
# 
# The $WSS$ or inertia is not really indicative of a significant value of $K$. The silhouette coefficient is greatest for a quite small number of clusters (smaller than 5) and in general not greater than 20. This appears to be the case for the Calinski-Harabasz index too and, while the same cannot be said of the Davies-Bouldin index, beyond said threshold the index is bound to improve. Our final choice should fall on 18, as we believe that clustering the whole set of stations into three great TAZ bears no practical utility for the purpose of the policymaker - i.e., forecasting bikes demand in Milan.

# In[3]:


def get_kmeans_metrics(data, k_max, random_state) -> pd.DataFrame:
    scaled_data = StandardScaler().fit_transform(data)
    k_range = range(2, k_max + 1)

    Metrics = tuple[float, float, float, float]

    def compute_scores(source_data, num_clusters, rand) -> Metrics:
        kmeans = KMeans(num_clusters, random_state=rand).fit(source_data)

        inertia = kmeans.inertia_
        silhouette_coefficient = silhouette_score(source_data, kmeans.labels_, metric="euclidean")
        calinski_harabasz = calinski_harabasz_score(source_data, kmeans.labels_)
        davies_bouldin = davies_bouldin_score(source_data, kmeans.labels_)

        return inertia, silhouette_coefficient, calinski_harabasz, davies_bouldin

    scores: list[Metrics] = [compute_scores(scaled_data, k, random_state) for k in k_range]

    output: pd.DataFrame = pd.DataFrame(
        scores,
        index=k_range,
        columns=["inertia", "silhouette_coefficient", "calinski_harabasz", "davies_bouldin"]
    )

    output.index.name = "k"

    return output


def plot_all_metrics(metrics_data: pd.DataFrame) -> None:
    def plot_metric(dataf: pd.DataFrame, col: str, _ax: plt.Axes):
        plot_title: str = col.replace("_", " ").title()
        _ax.plot(dataf[col])
        _ax.set_title(f"{plot_title}", **title_font)

    fig, ax = plt.subplots(2, 2, figsize=(12, 12))

    for ax, metric in zip(ax.reshape(-1), metrics_data.columns):
        plot_metric(metrics_data, metric, ax)


selected_stalls_metrics = get_kmeans_metrics(selected_stalls_lonlat, 70, 42)

plot_all_metrics(selected_stalls_metrics)


# However, as mentioned above, the total number of neighbourhoods inside our area of analysis is 25: at this point, the policymaker might just be better off aggregating the data by neighbourhood, without going through the hassle of computing the *k*-Means. Indeed, in this context the purpose of fitting a *k*-Means should be to isolate a small amount of Traffic Analysis Zones inside each neighbourhood, in order to be able to aggregate together stations that are quite close to each other and that cannot all be used at once as independent variables for the forecasting models without injecting error in the estimates via multicollinearity. In this way, it would also be possible to include neighbourhood fixed effects - which would otherwise be impossible to add to the regression if data was aggregated ad that level.

# In[4]:


def fit_kmeans(input_data, k, rand) -> object:
    return make_pipeline(StandardScaler(), KMeans(n_clusters=k, random_state=rand)).fit(input_data)


def assign_clusters(input_data: pd.DataFrame, fitted_kmeans) -> pd.DataFrame:
    return input_data.assign(cluster=fitted_kmeans.labels_)


def make_geodataframe(
        data: pd.DataFrame,
        cols=None,
        crs: int = 4326
) -> geopandas.GeoDataFrame:
    if cols is None:
        cols = ["longitudine", "latitudine"]
    return data.pipe(
        geopandas.GeoDataFrame,
        geometry=geopandas.points_from_xy(
            data[cols[0]],
            data[cols[1]],
            crs=crs)
    )


def plot_clusters(
        geodata: geopandas.GeoDataFrame,
        cluster_col: str,
        layer: Union[geopandas.GeoDataFrame, None]
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    ax.axis("off")

    num_of_clusters: int = geodata[cluster_col].nunique()
    ax.set_title(f"Bikemi Stalls, Result of KMeans Clustering ($k$ = {num_of_clusters})", **title_font)

    if layer is not None:
        layer.to_crs(3857).plot(ax=ax, cmap="summer", alpha=0.05)
    geodata.to_crs(3857).plot(column=cluster_col, ax=ax)
    cx.add_basemap(ax=ax)

    plt.show()


# In[5]:


kmeans_18 = fit_kmeans(selected_stalls_lonlat, k=18, rand=42)

selected_stalls_18k: geopandas.GeoDataFrame = (
    selected_stalls_lonlat
        .pipe(assign_clusters, fitted_kmeans=kmeans_18[-1])
        .pipe(make_geodataframe)
)

plot_clusters(
    selected_stalls_18k,
    cluster_col="cluster",
    layer=nils.sjoin(selected_stalls_18k)
)


# If we sort the metrics matrix by the values of `sihlouette_coefficient`, the first number of clusters greater than 25 than we meet is 46. This, however, is only the 18th best value for $K$ according to the `silhouette_score`.

# In[6]:


def get_k_greater_than_25(metrics_data: pd.DataFrame, metric: str = "silhouette_coefficient") -> pd.DataFrame:
    return metrics_data.sort_values(metric, ascending=False).assign(ranking=range(metrics_data.shape[0])).query(
        "k > 25").head(1)


get_k_greater_than_25(selected_stalls_metrics)


# In[7]:


kmeans_46 = fit_kmeans(selected_stalls_lonlat, k=46, rand=42)

selected_stalls_46k: geopandas.GeoDataFrame = (
    selected_stalls_lonlat
        .pipe(assign_clusters, kmeans_46[-1])
        .pipe(make_geodataframe)
)

selected_nills_46k: geopandas.GeoDataFrame = nils.sjoin(selected_stalls_46k)

plot_clusters(
    selected_stalls_46k,
    cluster_col="cluster",
    layer=selected_nills_46k
)


# After choosing the value for $K$, stall data is aggregated to compute the coordinates of these new "virtual stalls" as the average of the coordinates of the stations inside the cluster. Because of how spatial joins work, one neighbourhood is not plotted, and there seems to be one cluster without left without a NIL. However, the identifier (`12`, i.e. Maciachini - Maggiolina) is still correctly assigned. As the last step, these virtual clusters are assigned with a join operation to the list of selected stalls, and exported.

# In[8]:


def plot_virtual_clusters(
        virtual_clusters: geopandas.GeoDataFrame,
        layer: Union[geopandas.GeoDataFrame, None]
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    if layer is not None:
        layer.to_crs(3857).plot(ax=ax, cmap="summer", alpha=0.05)
    virtual_clusters.to_crs(3857).plot(ax=ax)
    cx.add_basemap(ax=ax)

    plt.axis("off")
    plt.title("Virtual Stalls - KMeans Clustering ($k$=46)", **title_font)

    plt.show()


virtual_stalls: geopandas.GeoDataFrame = (
    selected_stalls_46k
        .groupby("cluster")[["longitudine", "latitudine"]]
        .mean()
        .pipe(make_geodataframe)
        .rename({"longitudine": "lon_cluster", "latitudine": "lat_cluster"}, axis=1)
        .sjoin(nils, how="left")
        .rename({"index_right": "id_nil"}, axis=1)
)

plot_virtual_clusters(virtual_stalls, layer=selected_nills_46k)


# In[9]:


clustered_selected_stalls: pd.DataFrame = (
    bikemi_selected_stalls.filter(["nome_stazione"])
        .join(selected_stalls_46k.drop(columns="geometry"))
        .join(virtual_stalls.drop(columns="geometry"), on="cluster")
        .rename(columns={"id_nil": "cluster_id_nil", "nil": "cluster_nil"})
)

clustered_selected_stalls.to_csv(milan_data / "bikemi-selected_stalls-clusters.csv", index=True)

