# skill-segmentation
Exploratory data analysis for job skills segmentation.


## Goal

This code aims at clustering job skills given applicant data. The data is given as a one hot encoded dataframe where each row represent a candidate. The programme reads this data, performs Kmeans clustering. The result of this clustering in then represented and exported as a graph. 

## Install

```bash
git clone https://github.com/simonzabrocki/skill-segmentation.git

cd skill-segmentation

pip install requirements.txt
```

## How to

To run the scripts with the default parameters:

```bash
python segmentation.py
```

The graph is exported as pdf in output/ folder.

For custom analysis run:

```bash
python segmentation.py --path data/hot_encoded.parquet --n_clusters 12 n_skills 5
```

With:
- path: path of a one hot encoded parquet data file
- n_clusters: The number of clusters used in kmeans
- n_skills: The number of most frequent ones skills displayed per clusters

## Improvement

In terms of clustering, a more natural but similar approach would be to use Non Negative matrix factorization rather than Kmeams. 

In terms of visualisation, the size and color attributes of the graph could be improve to reflect properties of the cluster (connectivity, number of candidates per skill/cluster)


## Author

Simon Zabrocki (simon.zabrocki@gmail.com)

## Acknowledgement

This repository is a simplified and updated version of the work done at Manatal during my 2018 summer internship.