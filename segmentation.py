import argparse
import pandas as pd
import networkx as nx
import graphviz
from sklearn.cluster import KMeans


def process_encoded_data(path='data/hot_encoded.parquet', freq_thres=5):
    '''Read and filter skills by frequency'''
    df = pd.read_parquet(path)
    df = df.loc[:, (df.sum(axis=0) > freq_thres)]
    return df


def get_skill_cluster_df(df, n_clusters=10):
    '''Cluster and format results
    '''
    km = KMeans(n_clusters=n_clusters).fit(df)
    df['cluster'] = km.labels_
    
    return (
        df.melt(id_vars='cluster', var_name='skill')
          .query('value > 0')
          .groupby(['cluster', 'skill']).count()
          .rename(columns={'value': 'count'})
          .sort_values(by=['cluster', 'count'])
          .reset_index()
          .astype({'cluster': str})


    )


def get_top_n_by_cluster(skill_cluster_df, n_top=10):
    '''
    Select top skills by cluster
    '''
    top_df = (
        skill_cluster_df.groupby(['cluster']).apply(lambda x: x.tail(n_top).drop(columns=['cluster']))
                        .reset_index(level='cluster')
    )
    return top_df


def get_nodes(cluster_df):
    '''Get nodes from cluster_df
    '''
    nodes = []
    skill_sizes = cluster_df.groupby('skill').sum().reset_index()
    nodes += [(v[0], {'n_candidates': v[1]}) for v in skill_sizes.values]
    
    cluster_size =  cluster_df.groupby('cluster').sum().reset_index()
    nodes += [(v[0], {'n_candidates': v[1]}) for v in cluster_size.values]

    return nodes



def get_edges(cluster_df):
    '''Get edges from cluster_df
    '''
    edges = list(map(tuple, cluster_df[['cluster', 'skill']].values))
    return edges


def make_cluster_graph(cluster_df):
    '''
    Create the graph object from the dataframe
    '''
    edges = get_edges(cluster_df)
    nodes = get_nodes(cluster_df)
    Graph = nx.Graph()
    Graph.add_nodes_from(nodes)
    Graph.add_edges_from(edges)
    return Graph


class GraphDrawer():
    '''A class to draw the Graph.
    '''

    def __init__(self):
        '''Initialize the GraphDrawer
        '''
        return None

    def draw_node(self, dot, node):
        '''Draw a node of the graph.
        Args:
            dot(dot): dot object for drawing.
            node(node): Node of the graph.
        Returns:
            None, updates the dot object.
        '''
        dot.node(node[0], node[0], {
                                    'label': node[0],
                                    'fillcolor': '#e76f51',
                                    "shape": "circle",
                                    "peripheries": "0",
                                    'style': 'filled',
                                    'fontcolor': '#eeeeee',
                                    'fontname': 'roboto',
                                    'width': "2.1",
                                    'fontsize':'12',
                                    }
                 )

    def draw_edge(self, dot, a, b):
        '''Draw an edge of the graph.
        Args:
            dot(dot): dot object for drawing.
            a(node): Node of the graph.
            b(node): Node of the graph.
        '''
        dot.edge(a, b, color='#A9A9A9')

    def draw(self, G):
        '''Draw a full Graph
        Args:
            G(GraphModel): A graph.
        Returns:
            dot(obj): The plot object of the graph.
        '''
        dot = graphviz.Graph(engine='neato', graph_attr={'overlap' :'scale'})
        for node in G.nodes(data=True):
            self.draw_node(dot, node)
        for a, b in G.edges:
            self.draw_edge(dot, a, b)
        return dot
    
    
def main(args):
    
    df = process_encoded_data(path=args['path'])
    cluster_df = get_skill_cluster_df(df, n_clusters=args['n_clusters'])
    top_cluster_df = get_top_n_by_cluster(cluster_df, n_top=args['n_skills'])

    Graph = make_cluster_graph(top_cluster_df)
 
    save_path = f"output/graph_{args['n_clusters']}_cluster_{args['n_skills']}_skills"
    GraphDrawer().draw(Graph).render(save_path)
    

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(description='Cluster skills.')
    parser.add_argument('--path', type=str, help='Parquet data file path', default='data/hot_encoded.parquet')
    parser.add_argument('--n_clusters', type=int, help='Number of clusters', default=10)
    parser.add_argument('--n_skills', type=int, help='Number of top skills per clusters', default=5)

    args = parser.parse_args()
    main(vars(args))