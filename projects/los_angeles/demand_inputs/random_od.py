import pandas as pd 
import numpy as np 
import random 
import os 

absolute_path = os.path.dirname(os.path.abspath(__file__))

def random_od(target_od):
    nodes_df = pd.read_csv(absolute_path+'/../network_inputs/osm_nodes.csv')
    # print(nodes_df.head())
    
    o_list = np.random.choice(nodes_df['node_id_igraph'], size=target_od)
    d_list = np.random.choice(nodes_df['node_id_igraph'], size=target_od)
    od_df = pd.DataFrame({'node_id_igraph_O': o_list, 'node_id_igraph_D': d_list})
    od_df['agent_id'] = np.arange(od_df.shape[0])
    od_df = od_df[['agent_id', 'node_id_igraph_O', 'node_id_igraph_D']]
    print(od_df.head())

    od_df.to_csv(absolute_path+'/random_od_{}.csv'.format(target_od), index=False)

if __name__ == '__main__':

    random_od(1000)