import pandas as pd 
import numpy as np 
from scipy import stats

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    From: https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r * 1000


def main():
    nodes_df = pd.read_csv('../network_inputs/nodes_residual_demand.csv')
    # nodes_df = nodes_df.set_index('node_id_igraph')[['lon', 'lat']]
    nodes_df = nodes_df[['node_id_igraph', 'lon', 'lat']]
    print(nodes_df.head())

    demand_file_list = []
    for demand_file_id in [0, 1, 2]:
        demand_file_list.append(pd.read_csv('od_residual_demand_{}.csv'.format(demand_file_id)))
    demand_df = pd.concat(demand_file_list)

    demand_df = demand_df.merge(nodes_df, how='left', left_on='node_id_igraph_O', right_on='node_id_igraph')
    demand_df = demand_df.merge(nodes_df, how='left', left_on='node_id_igraph_D', right_on='node_id_igraph', suffixes=['_O', '_D'])
    # demand_df = demand_df.set_index('node_id_igraph_O').reindex(nodes_df.index)
    print(demand_df.head())

    distance_array = haversine(demand_df['lat_O'], demand_df['lon_O'], demand_df['lat_D'], demand_df['lon_D'])
    print(np.mean(distance_array), np.std(distance_array))
    print(np.percentile(distance_array, list(range(0, 100, 10))))
    print(stats.describe(distance_array))

if __name__ == '__main__':
    main()