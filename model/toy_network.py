import sys 
import time 
import random
import logging
import numpy as np
import pandas as pd 
import geopandas as gpd
from shapely.wkt import loads
from residual_demand_assignment import assignment

random_seed = 0
random.seed(0)
np.random.seed(0)

def main():

    ### scen_name
    scen_nm = '10s'
    home_dir = '/home/bingyu/Documents/residual_demand'
    work_dir = '/home/bingyu/Documents/residual_demand'
    scratch_dir = '/home/bingyu/Documents/residual_demand'

    ### input files
    network_file_edges = work_dir + '/projects/toy_network/network_inputs/links.csv'
    network_file_nodes = work_dir + '/projects/toy_network/network_inputs/nodes.csv'
    demand_files = []
    simulation_outputs = scratch_dir + '/projects/toy_network/simulation_outputs'

    ### log file*
    logging.basicConfig(filename=simulation_outputs+'/log/{}.log'.format(scen_nm), level=logging.INFO, force=True, filemode='w')

    ### network processing
    edges_df = pd.read_csv( network_file_edges )
    edges_df = gpd.GeoDataFrame(edges_df, geometry=edges_df['geometry'].map(loads))
    edges_df['fft'] = edges_df['length']/edges_df['max_kmph']*3.6
    edges_df['t_avg'] = edges_df['fft']
    edges_df['normal_capacity'] = edges_df['capacity']
    edges_df['normal_fft'] = edges_df['fft']
    edges_df['u'] = 0
    edges_df['v'] = 0
    edges_df['is_highway'] = 1
    edges_df['edge_str'] = edges_df['start_nid'].astype('str') + '-' + edges_df['end_nid'].astype('str')
    edges_df.to_csv(simulation_outputs + '/network/edges_{}.csv'.format(scen_nm), index=False)
    edges_df = edges_df.set_index('edge_str')
    ### closures
    # closed_links = pd.read_csv(work_dir + '/projects/tokyo_osmnx/network_inputs/20160304_closed_links.csv')
    
    nodes_df = pd.read_csv( network_file_nodes )
    nodes_df = nodes_df.set_index('node_id')

    ### demand processing
    od_list = []
    for trip_id in range(100):
        od_list.append([trip_id, 0, 4, trip_id*36])
    for trip_id in range(100, 120):
        od_list.append([trip_id, 1, 4, (trip_id-100)*180])
    od_all = pd.DataFrame(od_list, columns=['agent_id', 'origin_nid', 'destin_nid', 'dept_time'])
    od_all['hour'] = od_all['dept_time']//3600
    quarter_counts = 360
    od_all['quarter'] = od_all['dept_time']%3600//int(3600/quarter_counts)
    od_all.to_csv(simulation_outputs+'/trip_info/od_{}.csv'.format(scen_nm), index=False)
    logging.info('Generate {} OD pairs'.format(od_all.shape[0]))
    
    ### run residual_demand_assignment
    assignment(edges_df=edges_df, nodes_df=nodes_df, od_all=od_all, simulation_outputs=simulation_outputs,
                scen_nm=scen_nm, hour_list=[0, 1], quarter_counts=quarter_counts)

    return True

if __name__ == "__main__":
    status = main()
