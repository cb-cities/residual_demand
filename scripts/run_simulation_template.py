import os
import sys 
import time 
import logging
import numpy as np
import pandas as pd 
import geopandas as gpd
from shapely.wkt import loads
from residual_demand_assignment import assignment

abs_path = os.path.dirname(os.path.abspath(__file__))

def main(hour_list=None, quarter_list=None, closure_hours=None):


    ############# CHANGE HERE ############# 
    ### scenario name and input files
    scen_nm = 'scenario_name'

    ### input files
    network_file_edges = abs_path + '/../projects/test/network_inputs/test_edges.csv'
    network_file_nodes = abs_path + '/../projects/test/network_inputs/test_nodes.csv'
    closed_links_file = abs_path + '/../projects/test/network_inputs/test_closed_links.csv'
    demand_file = abs_path + '/../projects/test/demand_inputs/test_od.csv'
    simulation_outputs = abs_path + '/../projects/test/simulation_outputs'

 
    ############# NO CHANGE HERE ############# 
    ### log file
    logging.basicConfig(filename=simulation_outputs+'/log/{}.log'.format(scen_nm), level=logging.INFO, force=True, filemode='w')

    ### network processing
    edges_df = pd.read_csv( network_file_edges )
    edges_df = gpd.GeoDataFrame(edges_df, crs='epsg:4326', geometry=edges_df['geometry'].map(loads))
    edges_df = edges_df.sort_values(by='fft', ascending=False).drop_duplicates(subset=['start_nid', 'end_nid'], keep='first')
    ### pay attention to the unit conversion
    edges_df['fft'] = edges_df['length']/edges_df['maxspeed']*2.23694
    edges_df['edge_str'] = edges_df['start_nid'].astype('str') + '-' + edges_df['end_nid'].astype('str')
    edges_df['capacity'] = np.where(edges_df['capacity']<1, 950, edges_df['capacity'])
    edges_df['is_highway'] = np.where(edges_df['type'].isin(['motorway', 'motorway_link']), 1, 0)
    edges_df['normal_capacity'] = edges_df['capacity']
    edges_df['normal_fft'] = edges_df['fft']
    edges_df['t_avg'] = edges_df['fft']
    edges_df['u'] = edges_df['start_nid']
    edges_df['v'] = edges_df['end_nid']
    edges_df = edges_df.set_index('edge_str')
    ### closure locations
    closed_links = pd.read_csv(closed_links_file)
    for row in closed_links.itertuples():
        edges_df.loc[(edges_df['uniqueid']==getattr(row, 'uniqueid')), 'capacity'] = 1
        edges_df.loc[(edges_df['uniqueid']==getattr(row, 'uniqueid')), 'fft'] = 36000
    ### output closed file for visualization
    edges_df.loc[edges_df['fft'] == 36000, ['uniqueid', 'start_nid', 'end_nid', 'capacity', 'fft', 'geometry']].to_csv(simulation_outputs + '/closed_links_{}.csv'.format(scen_nm))

    ### nodes processing
    nodes_df = pd.read_csv( network_file_nodes )
    nodes_df['x'] = nodes_df['lon']
    nodes_df['y'] = nodes_df['lat']
    nodes_df = nodes_df.set_index('node_id')

    ### demand processing
    t_od_0 = time.time()
    od_all = pd.read_csv(demand_file)
    t_od_1 = time.time()
    logging.info('{} sec to read {} OD pairs'.format(t_od_1-t_od_0, od_all.shape[0]))
    
    ### run residual_demand_assignment
    assignment(edges_df=edges_df, nodes_df=nodes_df, od_all=od_all, simulation_outputs=simulation_outputs, scen_nm=scen_nm, hour_list=hour_list, quarter_list=quarter_list, closure_hours=closure_hours, closed_links=closed_links)

    return True

if __name__ == "__main__":
    
    status = main(hour_list=list(range(6, 12)), quarter_list=[0,1,2,3], closure_hours=[15,16])
