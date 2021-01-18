import sys 
import logging
import numpy as np
import pandas as pd 
import geopandas as gpd
from shapely.wkt import loads
from residual_demand_assignment import assignment

def main(hour_list=None, quarter_list=None, eq_i=None, building_threshold=None, pipeline_threashold=None, bay_bridge_capacity=None):

    ### scen_name
    demand_scen_nm = 'eq{}_bld{}_pip{}'.format(eq_i, building_threshold, pipeline_threashold)
    supply_scen_nm = 'bb{}'.format(bay_bridge_capacity)
    scen_nm = 'ctpp_{}_{}'.format(demand_scen_nm, supply_scen_nm)
    home_dir = '/home/bingyu/Documents/residual_demand'

    ### input files
    network_file_edges = home_dir + '/projects/bay_area_peer/network_inputs/edges_peer.csv'
    network_file_nodes = home_dir + '/projects/bay_area_peer/network_inputs/nodes_peer.csv'
    demand_files = [home_dir + "/projects/bay_area_peer/demand_inputs/ctpp/od_{}-0.csv".format(demand_scen_nm),
                    home_dir + "/projects/bay_area_peer/demand_inputs/ctpp/od_{}-1.csv".format(demand_scen_nm),
                    home_dir + "/projects/bay_area_peer/demand_inputs/ctpp/od_{}-2.csv".format(demand_scen_nm)]
    simulation_outputs = home_dir + '/projects/bay_area_peer/simulation_outputs'
    ### log file*
    logging.basicConfig(filename=simulation_outputs+'/log/{}.log'.format(scen_nm), level=logging.INFO, force=True, filemode='w')

    ### network processing
    edges_df = pd.read_csv( network_file_edges )
    edges_df = gpd.GeoDataFrame(edges_df, crs='epsg:4326', geometry=edges_df['geometry'].map(loads))
    edges_df = edges_df.sort_values(by='fft', ascending=False).drop_duplicates(subset=['start_nid', 'end_nid'], keep='first')
    edges_df['edge_str'] = edges_df['start_nid'].astype('str') + '-' + edges_df['end_nid'].astype('str')
    edges_df['capacity'] = np.where(edges_df['capacity']<1, 950, edges_df['capacity'])
    edges_df['is_highway'] = np.where(edges_df['type'].isin(['motorway', 'motorway_link']), 1, 0)
    edges_df = edges_df.set_index('edge_str')
    ### closures
    bay_bridge_links = [76239, 285158, 313500, 425877]
    # print(edges_df.loc[edges_df['uniqueid'].isin(bay_bridge_links), 'capacity'])
    for bb_link in bay_bridge_links:
        edges_df.loc[edges_df['uniqueid']==bb_link, 'capacity'] *= bay_bridge_capacity
        edges_df.loc[edges_df['uniqueid']==bb_link, 'capacity'] += 1
        # print(bay_bridge_capacity, bay_bridge_capacity==0)
        if bay_bridge_capacity == 0:
           edges_df.loc[edges_df['uniqueid']==bb_link, 'fft'] = 36000
        #    print(edges_df.loc[edges_df['uniqueid']==bb_link, 'fft'])
    print(edges_df.loc[edges_df['uniqueid'].isin(bay_bridge_links), ['capacity', 'fft']])
    # sys.exit(0)
    
    nodes_df = pd.read_csv( network_file_nodes )
    nodes_df = nodes_df.set_index('node_id')
    
    ### run residual_demand_assignment
    assignment(edges_df=edges_df, nodes_df=nodes_df, demand_files=demand_files, simulation_outputs=simulation_outputs, scen_nm=scen_nm, hour_list=hour_list, quarter_list=quarter_list)

    return True

if __name__ == "__main__":
    for eq_i in [1,2,3,4]:
        status = main(hour_list=list(range(6, 12)), quarter_list=[0,1,2,3], eq_i = eq_i, building_threshold=5, pipeline_threashold=100, bay_bridge_capacity=1)
        # status = main(hour_list=list(range(6, 12)), quarter_list=[0,1,2,3], eq_i = eq_i, building_threshold=5, pipeline_threashold=100, bay_bridge_capacity=0)
        status = main(hour_list=list(range(3, 12)), quarter_list=[0,1,2,3], eq_i = eq_i, building_threshold=4, pipeline_threashold=3, bay_bridge_capacity=0.5)
        status = main(hour_list=list(range(3, 12)), quarter_list=[0,1,2,3], eq_i = eq_i, building_threshold=3, pipeline_threashold=1, bay_bridge_capacity=0)
        # status = main(hour_list=list(range(3, 12)), quarter_list=[0,1,2,3], eq_i = eq_i, building_threshold=3, pipeline_threashold=100)
        # status = main(hour_list=list(range(3, 12)), quarter_list=[0,1,2,3], eq_i = eq_i, building_threshold=5, pipeline_threashold=1)