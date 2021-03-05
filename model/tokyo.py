import sys 
import time 
import logging
import numpy as np
import pandas as pd 
import geopandas as gpd
from shapely.wkt import loads
from residual_demand_assignment import assignment

def main(hour_list=None, quarter_list=None, closure_hours=None, log_suffix = None):

    ### scen_name
    scen_nm = 'time_{}'.format(log_suffix)
    home_dir = '/home/bingyu/Documents/residual_demand'
    work_dir = '/home/bingyu/Documents/residual_demand'
    scratch_dir = '/home/bingyu/Documents/residual_demand'

    ### input files
    network_file_edges = work_dir + '/projects/tokyo_osmnx/network_inputs/tokyo_edges.csv'
    network_file_nodes = work_dir + '/projects/tokyo_osmnx/network_inputs/tokyo_nodes.csv'
    demand_files = [work_dir + "/projects/tokyo_osmnx/demand_inputs/od_0.csv",
                    work_dir + "/projects/tokyo_osmnx/demand_inputs/od_1.csv",
                    work_dir + "/projects/tokyo_osmnx/demand_inputs/od_2.csv"]
    simulation_outputs = scratch_dir + '/projects/tokyo_osmnx/simulation_outputs'

    ### log file*
    logging.basicConfig(filename=simulation_outputs+'/log/{}.log'.format(scen_nm), level=logging.INFO, force=True, filemode='w')

    ### network processing
    edges_df = pd.read_csv( network_file_edges )
    edges_df = gpd.GeoDataFrame(edges_df, crs='epsg:4326', geometry=edges_df['geometry'].map(loads))
    edges_df = edges_df.sort_values(by='fft', ascending=False).drop_duplicates(subset=['start_nid', 'end_nid'], keep='first')
    edges_df['maxspeed'] = np.where(edges_df['type'].isin(['trunk', 'trunk_link']), 70, edges_df['maxspeed'])
    edges_df['fft'] = edges_df['length']/edges_df['maxspeed']*3.6
    edges_df['edge_str'] = edges_df['start_nid'].astype('str') + '-' + edges_df['end_nid'].astype('str')
    edges_df['capacity'] = np.where(edges_df['capacity']<1, 950, edges_df['capacity'])
    edges_df['is_highway'] = np.where(edges_df['type'].isin(['motorway', 'motorway_link']), 1, 0)
    edges_df['normal_capacity'] = edges_df['capacity']
    edges_df['normal_fft'] = edges_df['fft']
    edges_df['t_avg'] = edges_df['fft']
    edges_df = edges_df.set_index('edge_str')
    ### closures
    closed_links = pd.read_csv(work_dir + '/projects/tokyo_osmnx/network_inputs/20160304_closed_links.csv')
    for row in closed_links.itertuples():
        edges_df.loc[(edges_df['u']==getattr(row, 'u')) & (edges_df['v']==getattr(row, 'v')), 'capacity'] = 1
        edges_df.loc[(edges_df['u']==getattr(row, 'u')) & (edges_df['v']==getattr(row, 'v')), 'fft'] = 36000
    print(edges_df.loc[edges_df['fft'] == 36000, ['u', 'v', 'capacity', 'fft']])
    edges_df.loc[edges_df['fft'] == 36000, ['u', 'v', 'capacity', 'fft', 'geometry']].to_csv(simulation_outputs + '/closed_2016.csv')
    # sys.exit(0)
    
    nodes_df = pd.read_csv( network_file_nodes )
    nodes_df['x'] = nodes_df['lon']
    nodes_df['y'] = nodes_df['lat']
    nodes_df = nodes_df.set_index('node_id')

    ### demand processing
    t_od_0 = time.time()
    od_list = []
    for demand_file in demand_files:
        od_chunk = pd.read_csv( demand_file )
        od_list.append(od_chunk)
    od_all = pd.concat(od_list, ignore_index=True)
    od_all['origin_nid'] = od_all['O']
    od_all['destin_nid'] = od_all['D']
    od_all['hour'] = od_all['trip_hour']
    od_all = od_all[['agent_id', 'origin_nid', 'destin_nid', 'hour']]

    # od_all = od_all.iloc[0:10]
    # od_all['hour'] = 5

    t_od_1 = time.time()
    logging.info('{} sec to read {} OD pairs'.format(t_od_1-t_od_0, od_all.shape[0]))
    
    ### run residual_demand_assignment
    assignment(edges_df=edges_df, nodes_df=nodes_df, od_all=od_all, simulation_outputs=simulation_outputs, scen_nm=scen_nm, hour_list=hour_list, quarter_list=quarter_list, closure_hours=closure_hours, closed_links=closed_links)

    return True

if __name__ == "__main__":
    # status = main(hour_list=list(range(5, 10)), quarter_list=[0,1,2,3], closure_hours=[], log_suffix='AM')
    status = main(hour_list=list(range(14, 18)), quarter_list=[0,1,2,3], closure_hours=[15,16], log_suffix='PM_close')
