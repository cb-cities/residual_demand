from __future__ import print_function
import os.path
import sys
import time
import random
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.wkt import loads
import matplotlib.pyplot as plt 

import pandana.network as pdna

### random seed
random.seed(0)
np.random.seed(0)

def substep_assignment(nodes_df=None, weighted_edges_df=None, od_ss=None, quarter_demand=None, assigned_demand=None, quarter_counts=4):

    # print(nodes_df.shape, edges_df.shape)
    # print(len(np.unique(nodes_df.index)), len(np.unique(edges_df.index)))
    # sys.exit(0)
    net = pdna.Network(nodes_df["lon"], nodes_df["lat"], weighted_edges_df["start_igraph"], weighted_edges_df["end_igraph"], weighted_edges_df[["weight"]], twoway=False)
    net.set(pd.Series(net.node_ids))

    nodes_origin = od_ss['node_id_igraph_O'].values
    nodes_destin = od_ss['node_id_igraph_D'].values
    agent_ids = od_ss['agent_id'].values
    paths = net.shortest_paths(nodes_origin, nodes_destin)

    all_path_vol = dict()
    edge_travel_time_dict = weighted_edges_df['t_avg'].T.to_dict()
    od_residual_ss_list = []
    path_i = 0
    for p in paths:
        p_dist = 0
        for edge_s, edge_e in zip(p, p[1:]):
            edge_str = "{}-{}".format(edge_s, edge_e)
            try:
                all_path_vol[edge_str] += 1
            except KeyError:
                all_path_vol[edge_str] = 1
            try:
                p_dist += edge_travel_time_dict[edge_str]
            except KeyError:
                print(edge_str)
                print([pi for pi in p])
                sys.exit(0)
            if p_dist > 3600/quarter_counts:
                od_residual_ss_list.append([agent_ids[path_i], edge_e, p[-1]])
                break
        path_i += 1

    all_path_vol_df = pd.DataFrame.from_dict(all_path_vol, orient='index', columns=['vol_ss'])
    logging.info('   # path {}'.format(path_i))
    # od_residual_ss = pd.DataFrame(od_residual_ss_list, columns=['agent_id', 'node_id_igraph_O', 'node_id_igraph_D'])
    
    new_edges_df = weighted_edges_df[['start_igraph', 'end_igraph', 'fft', 'capacity', 'length', 'is_highway', 'vol_true', 'vol_tot', 'geometry']].copy()
    new_edges_df = new_edges_df.join(all_path_vol_df, how='left')
    new_edges_df['vol_ss'] = new_edges_df['vol_ss'].fillna(0)
    new_edges_df['vol_true'] += new_edges_df['vol_ss']
    new_edges_df['vol_tot'] += new_edges_df['vol_ss']
    new_edges_df['flow'] = (new_edges_df['vol_true']*quarter_demand/assigned_demand)*quarter_counts
    new_edges_df['t_avg'] = new_edges_df['fft'] * ( 1 + 0.6 * (new_edges_df['flow']/new_edges_df['capacity'])**4 ) * 1.2
    new_edges_df['t_avg'] = new_edges_df['t_avg'].round(2)

    return new_edges_df, od_residual_ss_list

def read_od(demand_files=None):
    ### Read the OD table of this time step

    t_od_0 = time.time()

    od_list = []
    for demand_file in demand_files:
        od_chunk = pd.read_csv( demand_file )
        od_list.append(od_chunk)
    
    od_all = pd.concat(od_list, ignore_index=True)
    od_all = od_all[['agent_id', 'node_id_igraph_O', 'node_id_igraph_D', 'hour']]
    # od_all = od_all.iloc[0:10000000]

    t_od_1 = time.time()
    logging.info('{} sec to read {} OD pairs'.format(t_od_1-t_od_0, od_all.shape[0]))
    return od_all

def write_edge_vol(edges_df=None, simulation_outputs=None, quarter=None, hour=None, scen_nm=None):

    if 'flow' in edges_df.columns:
        edges_df.loc[edges_df['vol_true']>0, ['start_igraph', 'end_igraph', 'vol_true', 'flow', 't_avg']].to_csv(simulation_outputs+'/edge_vol/edge_vol_hr{}_qt{}_{}.csv'.format(hour, quarter, scen_nm), index=False)

def assignment(quarter_counts=4, substep_counts=15, substep_size=100000, network_file_nodes=None, network_file_edges=None, demand_files=None, simulation_outputs=None, scen_nm=None, hour_list=None, quarter_list=None, cost_factor=None):

    ### network processing
    edges_df = pd.read_csv( network_file_edges )
    edges_df = gpd.GeoDataFrame(edges_df, crs='epsg:4326', geometry=edges_df['geometry'].map(loads))
    edges_df = edges_df.sort_values(by='fft', ascending=False).drop_duplicates(subset=['start_igraph', 'end_igraph'], keep='first')
    edges_df['edge_str'] = edges_df['start_igraph'].astype('str') + '-' + edges_df['end_igraph'].astype('str')
    edges_df['capacity'] = np.where(edges_df['capacity']<1, 1900, edges_df['capacity'])
    edges_df['is_highway'] = np.where(edges_df['type'].isin(['motorway', 'motorway_link']), 1, 0)
    # edges_df['is_highway'] = np.where(edges_df['type'].isin([1, 2]), 1, 0)
    edges_df = edges_df.set_index('edge_str')

    nodes_df = pd.read_csv( network_file_nodes )
    nodes_df = nodes_df.set_index('node_id_igraph')

    ### OD processing
    od_all = read_od(demand_files=demand_files)
    ### Quarters and substeps
    ### probability of being in each division of hour
    quarter_ps = [1/quarter_counts for i in range(quarter_counts)]
    quarter_ids = [i for i in range(quarter_counts)]

    ### initial setup
    edges_df['t_avg'] = edges_df['fft'] * 1.2
    od_residual_list = []
    ### accumulator
    edges_df['vol_tot'] = 0
    
    ### Loop through days and hours
    for day in ['na']:
        for hour in hour_list:

            ### Read OD
            od_hour = od_all[od_all['hour']==hour].copy()
            if od_hour.shape[0] == 0:
                od_hour = pd.DataFrame([], columns=['agent_id', 'node_id_igraph_O', 'node_id_igraph_D', 'hour'])

            ### Divide into quarters
            od_quarter_msk = np.random.choice(quarter_ids, size=od_hour.shape[0], p=quarter_ps)
            od_hour['quarter'] = od_quarter_msk

            for quarter in quarter_list:

                ### New OD in assignment period
                od_quarter = od_hour[od_hour['quarter']==quarter]
                ### Add resudal OD
                od_residual = pd.DataFrame(od_residual_list, columns=['agent_id', 'node_id_igraph_O', 'node_id_igraph_D'])
                od_residual['quarter'] = quarter
                ### Total OD in each assignment period is the combined of new and residual OD
                od_quarter = pd.concat([od_quarter, od_residual], sort=False, ignore_index=True)
                ### Residual OD is no longer residual after it has been merged to the quarterly OD
                od_residual_list = []
                od_quarter = od_quarter[od_quarter['node_id_igraph_O'] != od_quarter['node_id_igraph_D']]

                quarter_demand = od_quarter.shape[0] ### total demand for this quarter, including total and residual demand
                residual_demand = od_residual.shape[0] ### how many among the OD pairs to be assigned in this quarter are actually residual from previous quarters
                assigned_demand = 0

                substep_counts = (quarter_demand // substep_size) + 1
                logging.info('HR {} QT {} has {}/{} ods/residuals {} substeps'.format(hour, quarter, quarter_demand, residual_demand, substep_counts))
                substep_ps = [1/substep_counts for i in range(substep_counts)] 
                substep_ids = [i for i in range(substep_counts)]
                od_substep_msk = np.random.choice(substep_ids, size=quarter_demand, p=substep_ps)
                od_quarter['ss_id'] = od_substep_msk

                ### reset volume at each quarter
                edges_df['vol_true'] = 0

                for ss_id in substep_ids:

                    time_ss_0 = time.time()
                    od_ss = od_quarter[od_quarter['ss_id']==ss_id]
                    assigned_demand += od_ss.shape[0]
                    if assigned_demand == 0:
                        continue
                    ### calculate weight
                    weighted_edges_df = edges_df.copy()
                    weighted_edges_df['weight'] = edges_df['t_avg'] + cost_factor*edges_df['length']*0.1*(edges_df['is_highway']) ### 10 yen per 100 m --> 0.1 yen per m
                    weighted_edges_df['weight'] = np.where(weighted_edges_df['weight']<0.1, 0.1, weighted_edges_df['weight'])
                    ### traffic assignment with truncated path
                    edges_df, od_residual_ss_list = substep_assignment(nodes_df=nodes_df, weighted_edges_df=weighted_edges_df, od_ss=od_ss, quarter_demand=quarter_demand, assigned_demand=assigned_demand, quarter_counts=quarter_counts)
                    od_residual_list += od_residual_ss_list
                    logging.info('HR {} QT {} SS {} finished, max vol {}, max hwy vol {}, time {}'.format(hour, quarter, ss_id, np.max(edges_df['vol_true']), np.max(edges_df.loc[edges_df['is_highway']==1, 'vol_true']), time.time()-time_ss_0))
                
                ### write quarterly results
                if True: # hour >=16 or (hour==15 and quarter==3):
                    write_edge_vol(edges_df=edges_df, simulation_outputs=simulation_outputs, quarter=quarter, hour=hour, scen_nm=scen_nm)
                    plot_edge_flow(edges_df=edges_df, simulation_outputs=simulation_outputs, quarter=quarter, hour=hour, scen_nm=scen_nm)

def main(hour_list=None, quarter_list=None, scen_nm=None, cost_factor=None):
    ### input files
    network_file_edges = 'edges_residual_demand.csv'
    network_file_nodes = 'nodes_residual_demand.csv'
    demand_files = ["od_residual_demand_0.csv"]
    simulation_outputs = 'simulation_outputs'

    ### log file
    if sys.version_info[1]==8:
        logging.basicConfig(filename=simulation_outputs+'/log/{}.log'.format(scen_nm), level=logging.INFO, force=True)
    elif sys.version_info[1]<8:
        logging.basicConfig(filename=simulation_outputs+'/log/{}.log'.format(scen_nm), level=logging.INFO)
    else:
        print('newer version than 3.8')
    
    ### run residual_demand_assignment
    assignment(network_file_edges=network_file_edges, network_file_nodes=network_file_nodes, demand_files=demand_files, simulation_outputs=simulation_outputs, scen_nm=scen_nm, hour_list=hour_list, quarter_list=quarter_list, cost_factor=cost_factor)

    return True

if __name__ == "__main__":
    status = main(hour_list=list(range(3, 10)), quarter_list=[0,1,2,3], scen_nm='', cost_factor=-2)
    # for cost_factor in [-2, -1, -0.5, 0, 0.5]:
    #     status = main(hour_list=[3,4,5,6,7,8,9,10,11,12], quarter_list=[0,1,2,3], scen_nm='costfct{}'.format(cost_factor), cost_factor=cost_factor)