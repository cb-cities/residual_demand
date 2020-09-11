import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as ssparse
from ctypes import *
import scipy.io as sio
import subprocess

sys.path.insert(0, '/home/bingyu/Documents')
from sp import interface

sys.path.insert(0, '/home/bingyu/Documents/residual_demand')
import residual_demand_pandana

import haversine

def process_observations(links_df0):

    ### read validation data: traffic flow by quarter
    quarterly_measures = pd.read_csv('quarterly_measure.csv')

    ### get observation groups
    group_measures = quarterly_measures.groupby('obs_grp_id').agg({'start': 'first', 'end': 'first', 'dir': 'first'}).reset_index()

    ### this network prioritize motorway and motorway link
    # g = interface.readgraph(bytes('network_sparse_mex_analysis.mtx', encoding='utf-8'))
    g = interface.from_dataframe(links_df0, 'start_igraph', 'end_igraph', 'fft')

    ### find the node id in graph
    nodes_df = pd.read_csv('../network_inputs/nodes_residual_demand.csv')
    group_measures = group_measures.merge(nodes_df[['node_osmid', 'node_id_igraph', 'lon', 'lat']], how='left', left_on='start', right_on='node_osmid')
    group_measures = group_measures.merge(nodes_df[['node_osmid', 'node_id_igraph', 'lon', 'lat']], how='left', left_on='end', right_on='node_osmid', suffixes=['_start', '_end'])
    group_measures = group_measures.dropna(subset=['node_id_igraph_start', 'node_id_igraph_end'])
    group_measures['start_igraph'] = group_measures['node_id_igraph_start'].astype(int)
    group_measures['end_igraph'] = group_measures['node_id_igraph_end'].astype(int)

    tokyo_center_lat, tokyo_center_lon = 35.6843052, 139.7442715
    group_measures['distance_start'] = haversine.haversine(group_measures['lat_start'], group_measures['lon_start'], tokyo_center_lat, tokyo_center_lon)
    group_measures['distance_end'] = haversine.haversine(group_measures['lat_end'], group_measures['lon_end'], tokyo_center_lat, tokyo_center_lon)
    group_measures['distance'] = (group_measures['distance_start']+group_measures['distance_end'])/2
    group_measures['distance_weight'] = np.interp(group_measures['distance'], [np.min(group_measures['distance']), np.max(group_measures['distance'])],[1,0.1])

    ### build edge-obs_grp dataframe
    obs_grp_edge_list = []
    for row in group_measures.itertuples():
        obs_grp_id = getattr(row, 'obs_grp_id')
        distance_weight = getattr(row, 'distance_weight')
        obs_grp_dir = getattr(row, 'dir')
        if obs_grp_id in [110, 116]:
            continue
        elif obs_grp_id in range(43,88):
            if obs_grp_dir == 2:
                start_igraph = getattr(row, 'end_igraph')
                end_igraph = getattr(row, 'start_igraph')
            elif obs_grp_dir == 1:
                start_igraph = getattr(row, 'start_igraph')
                end_igraph = getattr(row, 'end_igraph')
            else:
                print('invalid direction')
                print(obs_grp_dir)
        else:
            if obs_grp_dir == 1:
                start_igraph = getattr(row, 'end_igraph')
                end_igraph = getattr(row, 'start_igraph')
            elif obs_grp_dir == 2:
                start_igraph = getattr(row, 'start_igraph')
                end_igraph = getattr(row, 'end_igraph')
            else:
                print('invalid direction')
                print(obs_grp_dir)

        try:
            sp = g.dijkstra(start_igraph, end_igraph)
        except ArgumentError:
            print(end_igraph)
        sp_dist = sp.distance(end_igraph)
        if sp_dist > 10e7:
            print('route not found')
            sp.clear()
        else:
            sp_route = sp.route(end_igraph)
            route_igraph = [(start, end) for (start, end) in sp_route]
            if len(route_igraph)>20:
                pass
            else:
                obs_grp_edge_list += [(start, end, obs_grp_id, distance_weight) for (start, end) in route_igraph]
            sp.clear()

    obs_grp_edge_df = pd.DataFrame(obs_grp_edge_list, columns=['start_igraph', 'end_igraph', 'obs_grp_id', 'distance_weight'])
    # obs_grp_edge_df = pd.merge(obs_grp_edge_df, links_df0[['start_igraph', 'end_igraph', 'length']], how='left', on=['start_igraph', 'end_igraph'])
    obs_grp_edge_df['edge_str'] = obs_grp_edge_df['start_igraph'].astype('str') + '-' + obs_grp_edge_df['end_igraph'].astype('str')
    # print(obs_grp_edge_df[obs_grp_edge_df['edge_str'].isin('91591-91599', '91599-405199', '182588-91604', '405199-182588', '197987-642902')])

    obs_grp_geom_df = pd.merge(obs_grp_edge_df, links_df0[['start_igraph', 'end_igraph', 'geometry']], how='left', on=['start_igraph', 'end_igraph'])
    return quarterly_measures, obs_grp_edge_df, obs_grp_geom_df

def main(one_quarter_hour =None):

    ### read edges_df file
    links_df0 = pd.read_csv('../network_inputs/edges_residual_demand.csv')
    links_df0['fft'] = links_df0['length']/links_df0['maxmph']*2.237

    ### read observation file
    quarterly_measures, obs_grp_edge_df, obs_grp_geom_df = process_observations(links_df0)

    ### daily/quarterly measures
    # daily_measures = quarterly_measures.groupby('obs_grp_id').agg({'Q': np.sum}).reset_index()
    # quarterly_measures = quarterly_measures[quarterly_measures['start_quarter']==one_quarter_hour *4].reset_index()
    hourly_measures = quarterly_measures.copy()
    hourly_measures['hour'] = quarterly_measures['start_quarter']//4
    hourly_measures = hourly_measures.groupby(['obs_grp_id', 'hour']).agg({'Q': np.sum}).reset_index()
    hourly_measures = hourly_measures[hourly_measures['hour']==one_quarter_hour].reset_index(drop=True)
    print('hourly measures ', hourly_measures.shape, len(hourly_measures['obs_grp_id'].unique()))

    for cost_factor in [-2, -1, -0.5, 0, 0.5]:
  
        ## run simulation
        # status = residual_demand_pandana.main(hour_list=[one_quarter_hour], quarter_list=[0], scen_nm='costfct{}'.format(cost_factor), cost_factor=cost_factor)

        ### process simulationr results
        hourly_sim_vol = obs_grp_edge_df.copy()
        hourly_sim_vol = hourly_sim_vol.drop_duplicates(subset='edge_str')
        hourly_sim_vol['Q_sim'] = 0
        for hour in [one_quarter_hour]:
            for quarter in [0,1,2,3]:
                quarter_sim_vol_res = pd.read_csv('../simulation_outputs/edge_vol/edge_vol_hr{}_qt{}_costfct{}.csv'.format(hour, quarter, cost_factor))
                quarter_sim_vol_res['edge_str'] = quarter_sim_vol_res['start_igraph'].astype('str') + '-' + quarter_sim_vol_res['end_igraph'].astype('str')
                quarter_sim_vol_res = quarter_sim_vol_res.loc[quarter_sim_vol_res['edge_str'].isin(hourly_sim_vol['edge_str'])]
                hourly_sim_vol = pd.merge(hourly_sim_vol[['obs_grp_id', 'edge_str', 'Q_sim', 'distance_weight']], quarter_sim_vol_res[['edge_str', 'vol_true']], how='left', on='edge_str')
                hourly_sim_vol['Q_sim'] += hourly_sim_vol.fillna(value={'vol_true': 0})['vol_true']
                hourly_sim_vol = hourly_sim_vol[['obs_grp_id', 'edge_str', 'Q_sim', 'distance_weight']]
        # daily_sim_vol.sort_values(by='daily_vol', ascending=False).head()
        hourly_sim_vol = hourly_sim_vol.groupby('obs_grp_id').agg({'Q_sim': np.mean, 'distance_weight': np.mean}).reset_index()
        print('hourly sim ', hourly_sim_vol.shape[0], len(hourly_sim_vol['obs_grp_id'].unique()))

        ### compare
        compare_df = pd.merge(hourly_measures, hourly_sim_vol, how='left', on='obs_grp_id')
        print(obs_grp_edge_df.shape, compare_df.shape)
        compare_df = compare_df.dropna(subset=['Q', 'distance_weight'])
        compare_df['Q_sim'] = compare_df['Q_sim'].fillna(0)
        compare_df['diff'] = compare_df['Q_sim'] - compare_df['Q']
        compare_df['weighted_diff'] = (compare_df['Q_sim'] - compare_df['Q'])*compare_df['distance_weight']
        print(compare_df[['Q', 'Q_sim', 'diff', 'weighted_diff']].describe())
        
        obs_grp_geom_df = obs_grp_geom_df[['obs_grp_id', 'start_igraph', 'end_igraph', 'geometry']].merge(compare_df, how='left', on='obs_grp_id')
        obs_grp_geom_df.to_csv('viz/q_diff_hr{}_costfct{}.csv'.format(one_quarter_hour, cost_factor), index=False)

        with open('parameter_sensitivity_hourly.csv', 'a') as outfile:
            outfile.write("{},{},".format(hour, cost_factor) 
                        + "{0:.2f},{1:.2f},".format(np.mean(compare_df['Q']), np.median(compare_df['Q']))
                        + "{0:.2f},{1:.2f},".format(np.mean(compare_df['Q_sim']), np.median(compare_df['Q_sim']))
                        + "{0:.2f},{1:.2f},".format(np.mean(compare_df['diff']), np.median(compare_df['diff']))
                        + "{0:.2f},{1:.2f},".format(np.percentile(compare_df['diff'], 25), np.percentile(compare_df['diff'], 75))
                        + "{0:.2f},{1:.2f},".format(np.mean(compare_df['weighted_diff']), np.median(compare_df['weighted_diff']))
                        + "{0:.2f},{1:.2f}\n".format(np.percentile(compare_df['weighted_diff'], 25), np.percentile(compare_df['weighted_diff'], 75))
                        )

if __name__ == "__main__":
    with open('parameter_sensitivity_hourly.csv', 'w') as outfile:
        outfile.write("hour,costfct,Q_obs_mean,Q_obs_med,Q_sim_mean,Q_sim_med,Q_diff_mean,Q_diff_med,Q_diff_25,Q_diff_75,Q_wdiff_mean,Q_wdiff_med,Q_wdiff_25,Q_wdiff_75\n")
    for one_quarter_hour in [3,4,5,6,7,8,9,10,11,12]:
        main(one_quarter_hour = one_quarter_hour)