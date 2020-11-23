import scipy.io as sio
import json
import sys
import numpy as np 
import pandas as pd 
import geopandas as gpd
import os 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import matplotlib.colors as pltcolors
import descartes 
import shapely.wkt 

absolute_path = os.path.dirname(os.path.abspath(__file__))

def main(day, hour, quarter, scen, random_seed):

    ### Get quarterly flow of a particular snapshot of the day
    edge_flow_df = pd.read_csv(absolute_path+'/../2_ABM/hpc_output/edges_df/edges_df_scen{}_r{}_DY{}_HR{}_QT{}.csv'.format(scen, random_seed, day, hour, quarter))

    ### Get attributes and geometry of each edge
    network_attr_df = pd.read_csv(absolute_path+'/../0_network/data/bayarea_osmnx/network_attributes.csv')
    edge_flow_df = pd.merge(edge_flow_df, network_attr_df, on = ['edge_id_igraph'])
    #bridge_volume = edge_flow_df[~pd.isnull(edge_flow_df['bridge']) & (edge_flow_df['true_vol']>0)][['uniqueid', 'bridge', 'true_vol']]
    #print(hour, quarter, bridge_volume.nlargest(8, 'true_vol'))

    edge_flow_df['edge_id_igraph_str'] = edge_flow_df['edge_id_igraph'].astype(str)
    edge_flow_df['voc'] = edge_flow_df['true_vol']*quarter/edge_flow_df['capacity']

    ### Merge results from two directions
    edge_flow_df['undir_uv_igraph'] = pd.DataFrame(np.sort(edge_flow_df[['start_igraph', 'end_igraph']].values, axis=1), columns=['small_igraph', 'large_igraph']).apply(lambda x:'%s_%s' % (x['small_igraph'],x['large_igraph']),axis=1)
    edge_flow_df_grp = edge_flow_df.groupby('undir_uv_igraph').agg({
            'true_vol': np.sum, 
            'tot_vol': np.sum, 
            'voc': np.max,
            'edge_id_igraph_str': lambda x: '-'.join(x),
            'geometry': 'first'}).reset_index()
    edge_flow_df_grp = edge_flow_df_grp.rename(columns={'true_vol': 'undirected_quart_vol', 'tot_vol': 'undirected_tot_vol', 'voc': 'larger_voc'})
    print(hour, quarter, np.max(edge_flow_df_grp['undirected_quart_vol']), np.max(edge_flow_df_grp['undirected_tot_vol']))

    edge_flow_df_grp.to_csv(absolute_path+'/quarterly_traffic/edges_df_scen{}_r{}_DY{}_HR{}_qt{}.csv'.format(scen, random_seed, day, hour, quarter), index=False)

def tot_vol():

    ### Get quarterly flow of a particular snapshot of the day
    edge_flow_df = pd.read_csv(absolute_path
        +'/../2_ABM/output/{}/edges_df_tot_vol.csv'.format(scenario))

    ### Get attributes and geometry of each edge
    network_attr_df = pd.read_csv(absolute_path+'/../0_network/data/{}/network_attributes.csv'.format(folder), index_col=0)
    edge_flow_df = pd.merge(edge_flow_df[['uniqueid', 'tot_vol']], network_attr_df, on = ['uniqueid'])
    edge_flow_df['uniqueid_str'] = edge_flow_df['uniqueid'].astype(str)

    ### Merge results from two directions
    edge_flow_df['undir_uv_igraph'] = pd.DataFrame(np.sort(edge_flow_df[['start', 'end']].values, axis=1), columns=['small_igraph', 'large_igraph']).apply(lambda x:'%s_%s' % (x['small_igraph'],x['large_igraph']),axis=1)
    edge_flow_df_grp = edge_flow_df.sort_values(by=['bridge', 'tunnel'], ascending=False).groupby('undir_uv_igraph').agg({
            'tot_vol': np.sum, 
            'bridge': 'first',
            'tunnel': 'first',
            'uniqueid_str': lambda x: '-'.join(x),
            'geometry': 'first'}).reset_index()
    edge_flow_df_grp = edge_flow_df_grp.rename(columns={'tot_vol': 'undirected_tot_vol'})
    print(np.max(edge_flow_df_grp['undirected_tot_vol']))
    bridge_volume = edge_flow_df_grp[~pd.isnull(edge_flow_df_grp['bridge']) | ~pd.isnull(edge_flow_df_grp['tunnel'])]
    print(bridge_volume.nlargest(8, 'undirected_tot_vol')[['undir_uv_igraph', 'bridge', 'tunnel', 'undirected_tot_vol']])

    edge_flow_df_grp.to_csv(absolute_path+'/quarterly_traffic/{}_edges_df_tot.csv'.format(scenario), index=False)

def compare_damage(day, hour, quarter, random_seed):
    undamaged_df = pd.read_csv(absolute_path+'/quarterly_traffic/edges_df_scenbase_r{}_DY{}_HR{}_qt{}.csv'.format(random_seed, day, hour, quarter))
    damaged_df = pd.read_csv(absolute_path+'/quarterly_traffic/edges_df_scenclosure_r{}_DY{}_HR{}_qt{}.csv'.format(random_seed, day, hour, quarter))
    diff = pd.merge(undamaged_df, damaged_df[['undir_uv_igraph', 'undirected_quart_vol', 'undirected_tot_vol']], on='undir_uv_igraph', suffixes=['_undam', '_dam'])
    diff['diff_quart'] = diff['undirected_quart_vol_dam'] - diff['undirected_quart_vol_undam']
    diff['diff_tot'] = diff['undirected_tot_vol_dam'] - diff['undirected_tot_vol_undam']
    print(diff.sort_values(by='diff_quart', ascending=False)[['undir_uv_igraph', 'diff_quart']].head())

    diff.to_csv(absolute_path+'/quarterly_traffic/diff_r{}_DY{}_HR{}_qt{}.csv'.format(random_seed, day, hour, quarter), index=False)


if __name__ == '__main__':

    main('na', 6, 1, 'closure', 0)
    # for day in ['na']:
    #     for hour in [0,1,2,3]:
    #         for quarter in [0,1,2,3]:
    #             for residual in [True]:
    #                 for random_seed in [0]:
    #                     main(day, hour, quarter, residual, random_seed)

    #tot_vol()
    compare_damage('na', 6, 1, 0)



