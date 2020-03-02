### Based on https://mikecvet.wordpress.com/2010/07/02/parallel-mapreduce-in-python/
import json
import sys
import numpy as np
import scipy.sparse as ssparse
import scipy.io as sio
import multiprocessing
from multiprocessing import Pool 
import time 
import os
import datetime
import warnings
import pandas as pd 
from ctypes import *
import gc 
from heapq import nlargest
import shapely.wkt

pd.set_option('display.max_columns', 10)

absolute_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, absolute_path+'/../')
sys.path.insert(0, '/Users/bz247/')
from sp import interface 

def map_edge_flow_residual(arg):
    ### Find shortest path for each unique origin --> one destination
    ### In the future change to multiple destinations
    row = arg[0]
    quarter_counts = arg[1]

    agent_id = int(OD_ss['agent_id'].iloc[row])
    origin_ID = int(OD_ss['origin_sp'].iloc[row])
    destin_ID = int(OD_ss['destin_sp'].iloc[row])

    sp = g.dijkstra(origin_ID, destin_ID) ### g_0 is the network with imperfect information for route planning
    sp_dist = sp.distance(destin_ID) ### agent believed travel time with imperfect information
    if sp_dist > 10e7:
        # print(agent_id, sp_dist)
        return {'agent_id': agent_id, 'o_sp': origin_ID, 'd_sp': destin_ID, 'route': pd.DataFrame([], columns=['start_sp', 'end_sp']), 'arr': 'n'} ### empty path; not reach destination; travel time 0
    else:
        sp_route = sp.route(destin_ID) ### agent route planned with imperfect information

        sp_route_df = pd.DataFrame([edge for edge in sp_route], columns=['start_sp', 'end_sp'])
        #sp_route_df.insert(0, 'seq_id', range(sp_route_df.shape[0]))
        sub_edges_df = edges_df[(edges_df['start_sp'].isin(sp_route_df['start_sp'])) & (edges_df['end_sp'].isin(sp_route_df['end_sp']))]

        #sp_route_df = pd.merge(sp_route_df, sub_edges_df[['previous_t']], how='left')
        sp_route_df = sp_route_df.merge(sub_edges_df[['start_sp', 'end_sp', 'previous_t', 'true_vol']], on=['start_sp', 'end_sp'], how='left')
        sp_route_df['timestamp'] = sp_route_df['previous_t'].cumsum()

        trunc_sp_route_df = sp_route_df[sp_route_df['timestamp']<(3600/quarter_counts)]
        #stop_node = trunc_sp_route_df.iloc[-1]['end_sp']
        try:
            stop_node = trunc_sp_route_df.iloc[-1]['end_sp']
            travel_time = trunc_sp_route_df.iloc[-1]['timestamp']
        except IndexError:
            ### cannot pass even the first link
            #print(sp_route_df.iloc[0][['start_sp', 'end_sp', 'timestamp', 'true_vol','previous_t']])
            link_time = sp_route_df.iloc[0]['timestamp']
            stop_node = np.random.choice([sp_route_df.iloc[0]['start_sp'], sp_route_df.iloc[0]['end_sp']], p=[1-(3600/quarter_counts)/link_time, (3600/quarter_counts)/link_time])
            travel_time = 3600/quarter_counts
        trunc_edges = trunc_sp_route_df[['start_sp', 'end_sp']]

        results = {'agent_id': agent_id, 'o_sp': origin_ID, 'd_sp': destin_ID, 'h_sp': stop_node, 'travel_time': travel_time, 'route': trunc_edges, 'arr': 'a'}
        ### [(edge[0], edge[1]) for edge in sp_route]: agent's choice of route
        return results


def reduce_edge_flow_pd(agent_info_routes_found, day, hour, quarter, ss_id):
    ### Reduce (count the total traffic flow per edge) with pandas groupby

    t0 = time.time()
    # flat_L = [(e[0], e[1], r['vol'], r['probe']) for r in agent_info_routes for e in zip(r['route'], r['route'][1:])]
    # df_L = pd.DataFrame(flat_L, columns=['start_sp', 'end_sp', 'vol', 'probe'])
    df_L = pd.concat([r['route'] for r in agent_info_routes_found])
    df_L_flow = df_L.groupby(['start_sp', 'end_sp']).size().reset_index().rename(columns={0: 'ss_vol'}) # link_flow counts the number of vehicles, link_probe counts the number of probe vehicles
    t1 = time.time()
    # print('DY{}_HR{}_QT{} SS {}: reduce find {} edges, {} sec w/ pd.groupby, max substep volume {}'.format(day, hour, quarter, ss_id, df_L_flow.shape[0], t1-t0, np.max(df_L_flow['ss_vol'])))
    
    return df_L_flow

def map_reduce_edge_flow(day, hour, quarter, ss_id, quarter_counts):
    ### One time step of ABM simulation
    
    t_odsp_0 = time.time()

    ### skip when no OD calculation is required
    unique_origin = OD_ss.shape[0]
    if unique_origin == 0:
        return pd.DataFrame([], columns=['start_sp', 'end_sp', 'ss_vol']), [], [], 0

    ### Build a pool
    process_count = 7
    pool = Pool(processes=process_count, maxtasksperchild=1000)

    ### Find shortest pathes
    res = pool.imap_unordered(map_edge_flow_residual, [(i, quarter_counts) for i in range(unique_origin)])

    ### Close the pool
    pool.close()
    pool.join()
    # if len(multiprocessing.active_children())>0:
    #     print(ss_id, len(multiprocessing.active_children()), OD_ss.shape, unique_origin)
    #     sys.exit(0)
    t_odsp_1 = time.time()

    ### Organize results
    agent_info_routes = list(res)
    agent_info_routes_found = [a for a in agent_info_routes if a['arr']=='a']
    agent_info_routes_notfound = [a for a in agent_info_routes if a['arr']=='n']

    edge_volume = reduce_edge_flow_pd(agent_info_routes_found, day, hour, quarter, ss_id)
    # try:
    #     edge_volume = reduce_edge_flow_pd(agent_info_routes, day, hour, quarter, ss_id)
    # except TypeError:
    #     return pd.DataFrame([], columns=['start_sp', 'end_sp', 'ss_vol']), [], [], 0
    
    ss_residual_OD_list = [(r['agent_id'], r['h_sp'], r['d_sp']) for r in agent_info_routes_found if r['h_sp']!=r['d_sp']]
    #ss_travel_time_list = [(r['agent_id'], day, hour, quarter, ss_id, r['travel_time']) for r in agent_info_routes]
    ss_travel_time_list = []
    # print('ss {}, total od {}, found {}, not found {}'.format(ss_id, unique_origin, len(agent_info_routes_found), len(agent_info_routes_notfound)))
    # print('DY{}_HR{}_QT{} SS {}: {} O --> {} D found, dijkstra pool {} sec on {} processes'.format(day, hour, quarter, ss_id, unique_origin, len(agent_info_routes_found), t_odsp_1 - t_odsp_0, process_count))

    return edge_volume, ss_residual_OD_list, ss_travel_time_list, len(agent_info_routes_notfound)

def update_graph(edge_volume, edges_df, day, hour, quarter, ss_id, quarter_demand, assigned_demand, quarter_counts):
    ### Update graph

    t_update_0 = time.time()

    ### first update the cumulative link volume in the current time step
    edges_df = pd.merge(edges_df, edge_volume, how='left', on=['start_sp', 'end_sp'])
    edges_df = edges_df.fillna(value={'ss_vol': 0}) ### fill volume for unused edges as 0
    edges_df['true_vol'] += edges_df['ss_vol'] ### update the total volume (newly assigned + carry over)
    edges_df['tot_vol'] += edges_df['ss_vol'] ### tot_vol is not reset to 0 at each time step

    ### True flux
    edges_df['true_flow'] = (edges_df['true_vol']*quarter_demand/assigned_demand)*quarter_counts ### divided by 0.25 h to get the hourly flow.
    #edges_df['true_flow'] = (edges_df['true_vol'])/0.25 

    edges_df['t_avg'] = edges_df['fft']*(1 + 0.6*(edges_df['true_flow']/edges_df['capacity'])**4)*1.2

    update_df = edges_df.loc[edges_df['t_avg'] != edges_df['previous_t']].copy().reset_index()

    if update_df.shape[0] == 0:
        pass
    else:
        for row in update_df.itertuples():
            g.update_edge(getattr(row,'start_sp'), getattr(row,'end_sp'), c_double(getattr(row,'t_avg')))

    edges_df['previous_t'] = edges_df['t_avg']
    edges_df = edges_df.drop(columns=['ss_vol'])

    ### test damage
    #test_sp = g.dijkstra(112661,94033)
    #print(test_sp.distance(94033), [edge for edge in test_sp.route(94033)])

    t_update_1 = time.time()

    return edges_df

def read_OD(nodes_df=None, project_folder=None, chunk=False):
    ### Read the OD table of this time step

    t_OD_0 = time.time()

    ### Change OD list from using osmid to sequential id. It is easier to find the shortest path based on sequential index.
    if chunk:
        od_list = []
        for chunk_num in range(3):
            sub_od = pd.read_csv(absolute_path+'{}/od_residual_demand_{}.csv'.format(project_folder, chunk_num))
            od_list.append(sub_od)
        OD = pd.concat(od_list, ignore_index=True)
    else:
        OD = pd.read_csv(absolute_path+'{}/od_residual_demand.csv'.format(project_folder))
    OD['origin_sp'] = OD['node_id_igraph_O'] + 1 ### the node id in module sp is 1 higher than igraph id
    OD['destin_sp'] = OD['node_id_igraph_D'] + 1
    if 'agent_id' not in OD.columns:
        OD['agent_id'] = np.arange(OD.shape[0])
    if 'hour' not in OD.columns:
        OD['hour'] = 0
    OD = OD[['agent_id', 'origin_sp', 'destin_sp', 'hour']]
    OD = OD.iloc[0:1000000]

    t_OD_1 = time.time()
    print('{} sec to read {} OD pairs'.format(t_OD_1-t_OD_0, OD.shape[0]))

    return OD

def output_edges_df(edges_df, day, hour, quarter, random_seed=None, scen_id=None, project_folder=None):

    ### Aggregate and calculate link-level variables after all increments
    edges_df[['edge_id_igraph', 'type', 'tot_vol', 'true_vol', 't_avg']].to_csv(absolute_path+'{}/outputs/edges_df/edges_df_scen{}_r{}_DY{}_HR{}_QT{}.csv'.format(project_folder, scen_id, random_seed, day, hour, quarter), index=False)

def sta(random_seed=0, quarter_counts=4, scen_id='base', damage_df=None, project_folder=None, od_chunk=False):

    t_main_0 = time.time()
    ### Fix random seed
    np.random.seed(random_seed)
    ### Define global variables to be shared with subprocesses
    global g ### weighted graph
    global OD_ss ### substep demand
    global edges_df ### link weights

    ### Read in the edge attribute for volume delay calculation later
    edges_df0 = pd.read_csv(absolute_path+'{}/edges_residual_demand.csv'.format(project_folder))
    nodes_df = pd.read_csv(absolute_path+'{}/nodes_residual_demand.csv'.format(project_folder))
    
    ### damage
    if damage_df is None:
        pass
    else:
        edges_df0.loc[edges_df0['edge_id_igraph'].isin(damage_df['edge_id_igraph']), 'fft'] = 10e7

    node_count = max(len(np.unique(edges_df0['start_igraph'])), len(np.unique(edges_df0['end_igraph'])))
    g_coo = ssparse.coo_matrix((edges_df0['fft']*1.2, (edges_df0['start_igraph'], edges_df0['end_igraph'])), shape=(node_count, node_count))
    edges_df0 = edges_df0[['edge_id_igraph', 'start_sp', 'end_sp', 'length', 'capacity', 'fft', 'type']]
    sio.mmwrite(absolute_path+'{}/network_sparse_scen{}_r{}.mtx'.format(project_folder, scen_id, random_seed), g_coo)

    all_OD = read_OD(nodes_df = nodes_df, project_folder=project_folder, chunk=od_chunk)

    ### Define quarter and substep parameters
    quarter_ps = [1/quarter_counts for i in range(quarter_counts)] ### probability of being in each division of hour
    quarter_ids = [i for i in range(quarter_counts)]
    
    substep_counts = 15
    substep_ps = [1/substep_counts for i in range(substep_counts)] ### probability of being in each substep
    substep_ids = [i for i in range(substep_counts)]
    print('{} quarters per hour, {} substeps'.format(quarter_counts, substep_counts))

    sta_stats = []
    residual_OD_list = []
    travel_time_list = []

    ### Loop through days and hours
    for day in ['na']:

        ### Read in the initial network (free flow travel time
        g = interface.readgraph(bytes(absolute_path+'{}/network_sparse_scen{}_r{}.mtx'.format(project_folder, scen_id, random_seed), encoding='utf-8'))
        ### test damage
        # test_sp = g.dijkstra(31269, 131637)
        # print(test_sp.distance(131637), [edge for edge in test_sp.route(131637)])
        # sys.exit(0)

        ### Variables reset at the beginning of each day
        edges_df = edges_df0.copy() ### length, capacity and fft that should never change in one simulation
        edges_df['previous_t'] = edges_df['fft'] ### Used to find which edge to update. At the beginning of each day, previous_t is the free flow time.
        edges_df['tot_vol'] = 0
        tot_non_arrival = 0

        for hour in range(3,4):

            t_hour_0 = time.time()

            ### Read OD
            OD = all_OD[all_OD['hour']==hour].copy()
            if OD.shape[0] == 0:
                OD = pd.DataFrame([], columns=['agent_id', 'origin_sp', 'destin_sp'])

            ### Divide into quarters
            OD_quarter_msk = np.random.choice(quarter_ids, size=OD.shape[0], p=quarter_ps)
            OD['quarter'] = OD_quarter_msk

            for quarter in range(quarter_counts):

                ### New OD in assignment period
                OD_quarter = OD[OD['quarter']==quarter]
                ### Add resudal OD
                OD_residual = pd.DataFrame(residual_OD_list, columns=['agent_id', 'origin_sp', 'destin_sp'])
                OD_residual['quarter'] = quarter
                ### Total OD in each assignment period is the combined of new and residual OD
                OD_quarter = pd.concat([OD_quarter, OD_residual], sort=False, ignore_index=True)
                ### Residual OD is no longer residual after it has been merged to the quarterly OD
                residual_OD_list = []
                OD_quarter = OD_quarter[OD_quarter['origin_sp'] != OD_quarter['destin_sp']]
                
                quarter_demand = OD_quarter.shape[0] ### total demand for this quarter, including total and residual demand
                residual_demand = OD_residual.shape[0] ### how many among the OD pairs to be assigned in this quarter are actually residual from previous quarters
                assigned_demand = 0

                print('DY {}, HR {}, QT {}, quarter_demand {}'.format(day, hour, quarter, quarter_demand))

                if quarter_demand == 0:
                    continue
                else:
                    OD_quarter = OD_quarter.sample(frac=1).reset_index(drop=True)

                OD_substep_msk = np.random.choice(substep_ids, size=quarter_demand, p=substep_ps)
                OD_quarter['ss_id'] = OD_substep_msk

                ### Reset some variables at the beginning of each time step
                edges_df['true_vol'] = 0

                for ss_id in substep_ids:

                    t_substep_0 = time.time()

                    ### Get the substep demand
                    OD_ss = OD_quarter[OD_quarter['ss_id'] == ss_id]
                    assigned_demand += OD_ss.shape[0]

                    if assigned_demand == 0: 
                        ### assigned_demand could appear as denominator
                        pass
                    else:
                        ### Routing for this substep (map reduce)
                        edge_volume, ss_residual_OD_list, ss_travel_time_list, ss_non_arrival = map_reduce_edge_flow(day, hour, quarter, ss_id, quarter_counts)
                        tot_non_arrival += ss_non_arrival
                        residual_OD_list += ss_residual_OD_list
                        # travel_time_list += ss_travel_time_list

                        ### Updating
                        edges_df = update_graph(edge_volume, edges_df, day, hour, quarter, ss_id, quarter_demand, assigned_demand, quarter_counts)

                    t_substep_1 = time.time()
                    print('DY{}_HR{} SS {}: {} sec, {} OD pairs'.format(day, hour, ss_id, t_substep_1-t_substep_0, OD_ss.shape[0], ))

                output_edges_df(edges_df, day, hour, quarter, random_seed=random_seed, scen_id=scen_id, project_folder=project_folder)

                ### Update carry over flow
                sta_stats.append([
                    random_seed, day, hour, quarter, quarter_demand, residual_demand, len(residual_OD_list),
                    np.sum(edges_df['t_avg']*edges_df['true_vol']/(quarter_demand*60)),
                    np.sum(edges_df['length']*edges_df['true_vol']/(quarter_demand*1000)),
                    np.mean(edges_df.nlargest(10, 'true_vol')['true_vol'])
                    ])
                
                ### Travel time by road category
                edges_df['tot_t_hr{}_qt{}'.format(hour, quarter)] = edges_df['true_vol'] * edges_df['t_avg']

                t_hour_1 = time.time()
                ### log hour results before resetting the flow for the next time step
                print('DY{}_HR{}_QT{}: {} sec, OD {}, {} residual'.format(day, hour, quarter, round(t_hour_1-t_hour_0, 3), quarter_demand, len(residual_OD_list)))
                gc.collect()

    #output_edges_df(edges_df, day, hour, quarter, random_seed, True)
    print('total non arrival {}'.format(tot_non_arrival))
    
    t_main_1 = time.time()
    print('total run time: {} sec \n\n\n\n\n'.format(t_main_1 - t_main_0))
    return sta_stats, travel_time_list, all_OD.shape[0]

def main(random_seed=0, scen_id='base', damage_df=None, quarter_counts=4, project_folder='', od_chunk=False):

    # random_seed = 0#int(os.environ['RANDOM_SEED'])
    # scen_id = 2#int(os.environ['SCEN_ID'])
    print('random_seed', random_seed, 'scen_id', scen_id)

    ### carry out sta/semi-dynamic assignment
    sta_results = sta(random_seed=random_seed, quarter_counts=quarter_counts, scen_id=scen_id, damage_df=damage_df, project_folder=project_folder, od_chunk=od_chunk)
    sta_stats = sta_results[0]
    travel_time_list = sta_results[1]
    total_od_counts = sta_results[2]

    ### origanize results
    sta_stats_df = pd.DataFrame(sta_stats, columns=['random_seed', 'day', 'hour', 'quarter', 'quarter_demand', 'residual_demand', 'residual_demand_produced', 'avg_veh_min', 'avg_veh_km', 'avg_top10_vol'])
    # sta_stats_df.to_csv(absolute_path+'/output/stats/stats_scen{}_{}_rec{}_r{}_od{}.csv'.format(scen_id, road_type, recovery_period, random_seed, total_od_counts), index=False)
    print('total travel hours', np.sum(sta_stats_df['quarter_demand']*sta_stats_df['avg_veh_min'])/60)
    print('total travel km', np.sum(sta_stats_df['quarter_demand']*sta_stats_df['avg_veh_km']))


if __name__ == '__main__':
    project_folder = '/projects/tokyo_residential_above'
    main(random_seed=0, scen_id='base', damage_df=None, quarter_counts=4, project_folder=project_folder, od_chunk=True)

