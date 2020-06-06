### Based on https://mikecvet.wordpress.com/2010/07/02/parallel-mapreduce-in-python/
import os
import gc 
import sys
import time 
import random
import logging
import numpy as np
import pandas as pd 
from ctypes import *
import scipy.io as sio
from heapq import nlargest
import scipy.sparse as ssparse
from multiprocessing import Pool 

from line_profiler import LineProfiler

### dir
home_dir = os.environ['HOME']+'/residual_demand'
work_dir = os.environ['WORK']+'/residual_demand'
scratch_dir = os.environ['SCRATCH']+'/residual_demand'
### user library
sys.path.insert(0, home_dir+'/..')
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
        sp.clear()
        return {'agent_id': agent_id, 'o_sp': origin_ID, 'd_sp': destin_ID, 'route': pd.DataFrame([], columns=['start_sp', 'end_sp']), 'arr': 'n'} ### empty path; not reach destination; travel time 0
    else:
        sp_route = sp.route(destin_ID) ### agent route planned with imperfect information
        try:
            sp_route_df = pd.DataFrame([edge for edge in sp_route], columns=['start_sp', 'end_sp'])
        except RecursionError:
            print(origin_ID, destin_ID, sp_dist)
        sp.clear()
        
        try:
            sub_edges_df = edges_df[(edges_df['start_sp'].isin(sp_route_df['start_sp'])) & (edges_df['end_sp'].isin(sp_route_df['end_sp']))]
            print(0, agent_id, origin_ID, destin_ID, 
            edges_df.shape, max(edges_df.index), min(edges_df.index), max(edges_df['start_sp']), min(edges_df['start_sp']), max(edges_df['end_sp']), min(edges_df['end_sp']), 
            sp_route_df.shape, max(sp_route_df.index), min(sp_route_df.index), max(sp_route_df['start_sp']), min(sp_route_df['start_sp']), max(sp_route_df['end_sp']), min(sp_route_df['end_sp']))
        except IndexError:
            pd.set_option('display.max_rows', 100)
            print(1, agent_id, origin_ID, destin_ID, 
            edges_df.shape, max(edges_df.index), min(edges_df.index), max(edges_df['start_sp']), min(edges_df['start_sp']), max(edges_df['end_sp']), min(edges_df['end_sp']), 
            sp_route_df.shape, max(sp_route_df.index), min(sp_route_df.index), max(sp_route_df['start_sp']), min(sp_route_df['start_sp']), max(sp_route_df['end_sp']), min(sp_route_df['end_sp']))
        if sub_edges_df.shape[0]==0:
            print(2, agent_id, origin_ID, destin_ID, 
            edges_df.shape, max(edges_df.index), min(edges_df.index), max(edges_df['start_sp']), min(edges_df['start_sp']), max(edges_df['end_sp']), min(edges_df['end_sp']), 
            sp_route_df.shape, max(sp_route_df.index), min(sp_route_df.index), max(sp_route_df['start_sp']), min(sp_route_df['start_sp']), max(sp_route_df['end_sp']), min(sp_route_df['end_sp']))
        sp_route_df = sp_route_df.merge(sub_edges_df[['start_sp', 'end_sp', 'previous_t', 'true_vol']], on=['start_sp', 'end_sp'], how='left')
        sp_route_df['timestamp'] = sp_route_df['previous_t'].cumsum()

        trunc_sp_route_df = sp_route_df[sp_route_df['timestamp']<(3600/quarter_counts)]
        try:
            stop_node = trunc_sp_route_df.iloc[-1]['end_sp']
            travel_time = trunc_sp_route_df.iloc[-1]['timestamp']
        except IndexError:
            ### cannot pass even the first link
            link_time = sp_route_df.iloc[0]['timestamp']
            stop_node = np.random.choice([sp_route_df.iloc[0]['start_sp'], sp_route_df.iloc[0]['end_sp']], p=[1-(3600/quarter_counts)/link_time, (3600/quarter_counts)/link_time])
            travel_time = 3600/quarter_counts
        trunc_edges = trunc_sp_route_df[['start_sp', 'end_sp']]

        return {'agent_id': agent_id, 'o_sp': origin_ID, 'd_sp': destin_ID, 'h_sp': stop_node, 'travel_time': travel_time, 'route': trunc_edges, 'arr': 'a'}

def reduce_edge_flow_pd(agent_info_routes_found, day, hour, quarter, ss_id):
    ### Reduce (count the total traffic flow per edge) with pandas groupby

    t0 = time.time()
    df_L = pd.concat([r['route'] for r in agent_info_routes_found])
    df_L_flow = df_L.groupby(['start_sp', 'end_sp']).size().reset_index().rename(columns={0: 'ss_vol'}) # link_flow counts the number of vehicles, link_probe counts the number of probe vehicles
    t1 = time.time()
    
    return df_L_flow

def map_reduce_edge_flow(day, hour, quarter, ss_id, quarter_counts):
    ### One time step of ABM simulation

    t_odsp_0 = time.time()
    ### skip when no OD calculation is required
    unique_origin = OD_ss.shape[0]
    if unique_origin == 0:
        return pd.DataFrame([], columns=['start_sp', 'end_sp', 'ss_vol']), [], [], 0

    ### Build a pool
    process_count = 40
    pool = Pool(processes=process_count)

    ### Find shortest pathes
    res = pool.imap_unordered(map_edge_flow_residual, [(i, quarter_counts) for i in range(unique_origin)])

    ### Close the pool
    pool.close()
    pool.join()
    t_odsp_1 = time.time()

    ### Organize results
    agent_info_routes = list(res)
    # agent_info_routes, agent_path_timing = zip(*res)
    agent_info_routes_found = [a for a in agent_info_routes if a['arr']=='a']
    agent_info_routes_notfound = [a for a in agent_info_routes if a['arr']=='n']

    edge_volume = reduce_edge_flow_pd(agent_info_routes_found, day, hour, quarter, ss_id)
    
    ss_residual_OD_list = [(r['agent_id'], r['h_sp'], r['d_sp']) for r in agent_info_routes_found if r['h_sp']!=r['d_sp']]
    #ss_travel_time_list = [(r['agent_id'], day, hour, quarter, ss_id, r['travel_time']) for r in agent_info_routes]
    ss_travel_time_list = []

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

def read_OD(nodes_df=None, demand_files=None):
    ### Read the OD table of this time step

    t_OD_0 = time.time()
    logger = logging.getLogger("bk_evac")

    ### Read OD from a list of files
    od_list = []
    for demand_file in demand_files:
        sub_od = pd.read_csv(demand_file)
        od_list.append(sub_od)
    OD = pd.concat(od_list, ignore_index=True)

    ### Change OD list from using osmid to sequential id. It is easier to find the shortest path based on sequential index.
    if 'node_id_igraph_O' not in OD.columns:
        OD = pd.merge(OD, nodes_df[['node_osmid', 'node_id_igraph']], how='left', left_on='O', right_on='node_osmid')
        OD = pd.merge(OD, nodes_df[['node_osmid', 'node_id_igraph']], how='left', left_on='D', right_on='node_osmid', suffixes=['_O', '_D'])
    if 'agent_id' not in OD.columns: OD['agent_id'] = np.arange(OD.shape[0])
    if 'hour' not in OD.columns: OD['hour'] = 0
    OD['origin_sp'] = OD['node_id_igraph_O'] + 1 ### the node id in module sp is 1 higher than igraph id
    OD['destin_sp'] = OD['node_id_igraph_D'] + 1
    OD = OD[['agent_id', 'origin_sp', 'destin_sp', 'hour']]
    OD = OD.iloc[0:1000]

    t_OD_1 = time.time()
    logging.info('{} sec to read {} OD pairs'.format(t_OD_1-t_OD_0, OD.shape[0]))
    return OD

def output_edges_df(edges_df, day, hour, quarter, random_seed=None, scen_nm=None, simulation_outputs=None):

    ### Aggregate and calculate link-level variables after all increments
    # edges_df.loc[edges_df['tot_vol']>0, ['edge_id_igraph', 'type', 'tot_vol', 'true_vol', 't_avg']].round({'t_avg', 2}).to_csv(simulation_outputs+'/edges_df/edges_df_scen{}_r{}_DY{}_HR{}_QT{}.csv'.format(scen_nm, random_seed, day, hour, quarter), index=False)
    edges_df.loc[edges_df['tot_vol']>0, ['type', 'tot_vol', 'true_vol', 't_avg']].to_csv(simulation_outputs+'/edges_df/edges_df_scen{}_r{}_DY{}_HR{}_QT{}.csv'.format(scen_nm, random_seed, day, hour, quarter), index=True)

def sta(random_seed=None, quarter_counts=None, scen_nm=None, damage_file_edges=None, 
        network_file_nodes=None, network_file_edges=None, demand_files=None, simulation_outputs=None):

    t_main_0 = time.time()
    logger = logging.getLogger("bk_evac")

    ### Define global variables to be shared with subprocesses
    global g ### weighted graph
    global OD_ss ### substep demand
    global edges_df ### link weights

    ### Read in the edge attribute for volume delay calculation later
    edges_df0 = pd.read_csv(network_file_edges)
    edges_df0['start_end_sp'] = edges_df0['start_sp']+'-'edges_df0['end_sp']
    edges_df0 = edges_df0.set_index(start_end_sp)
    nodes_df = pd.read_csv(network_file_nodes)
    ### damage
    if damage_file_edges is not None:
        damange_df = pd.read_csv(damage_file_edges)
        edges_df0.loc[edges_df0['edge_id_igraph'].isin(damage_df['edge_id_igraph']), 'fft'] = 10e7

    node_count = max(len(np.unique(edges_df0['start_igraph'])), len(np.unique(edges_df0['end_igraph'])))
    g_coo = ssparse.coo_matrix((edges_df0['fft']*1.2, (edges_df0['start_igraph'], edges_df0['end_igraph'])), shape=(node_count, node_count))
    # edges_df0 = edges_df0[['edge_id_igraph', 'start_sp', 'end_sp', 'length', 'capacity', 'fft', 'type']]
    edges_df0 = edges_df0[['length', 'capacity', 'fft', 'type']]
    sio.mmwrite(simulation_outputs+'/network_sparse_scen{}_r{}.mtx'.format(scen_nm, random_seed), g_coo)

    all_OD = read_OD(nodes_df = nodes_df, demand_files=demand_files)

    ### Quarters and substeps
    ### probability of being in each division of hour
    quarter_ps = [1/quarter_counts for i in range(quarter_counts)]
    quarter_ids = [i for i in range(quarter_counts)]
    ### probability of being in each substep
    substep_counts = 15
    substep_ps = [1/substep_counts for i in range(substep_counts)] 
    substep_ids = [i for i in range(substep_counts)]
    logging.info('{} quarters per hour, {} substeps'.format(quarter_counts, substep_counts))

    residual_OD_list = []
    travel_time_list = []

    ### Loop through days and hours
    for day in ['na']:

        ### Read in the initial network (free flow travel time
        g = interface.readgraph(bytes(simulation_outputs+'/network_sparse_scen{}_r{}.mtx'.format(scen_nm, random_seed), encoding='utf-8'))
        ### test damage
        # test_sp = g.dijkstra(253182, 253708)
        # print(test_sp.distance(253708), [edge for edge in test_sp.route(253708)])
        # sp_route_df = pd.DataFrame([edge for edge in test_sp.route(253708)], columns=['start_sp', 'end_sp'])
        # print(sp_route_df.shape)
        # test_sp.clear()
        # sub_edges_df = edges_df0[edges_df0['start_sp'].isin(sp_route_df['start_sp']) & edges_df0['end_sp'].isin(sp_route_df['end_sp'])]
        # print(sub_edges_df.shape, sp_route_df.shape)
        # print(sub_edges_df)
        # print(edges_df0[edges_df0['start_sp'].isin(sp_route_df['start_sp'])])
        # print(edges_df0[edges_df0['end_sp'].isin(sp_route_df['end_sp'])])
        # sys.exit(0)

        # test_sp = g.dijkstra(511231, 622140)
        # print(test_sp.distance(622140))
        # sp_route_df = pd.DataFrame([edge for edge in test_sp.route(622140)], columns=['start_sp', 'end_sp'])
        # print(sp_route_df.shape)
        # print(sp_route_df)
        # test_sp.clear()
        # sub_edges_df = edges_df0[(edges_df0['start_sp'].isin(sp_route_df['start_sp'].iloc[list(range(80))])) & (edges_df0['end_sp'].isin(sp_route_df['end_sp'].iloc[list(range(80))]))]
        # print(sub_edges_df.shape, sp_route_df.shape)
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

                logging.info('DY {}, HR {}, QT {}, quarter_demand {}'.format(day, hour, quarter, quarter_demand))

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
                    logging.info('DY{}_HR{} SS {}: {} sec, {} OD pairs'.format(day, hour, ss_id, t_substep_1-t_substep_0, OD_ss.shape[0], ))

                output_edges_df(edges_df, day, hour, quarter, random_seed=random_seed, scen_nm=scen_nm, simulation_outputs=simulation_outputs)

                ### stats
                ### step outputs
                with open(simulation_outputs+'/stats/stats_scen{}.csv'.format(scen_nm),'a') as stats_outfile:
                    stats_outfile.write(",".join([str(x) for x in [random_seed, day, hour, quarter, 
                        quarter_demand, residual_demand, len(residual_OD_list),
                        np.sum(edges_df['t_avg']*edges_df['true_vol']/(quarter_demand*60)),
                        np.sum(edges_df['length']*edges_df['true_vol']/(quarter_demand*1000)),
                        np.mean(edges_df.nlargest(10, 'true_vol')['true_vol'])]])+"\n")
                
                ### Travel time by road category
                # edges_df['tot_t_hr{}_qt{}'.format(hour, quarter)] = np.round(edges_df['true_vol'] * edges_df['t_avg'], 2)

                t_hour_1 = time.time()
                ### log hour results before resetting the flow for the next time step
                logging.info('DY{}_HR{}_QT{}: {} sec, OD {}, {} residual'.format(day, hour, quarter, round(t_hour_1-t_hour_0, 3), quarter_demand, len(residual_OD_list)))
                gc.collect()

    # Output
    # output_edges_df(edges_df, day, hour, quarter, random_seed=random_seed, scen_nm=scen_nm, project_folder=project_folder)
    print('total non arrival {}'.format(tot_non_arrival))
    
    t_main_1 = time.time()
    print('total run time: {} sec \n\n\n\n\n'.format(t_main_1 - t_main_0))
    return travel_time_list, all_OD.shape[0]

def main(random_seed=0, scen_nm='base', quarter_counts=4):

    ### input files
    print('main')
    network_file_edges = work_dir+'/projects/tokyo_residential_above/network_inputs/edges_residual_demand.csv'
    network_file_nodes = work_dir+'/projects/tokyo_residential_above/network_inputs/nodes_residual_demand.csv'
    demand_files = [work_dir+"/projects/tokyo_residential_above/demand_inputs/od_residual_demand_0.csv",
                   work_dir + "/projects/tokyo_residential_above/demand_inputs/od_residual_demand_1.csv",
                   work_dir + "/projects/tokyo_residential_above/demand_inputs/od_residual_demand_2.csv"]
    simulation_outputs = scratch_dir + '/projects/tokyo_residential_above/simulation_outputs'
    damage_file_edges = None

    ### random seed and logging
    random.seed(random_seed)
    np.random.seed(random_seed)
    logger = logging.getLogger("residual_demand")
    logging.basicConfig(filename=simulation_outputs+'/log/{}.log'.format(scen_nm), filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info(scen_nm)

    ### carry out sta/semi-dynamic assignment
    with open(simulation_outputs+'/stats/stats_scen{}.csv'.format(scen_nm),'w') as stats_outfile:
        stats_outfile.write(",".join(['random_seed', 'day', 'hour', 'quarter', 'quarter_demand', 'residual_demand', 'residual_demand_produced', 'avg_veh_min', 'avg_veh_km', 'avg_top10_vol'])+"\n")
    sta_results = sta(random_seed=random_seed, 
        quarter_counts=quarter_counts, scen_nm=scen_nm, damage_file_edges=damage_file_edges, 
        network_file_nodes=network_file_nodes, network_file_edges=network_file_edges, demand_files=demand_files, simulation_outputs=simulation_outputs)

if __name__ == '__main__':
    lp = LineProfiler()
    lp.add_function(sta)
    lp.add_function(output_edges_df)
    lp.add_function(read_OD)
    lp.add_function(update_graph)
    lp.add_function(map_reduce_edge_flow)
    lp.add_function(reduce_edge_flow_pd)
    lp_wrapper = lp(main)
    lp.run('main()')
    lp.print_stats()
    # python3 residual_demand_assignment.py > $SCRATCH/residual_demand/projects/tokyo_residential_above/simulation_outputs/profile_output.txt

    # main(random_seed=0, scen_nm='base', quarter_counts=4)

