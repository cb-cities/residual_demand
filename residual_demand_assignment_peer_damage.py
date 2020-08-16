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
from ctypes import c_double
import gc 
from heapq import nlargest
import shapely.wkt

pd.set_option('display.max_columns', 10)
sys.setrecursionlimit(2000)

absolute_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, absolute_path+'/../')
sys.path.insert(0, '/home/bingyu/')
from sp import interface 

def map_edge_flow_residual(arg):
    ### Find shortest path for each unique origin --> one destination
    ### In the future change to multiple destinations
    
    row = arg[0]
    quarter_counts = arg[1]
    agent_id = int(OD_ss['agent_id'].iloc[row])
    origin_ID = int(OD_ss['origin_sp'].iloc[row])
    destin_ID = int(OD_ss['destin_sp'].iloc[row])
    try:
        travel_dur = int(OD_ss['dur'].iloc[row])
        pass_damaged_link = int(OD_ss['pcl'].iloc[row])
    except ValueError:
        print(OD_ss.iloc[row])

    sp = g.dijkstra(origin_ID, destin_ID) ### g_0 is the network with imperfect information for route planning
    sp_dist = sp.distance(destin_ID) ### agent believed travel time with imperfect information
    
    if sp_dist > 10e7:
        sp.clear()
        return {'agent_id': agent_id, 'o_sp': origin_ID, 'd_sp': destin_ID, 'route': pd.DataFrame([], columns=['start_sp', 'end_sp']), 'arr': 'n', 'pcl': pass_damaged_link, 'dur': 0} ### empty path; not reach destination; travel time 0
    else:
        sp_route = sp.route(destin_ID) ### agent route planned with imperfect information
        try:
            sp_route_df = pd.DataFrame([edge for edge in sp_route], columns=['start_sp', 'end_sp'])
        except RecursionError:
            print(origin_ID, destin_ID, sp_dist)
        sp.clear()
        
        sub_edges_df = edges_df[(edges_df['start_sp'].isin(sp_route_df['start_sp'])) & (edges_df['end_sp'].isin(sp_route_df['end_sp']))]
        sp_route_df = sp_route_df.merge(sub_edges_df[['start_sp', 'end_sp', 'edge_id_igraph', 'previous_t', 'true_vol']], on=['start_sp', 'end_sp'], how='left')
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

        intersection_with_damaged_links = len(set(damage_df['edge_id_igraph'].values.tolist()) & set(trunc_sp_route_df['edge_id_igraph'].values.tolist()))
        if intersection_with_damaged_links > 0:
            pass_damaged_link = 1
        else:
            pass

        return {'agent_id': agent_id, 'o_sp': origin_ID, 'd_sp': destin_ID, 'h_sp': stop_node, 'travel_time': travel_time, 'route': trunc_edges, 'arr': 'a', 'pcl': pass_damaged_link, 'dur': travel_dur+travel_time}

def reduce_edge_flow_pd(agent_info_routes_found, day, hour, quarter, ss_id):
    ### Reduce (count the total traffic flow per edge) with pandas groupby

    t0 = time.time()
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
        return pd.DataFrame([], columns=['start_sp', 'end_sp', 'ss_vol']), [], [], [], 0

    ### Build a pool
    process_count = 35
    pool = Pool(processes=process_count, maxtasksperchild=1000)

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
    # try:
    #     edge_volume = reduce_edge_flow_pd(agent_info_routes, day, hour, quarter, ss_id)
    # except TypeError:
    #     return pd.DataFrame([], columns=['start_sp', 'end_sp', 'ss_vol']), [], [], 0
    
    ss_arrival_OD_list = [(r['agent_id'], r['dur'], r['pcl']) for r in agent_info_routes_found if r['h_sp']==r['d_sp']]
    ss_residual_OD_list = [(r['agent_id'], r['h_sp'], r['d_sp'], r['dur'], r['pcl']) for r in agent_info_routes_found if r['h_sp']!=r['d_sp']]
    #ss_travel_time_list = [(r['agent_id'], day, hour, quarter, ss_id, r['travel_time']) for r in agent_info_routes]
    ss_travel_time_list = []
    # print('ss {}, total od {}, found {}, not found {}'.format(ss_id, unique_origin, len(agent_info_routes_found), len(agent_info_routes_notfound)))
    # print('DY{}_HR{}_QT{} SS {}: {} O --> {} D found, dijkstra pool {} sec on {} processes'.format(day, hour, quarter, ss_id, unique_origin, len(agent_info_routes_found), t_odsp_1 - t_odsp_0, process_count))

    return edge_volume, ss_arrival_OD_list, ss_residual_OD_list, ss_travel_time_list, len(agent_info_routes_notfound)

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

def read_OD(nodes_df=None, project_folder=None, demand_scen_id='base', chunk=False):
    ### Read the OD table of this time step

    t_OD_0 = time.time()
    ## Change OD list from using osmid to sequential id. It is easier to find the shortest path based on sequential index.
    if chunk:
        od_list = []
        for chunk_num in range(3):
            sub_od = pd.read_csv(absolute_path+'{}/demand_inputs/od_{}_{}.csv'.format(project_folder, demand_scen_id, chunk_num))
            od_list.append(sub_od)
        OD = pd.concat(od_list, ignore_index=True)
    else:
        OD = pd.read_csv(absolute_path+'{}/demand_inputs/OD_scag_5pct.csv'.format(project_folder))
    # OD = pd.DataFrame({'O':[53018374], 'D':[65293753 ]})
    OD = pd.merge(OD, nodes_df[['node_osmid', 'node_id_igraph']], how='left', left_on='O', right_on='node_osmid')
    OD = pd.merge(OD, nodes_df[['node_osmid', 'node_id_igraph']], how='left', left_on='D', right_on='node_osmid', suffixes=['_O', '_D'])
    OD['origin_sp'] = OD['node_id_igraph_O'] + 1 ### the node id in module sp is 1 higher than igraph id
    OD['destin_sp'] = OD['node_id_igraph_D'] + 1
    OD['dur'] = 0
    OD['pcl'] = 0
    if 'agent_id' not in OD.columns:
        OD['agent_id'] = np.arange(OD.shape[0])
    if 'hour' not in OD.columns:
        # OD['hour'] = 0
        OD['hour'] = np.random.choice([6,7,8,9], size=OD.shape[0], p=[0.1, 0.4, 0.4, 0.1])
    OD = OD[['agent_id', 'origin_sp', 'destin_sp', 'hour', 'dur', 'pcl']]
    # OD = OD.iloc[0:1000]
    # print(OD[OD['agent_id']==757])

    t_OD_1 = time.time()
    print('{} sec to read {} OD pairs'.format(t_OD_1-t_OD_0, OD.shape[0]))
    return OD

def output_edges_df(edges_df, day, hour, quarter, random_seed=None, scen_id=None, project_folder=None, tot=False):

    ### Aggregate and calculate link-level variables after all increments
    if tot:
        edges_df.loc[edges_df['tot_vol']>0, ['edge_id_igraph', 'tot_vol']].round({'t_avg': 2}).to_csv(absolute_path+'{}/simulation_outputs/edges_df/edges_df_tot_scen{}_r{}_DY{}_HR{}_QT{}.csv'.format(project_folder, scen_id, random_seed, day, hour, quarter), index=False)
    else:
        edges_df.loc[edges_df['true_vol']>0, ['edge_id_igraph', 'true_vol', 't_avg']].round({'t_avg': 2}).to_csv(absolute_path+'{}/simulation_outputs/edges_df/edges_df_true_scen{}_r{}_DY{}_HR{}_QT{}.csv'.format(project_folder, scen_id, random_seed, day, hour, quarter), index=False)

def sta(random_seed=0, quarter_counts=4, scen_id='base', damage_df=None, project_folder=None, od_chunk=False):

    t_main_0 = time.time()
    ### Fix random seed
    np.random.seed(random_seed)

    ### Define global variables to be shared with subprocesses
    global g ### weighted graph
    global OD_ss ### substep demand
    global edges_df ### link weights

    ### Read in the edge attribute for volume delay calculation later
    edges_df0 = pd.read_csv(absolute_path+'{}/network_inputs/unique_id_network_attributes.csv'.format(project_folder))
    edges_df0 = edges_df0.rename(columns={'uniqueid': 'edge_id_igraph', 'start': 'start_igraph', 'end': 'end_igraph'})
    nodes_df = pd.read_csv(absolute_path+'{}/network_inputs/unique_id_nodes.csv'.format(project_folder))
    nodes_df = nodes_df.rename(columns={'osmid': 'node_osmid'})

    demand_scen = scen_id.split('%')[0]
    bridge_scen = scen_id.split('%')[1]
    
    ### damage
    ### each damaged osmid could have more than one edge_id_igraph
    for edge in damage_df.itertuples():
        edges_df0.loc[edges_df0['edge_id_igraph']==getattr(edge, "edge_id_igraph"), 'capacity'] *= getattr(edge, "capacity_discount_to")
    if float(bridge_scen) == 0:
        edges_df0.loc[edges_df0['edge_id_igraph'].isin(damage_df['edge_id_igraph']), 'fft'] = 10e7

    node_count = max(len(np.unique(edges_df0['start_igraph'])), len(np.unique(edges_df0['end_igraph'])))
    g_coo = ssparse.coo_matrix((edges_df0['fft']*1.2, (edges_df0['start_igraph'], edges_df0['end_igraph'])), shape=(node_count, node_count))
    edges_df0 = edges_df0[['edge_id_igraph', 'start_sp', 'end_sp', 'length', 'capacity', 'fft']]
    sio.mmwrite(absolute_path+'{}/simulation_outputs/network_sparse_scen{}_r{}.mtx'.format(project_folder, scen_id, random_seed), g_coo)

    all_OD = read_OD(nodes_df = nodes_df, project_folder=project_folder, demand_scen_id=demand_scen, chunk=od_chunk)

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
    agent_arr_list = []

    ### Loop through days and hours
    for day in ['na']:

        ### Read in the initial network (free flow travel time
        g = interface.readgraph(bytes(absolute_path+'{}/simulation_outputs/network_sparse_scen{}_r{}.mtx'.format(project_folder, scen_id, random_seed), encoding='utf-8'))
        ### test damage
        # test_sp = g.dijkstra(31269, 131637)
        # print(test_sp.distance(131637), [edge for edge in test_sp.route(131637)])
        # sys.exit(0)

        ### Variables reset at the beginning of each day
        edges_df = edges_df0.copy() ### length, capacity and fft that should never change in one simulation
        edges_df['previous_t'] = edges_df['fft'] ### Used to find which edge to update. At the beginning of each day, previous_t is the free flow time.
        edges_df['tot_vol'] = 0
        tot_non_arrival = 0

        for hour in range(6,11):

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
                OD_residual = pd.DataFrame(residual_OD_list, columns=['agent_id', 'origin_sp', 'destin_sp', 'dur', 'pcl'])
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
                        edge_volume, ss_arrival_OD_list, ss_residual_OD_list, ss_travel_time_list, ss_non_arrival = map_reduce_edge_flow(day, hour, quarter, ss_id, quarter_counts)
                        tot_non_arrival += ss_non_arrival
                        residual_OD_list += ss_residual_OD_list
                        agent_arr_list += ss_arrival_OD_list
                        # travel_time_list += ss_travel_time_list

                        ### Updating
                        edges_df = update_graph(edge_volume, edges_df, day, hour, quarter, ss_id, quarter_demand, assigned_demand, quarter_counts)

                    t_substep_1 = time.time()
                    print('DY{}_HR{} SS {}: {} sec, {} OD pairs'.format(day, hour, ss_id, t_substep_1-t_substep_0, OD_ss.shape[0], ))

                output_edges_df(edges_df, day, hour, quarter, random_seed=random_seed, scen_id=scen_id, project_folder=project_folder, tot=False)

                ### stats
                with open(absolute_path+project_folder+'/simulation_outputs/t_stats/t_stats_scen{}_r{}.csv'.format(scen_id, random_seed), 'a') as t_stats_outfile:
                    t_stats_outfile.write(",".join([ str(i) for i in [
                    random_seed, day, hour, quarter, quarter_demand, residual_demand, len(residual_OD_list),
                    np.sum(edges_df['t_avg']*edges_df['true_vol']/(quarter_demand*60)),
                    np.sum(edges_df['length']*edges_df['true_vol']/(quarter_demand*1000)),
                    np.mean(edges_df.nlargest(10, 'true_vol')['true_vol'])
                    ]]))
                
                ### Travel time by road category
                # edges_df['tot_t_hr{}_qt{}'.format(hour, quarter)] = np.round(edges_df['true_vol'] * edges_df['t_avg'], 2)

                t_hour_1 = time.time()
                ### log hour results before resetting the flow for the next time step
                print('DY{}_HR{}_QT{}: {} sec, OD {}, {} residual'.format(day, hour, quarter, round(t_hour_1-t_hour_0, 3), quarter_demand, len(residual_OD_list)))
                gc.collect()
                if len(agent_arr_list)>0:
                    # print(agent_arr_list)
                    print(len([1 for i in agent_arr_list if i[2]==1]))

    # Output
    output_edges_df(edges_df, day, hour, quarter, random_seed=random_seed, scen_id=scen_id, project_folder=project_folder, tot=True)
    print('total non arrival {}'.format(tot_non_arrival))
    agent_arr_dict = pd.DataFrame(agent_arr_list, columns=['agent_id', 'dur', 'pcl'])
    print(agent_arr_dict[agent_arr_dict['pcl']==1].head())
    agent_arr_dict.to_csv(absolute_path+project_folder+'/simulation_outputs/arr_dict/arr_dict_scen{}_r{}.csv'.format(scen_id, random_seed), index=False)
    
    
    t_main_1 = time.time()
    print('total run time: {} sec \n\n\n\n\n'.format(t_main_1 - t_main_0))

def main(random_seed=0, scen_id='base', damage_df=None, quarter_counts=4, project_folder='', od_chunk=False):

    # random_seed = 0#int(os.environ['RANDOM_SEED'])
    # scen_id = 2#int(os.environ['SCEN_ID'])
    print('random_seed', random_seed, 'scen_id', scen_id)

    with open(absolute_path+project_folder+'/simulation_outputs/t_stats/t_stats_scen{}_r{}.csv'.format(scen_id, random_seed), 'w') as t_stats_outfile:
        t_stats_outfile.write(",".join(['random_seed', 'day', 'hour', 'quarter', 'quarter_demand', 'residual_demand', 'residual_demand_produced', 'avg_veh_min', 'avg_veh_km', 'avg_top10_vol'])+"\n")

    ### carry out sta/semi-dynamic assignment
    sta(random_seed=random_seed, quarter_counts=quarter_counts, scen_id=scen_id, damage_df=damage_df, project_folder=project_folder, od_chunk=od_chunk)


if __name__ == '__main__':
    project_folder = '/projects/bay_area_peer'
    global damage_df

    # for (demand_scen, bridge_scen) in [('c', 0), ('hc', 0.5), ('n', 1), ('c', 1), ('hc', 1)]:
    for (demand_scen, bridge_scen) in [('hc', 0.5), ('c', 0)]:

        # damage_df0 = pd.read_csv(absolute_path+'{}/network_inputs/bridge_closure/bridge_closure_day90.csv'.format(project_folder))
        # damage_links = damage_df0['OSMWayID1'].values.tolist() + damage_df0.dropna(subset=['OSMWayID2'])['OSMWayID2'].values.tolist()
        damage_df = pd.DataFrame({'edge_id_igraph': [76239, 285158, 313500, 425877]}) ### Bay Bridge
        damage_df['capacity_discount_to'] = bridge_scen

        main(random_seed=0, scen_id='{}%{}'.format(demand_scen, bridge_scen), damage_df=damage_df, quarter_counts=4, project_folder=project_folder, od_chunk=True)

