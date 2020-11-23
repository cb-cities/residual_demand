from __future__ import print_function
import os.path
import sys
import time
import random
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import contextily as ctx
from shapely.wkt import loads
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable

if sys.version_info[1]==8:
    import pandana.network as pdna

from multiprocessing import Pool
sys.path.insert(0, '/home/bingyu/Documents')
from sp import interface

### dir
home_dir = '/home/bingyu/Documents/residual_demand' # os.environ['HOME']+'/residual_demand'
work_dir = '/home/bingyu/Documents/residual_demand' # os.environ['WORK']+'/residual_demand'
scratch_dir = '/home/bingyu/Documents/residual_demand' # os.environ['SCRATCH']+'/residual_demand'

### random seed
random.seed(0)
np.random.seed(0)

def map_edge_flow_residual(arg):
    ### Find shortest path for each unique origin --> one destination
    ### In the future change to multiple destinations
    
    row = arg[0]
    quarter_counts = arg[1]
    agent_id = int(od_ss_global['agent_id'].iloc[row])
    origin_ID = int(od_ss_global['node_id_O'].iloc[row])
    destin_ID = int(od_ss_global['node_id_D'].iloc[row])

    sp = g.dijkstra(origin_ID, destin_ID) ### g_0 is the network with imperfect information for route planning
    sp_dist = sp.distance(destin_ID) ### agent believed travel time with imperfect information
    
    if sp_dist > 10e7:
        sp.clear()
        return {'agent_id': agent_id, 'origin_node_id': origin_ID, 'destin_node_id': destin_ID, 'route': [], 'arr': 'n'} ### empty path; not reach destination; travel time 0
    else:
        sp_route = sp.route(destin_ID) ### agent route planned with imperfect information
        sp_route_trunc = []
        p_dist = 0
        for edge_s, edge_e in sp_route:
            edge_str = "{}-{}".format(edge_s, edge_e)
            sp_route_trunc.append('{}-{}'.format(edge_s, edge_e))
            p_dist += edge_travel_time_dict[edge_str]
            stop_node = edge_e
            if p_dist > 3600/quarter_counts:
                # stop_node = edge_e
                break
        sp.clear()

        return {'agent_id': agent_id, 'origin_node_id': origin_ID, 'destin_node_id': destin_ID, 'stop_at': stop_node, 'travel_time': p_dist, 'route': sp_route_trunc, 'arr': 'a'}

def reduce_edge_flow_pd(agent_info_routes_found):
    ### Reduce (count the total traffic flow per edge) with pandas groupby

    t0 = time.time()
    df_L = pd.concat([pd.DataFrame(r['route'], columns=['edge_str']) for r in agent_info_routes_found])
    df_L_flow = df_L.groupby(['edge_str']).size().to_frame('vol_ss') # link_flow counts the number of vehicles, link_probe counts the number of probe vehicles
    t1 = time.time()
    # print('DY{}_HR{}_QT{} SS {}: reduce find {} edges, {} sec w/ pd.groupby, max substep volume {}'.format(day, hour, quarter, ss_id, df_L_flow.shape[0], t1-t0, np.max(df_L_flow['ss_vol'])))
    
    return df_L_flow

def substep_assignment_sp(nodes_df=None, weighted_edges_df=None, od_ss=None, quarter_demand=None, assigned_demand=None, quarter_counts=4):
    global g, edge_travel_time_dict, od_ss_global
    g = interface.from_dataframe(weighted_edges_df, 'start_node_id', 'end_node_id', 'weight')
    edge_travel_time_dict = weighted_edges_df['t_avg'].T.to_dict()
    od_ss_global = od_ss.copy()

    ### skip when no OD calculation is required
    unique_origin = od_ss.shape[0]
    if unique_origin == 0:
        edge_volume = pd.DataFrame([], columns=['vol_ss'])
        ss_residual_OD_list = []
    
    else:

        ### Build a pool
        process_count = 7
        pool = Pool(processes=process_count, maxtasksperchild=1000)

        ### Find shortest pathes
        res = pool.imap_unordered(map_edge_flow_residual, [(i, quarter_counts) for i in range(unique_origin)])

        ### Close the pool
        pool.close()
        pool.join()

        ### Organize results
        agent_info_routes = list(res)
        # agent_info_routes, agent_path_timing = zip(*res)
        agent_info_routes_found = [a for a in agent_info_routes if a['arr']=='a']
        agent_info_routes_notfound = [a for a in agent_info_routes if a['arr']=='n']

        edge_volume = reduce_edge_flow_pd(agent_info_routes_found)
        
        ss_residual_OD_list = [(r['agent_id'], r['stop_at'], r['destin_node_id']) for r in agent_info_routes_found if r['stop_at']!=r['destin_node_id']]
        #ss_travel_time_list = [(r['agent_id'], day, hour, quarter, ss_id, r['travel_time']) for r in agent_info_routes]
        # print('ss {}, total od {}, found {}, not found {}'.format(ss_id, unique_origin, len(agent_info_routes_found), len(agent_info_routes_notfound)))
        # print('DY{}_HR{}_QT{} SS {}: {} O --> {} D found, dijkstra pool {} sec on {} processes'.format(day, hour, quarter, ss_id, unique_origin, len(agent_info_routes_found), t_odsp_1 - t_odsp_0, process_count))

    new_edges_df = weighted_edges_df[['start_node_id', 'end_node_id', 'fft', 'capacity', 'length', 'is_highway', 'vol_true', 'vol_tot']].copy()
    new_edges_df = new_edges_df.join(edge_volume, how='left')
    new_edges_df['vol_ss'] = new_edges_df['vol_ss'].fillna(0)
    new_edges_df['vol_true'] += new_edges_df['vol_ss']
    new_edges_df['vol_tot'] += new_edges_df['vol_ss']
    new_edges_df['flow'] = (new_edges_df['vol_true']*quarter_demand/assigned_demand)*quarter_counts
    new_edges_df['t_avg'] = new_edges_df['fft'] * ( 1 + 0.6 * (new_edges_df['flow']/new_edges_df['capacity'])**4 ) * 1.2
    new_edges_df['t_avg'] = new_edges_df['t_avg'].round(2)

    return new_edges_df, ss_residual_OD_list
    
def substep_assignment(nodes_df=None, weighted_edges_df=None, od_ss=None, quarter_demand=None, assigned_demand=None, quarter_counts=4):

    # print(nodes_df.shape, edges_df.shape)
    # print(len(np.unique(nodes_df.index)), len(np.unique(edges_df.index)))
    # sys.exit(0)
    net = pdna.Network(nodes_df["lon"], nodes_df["lat"], weighted_edges_df["start_node_id"], weighted_edges_df["end_node_id"], weighted_edges_df[["weight"]], twoway=False)
    net.set(pd.Series(net.node_ids))

    nodes_origin = od_ss['node_id_O'].values
    nodes_destin = od_ss['node_id_D'].values
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
    
    new_edges_df = weighted_edges_df[['start_node_id', 'end_node_id', 'fft', 'capacity', 'length', 'is_highway', 'vol_true', 'vol_tot', 'geometry']].copy()
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
        od_chunk = pd.read_csv( work_dir + demand_file )
        od_list.append(od_chunk)
    
    od_all = pd.concat(od_list, ignore_index=True)
    od_all['node_id_O'] = od_all['O']
    od_all['node_id_D'] = od_all['D']
    od_all['hour'] = od_all['trip_hour']
    od_all = od_all[['agent_id', 'node_id_O', 'node_id_D', 'hour']]
    # od_all = od_all.iloc[0:10000000]

    t_od_1 = time.time()
    logging.info('{} sec to read {} OD pairs'.format(t_od_1-t_od_0, od_all.shape[0]))
    return od_all

def write_edge_vol(edges_df=None, simulation_outputs=None, quarter=None, hour=None, scen_nm=None):

    if 'flow' in edges_df.columns:
        edges_df.loc[edges_df['vol_true']>0, ['start_node_id', 'end_node_id', 'vol_true', 'flow', 't_avg']].to_csv(scratch_dir+simulation_outputs+'/edge_vol/edge_vol_hr{}_qt{}_{}.csv'.format(hour, quarter, scen_nm), index=False)

def plot_edge_flow(edges_df=None, simulation_outputs=None, quarter=None, hour=None, scen_nm=None):
    
    if 'flow' in edges_df.columns:
        fig, ax = plt.subplots(1,1, figsize=(20,20))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        # edges_df[edges_df['flow']>5].to_crs(epsg=3857).plot(column='flow', lw=0.5, ax=ax, cax=cax, cmap='magma_r', legend=True, vmin=5, vmax=500)
        edges_df[edges_df['flow']>0].to_crs(epsg=3857).plot(column='flow', lw=0.5, ax=ax, cax=cax, cmap='magma_r', legend=True, vmin=10, vmax=200)
        ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite, alpha=0.2)
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(0.7)
        ax.set_title('Traffic flow (veh/hr) at {:02d}:{:02d}'.format(hour, quarter*15), font={'size': 30})
        plt.savefig(scratch_dir+simulation_outputs+'/../visualization_outputs/flow_map_hr{}_qt{}_{}.png'.format(hour, quarter, scen_nm), transparent=False)

def assignment(quarter_counts=4, substep_counts=15, substep_size=100000, network_file_nodes=None, network_file_edges=None, demand_files=None, simulation_outputs=None, scen_nm=None, hour_list=None, quarter_list=None, cost_factor=None):

    ### network processing
    edges_df = pd.read_csv( work_dir + network_file_edges )
    edges_df = gpd.GeoDataFrame(edges_df, crs='epsg:4326', geometry=edges_df['geometry'].map(loads))
    edges_df = edges_df.sort_values(by='fft', ascending=False).drop_duplicates(subset=['start_node_id', 'end_node_id'], keep='first')
    edges_df['edge_str'] = edges_df['start_node_id'].astype('str') + '-' + edges_df['end_node_id'].astype('str')
    edges_df['capacity'] = np.where(edges_df['capacity']<1, 950, edges_df['capacity'])
    # edges_df['is_highway'] = np.where(edges_df['type'].isin(['motorway', 'motorway_link']), 1, 0)
    edges_df['is_highway'] = np.where(edges_df['type'].isin(['motorway', 'motorway_link']), 1, 0)
    edges_df = edges_df.set_index('edge_str')

    nodes_df = pd.read_csv( work_dir + network_file_nodes )
    nodes_df = nodes_df.set_index('node_id')

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
                od_hour = pd.DataFrame([], columns=['agent_id', 'node_id_O', 'node_id_D', 'hour'])

            ### Divide into quarters
            od_quarter_msk = np.random.choice(quarter_ids, size=od_hour.shape[0], p=quarter_ps)
            od_hour['quarter'] = od_quarter_msk

            for quarter in quarter_list:

                ### New OD in assignment period
                od_quarter = od_hour[od_hour['quarter']==quarter]
                ### Add resudal OD
                od_residual = pd.DataFrame(od_residual_list, columns=['agent_id', 'node_id_O', 'node_id_D'])
                od_residual['quarter'] = quarter
                ### Total OD in each assignment period is the combined of new and residual OD
                od_quarter = pd.concat([od_quarter, od_residual], sort=False, ignore_index=True)
                ### Residual OD is no longer residual after it has been merged to the quarterly OD
                od_residual_list = []
                od_quarter = od_quarter[od_quarter['node_id_O'] != od_quarter['node_id_D']]

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
    network_file_edges = '/projects/tokyo_osmnx/network_inputs/tokyo_edges.csv'
    # network_file_edges = '/projects/tokyo_residential_above/network_inputs/tokyo_edges_discount.csv'
    network_file_nodes = '/projects/tokyo_osmnx/network_inputs/tokyo_nodes.csv'
    demand_files = ["/projects/tokyo_osmnx/demand_inputs/od_0.csv",
                    "/projects/tokyo_osmnx/demand_inputs/od_1.csv",
                    "/projects/tokyo_osmnx/demand_inputs/od_2.csv"]
    simulation_outputs = '/projects/tokyo_osmnx/simulation_outputs'

    ### log file
    if sys.version_info[1]==8:
        logging.basicConfig(filename=scratch_dir+simulation_outputs+'/log/{}.log'.format(scen_nm), level=logging.INFO, force=True)
    elif sys.version_info[1]<8:
        logging.basicConfig(filename=scratch_dir+simulation_outputs+'/log/{}.log'.format(scen_nm), level=logging.INFO)
    else:
        print('newer version than 3.8')
    
    ### run residual_demand_assignment
    assignment(network_file_edges=network_file_edges, network_file_nodes=network_file_nodes, demand_files=demand_files, simulation_outputs=simulation_outputs, scen_nm=scen_nm, hour_list=hour_list, quarter_list=quarter_list, cost_factor=cost_factor)

    return True

if __name__ == "__main__":
    status = main(hour_list=list(range(3, 8)), quarter_list=[0,1,2,3], scen_nm='', cost_factor=0)
    # for cost_factor in [-2, -1, -0.5, 0, 0.5]:
    #     status = main(hour_list=[3,4,5,6,7,8,9,10,11,12], quarter_list=[0,1,2,3], scen_nm='costfct{}'.format(cost_factor), cost_factor=cost_factor)