from __future__ import print_function
import gc 
import sys
import time
import random
import logging
import numpy as np
import pandas as pd
# import geopandas as gpd
# import contextily as ctx
# from shapely.wkt import loads
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable

if sys.version_info[1]==8:
    import pandana.network as pdna

### parallelized shortest path
from multiprocessing import Pool
from sp import interface

### random seed
random.seed(0)
np.random.seed(0)

### emission plot
def barth_2008(speed_mph_vector):
    
    b0 = 7.613534994965560
    b1 = -0.138565467462594
    b2 = 0.003915102063854
    b3 = -0.000049451361017
    b4 = 0.000000238630156 

    emission_gpermi_vector = np.exp(
        b0 + b1*speed_mph_vector  + b2*speed_mph_vector**2 + b3*speed_mph_vector**3 + b4*speed_mph_vector**4)

    return emission_gpermi_vector

def map_edge_flow_residual(arg):
    ### Find shortest path for each unique origin --> one destination
    ### In the future change to multiple destinations
    
    row = arg[0]
    quarter_counts = arg[1]
    agent_id = int(od_ss_global['agent_id'].iloc[row])
    origin_nid = int(od_ss_global['origin_nid'].iloc[row])
    destin_nid = int(od_ss_global['destin_nid'].iloc[row])
    agent_current_link = od_ss_global['current_link'].iloc[row]
    agent_current_link_time = float(od_ss_global['current_link_time'].iloc[row])

    sp = g.dijkstra(origin_nid, destin_nid) ### g_0 is the network with imperfect information for route planning
    sp_dist = sp.distance(destin_ID) ### agent believed travel time with imperfect information
    
    if sp_dist > 10e7:
        sp.clear()
        return {'agent_id': agent_id, 'origin_nid': origin_nid, 'destin_nid': destin_nid, 'stop_at': None, 'travel_time': None, 'route': [], 'status': 'no_route', 'current_link': None, 'current_link_time': None} ### empty path; not reach destination; travel time 0
    else:
        sp_route = sp.route(destin_nid) ### agent route planned with imperfect information
        sp_route_trunc = []
        p_dist = -agent_current_link_time
        new_current_link = None
        new_current_link_time = 0
        for edge_s, edge_e in sp_route:
            edge_str = "{}-{}".format(edge_s, edge_e)
            edge_travel_time =  edge_travel_time_dict[edge_str]
            remaining_time = 3600/quarter_counts - p_dist
            # enough time remaining for passing this link
            if remaining_time >= edge_travel_time:
                sp_route_trunc.append('{}-{}'.format(edge_s, edge_e))
                p_dist += edge_travel_time
                stop_node = edge_e
            # not enough time remaining for passing this link
            else:
                go_probability = remaining_time/edge_travel_time
                if random.uniform(0, 1) < go_probability:
                    stop_node = edge_e
                    sp_route_trunc.append('{}-{}'.format(edge_s, edge_e))
                else:
                    stop_node = edge_s
                    new_current_link = edge_str
                    new_current_link_time = remaining_time
                break
        sp.clear()

        return {'agent_id': agent_id, 'origin_nid': origin_nid, 'destin_nid': destin_nid, 'stop_at': stop_node, 'travel_time': 3600/quarter_counts, 'current_link': new_current_link, 'current_link_time': new_current_link_time, 'route': sp_route_trunc, 'status': 'enroute'}

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
    g = interface.from_dataframe(weighted_edges_df, 'start_nid', 'end_nid', 'weight')
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
        agent_info_routes_found = [a for a in agent_info_routes if a['status']=='enroute']
        agent_info_routes_notfound = [a for a in agent_info_routes if a['status']=='no_route']

        edge_volume = reduce_edge_flow_pd(agent_info_routes_found)
        
        ss_residual_OD_list = [(r['agent_id'], r['stop_at'], r['destin_nid']) for r in agent_info_routes_found if r['stop_at']!=r['destin_nid']]
        #ss_travel_time_list = [(r['agent_id'], day, hour, quarter, ss_id, r['travel_time']) for r in agent_info_routes]
        # print('ss {}, total od {}, found {}, not found {}'.format(ss_id, unique_origin, len(agent_info_routes_found), len(agent_info_routes_notfound)))
        # print('DY{}_HR{}_QT{} SS {}: {} O --> {} D found, dijkstra pool {} sec on {} processes'.format(day, hour, quarter, ss_id, unique_origin, len(agent_info_routes_found), t_odsp_1 - t_odsp_0, process_count))

    new_edges_df = weighted_edges_df[['start_nid', 'end_nid', 'fft', 'capacity', 'normal_fft', 'normal_capacity', 'length', 'is_highway', 'vol_true', 'vol_tot']].copy()
    new_edges_df = new_edges_df.join(edge_volume, how='left')
    new_edges_df['vol_ss'] = new_edges_df['vol_ss'].fillna(0)
    new_edges_df['vol_true'] += new_edges_df['vol_ss']
    new_edges_df['vol_tot'] += new_edges_df['vol_ss']
    new_edges_df['flow'] = (new_edges_df['vol_true']*quarter_demand/assigned_demand)*quarter_counts
    new_edges_df['t_avg'] = new_edges_df['fft'] * ( 1 + 0.6 * (new_edges_df['flow']/new_edges_df['capacity'])**4 ) * 1.2
    new_edges_df['t_avg'] = new_edges_df['t_avg'].round(2)

    return new_edges_df, ss_residual_OD_list
    
# @profile
def substep_assignment(nodes_df=None, weighted_edges_df=None, od_ss=None, quarter_demand=None, assigned_demand=None, quarter_counts=4, trip_info=None, agent_time_limit = None, sample_interval=1):

    # weighted_edges_df["weight"] = weighted_edges_df["length"]
    net = pdna.Network(nodes_df["x"], nodes_df["y"], weighted_edges_df["start_nid"], weighted_edges_df["end_nid"], weighted_edges_df[["weight"]], twoway=False)
    net.set(pd.Series(net.node_ids))

    nodes_origin = od_ss['origin_nid'].values
    nodes_destin = od_ss['destin_nid'].values
    nodes_current = od_ss['current_nid'].values
    agent_ids = od_ss['agent_id'].values
    agent_current_links = od_ss['current_link'].values
    agent_current_link_times = od_ss['current_link_time'].values
    paths = net.shortest_paths(nodes_current, nodes_destin)
    
    # check agent time limit
    path_lengths = net.shortest_path_lengths(nodes_current, nodes_destin)
    remove_agent_list = []
    if agent_time_limit is None:
        pass
    else:
        for agent_idx in range(len(agent_ids)):
            agent_id = agent_ids[agent_idx]
            planned_trip_length = path_lengths[agent_idx]
            trip_length_limit = agent_time_limit[agent_id]
            if planned_trip_length > trip_length_limit+0:
                remove_agent_list.append(agent_id)

    edge_travel_time_dict = weighted_edges_df['t_avg'].T.to_dict()
    edge_current_vehicles = weighted_edges_df['veh_current'].T.to_dict()
    edge_quarter_vol = weighted_edges_df['vol_true'].T.to_dict()
    od_residual_ss_list = []
    # all_paths = []
    path_i = 0
    for p in paths:
        # p_length = path_lengths[path_i]
        # if p_length >= 36000:
        #     trip_info[(agent_id, trip_origin, trip_destin)][0] = 'too_long'
        #     path_i += 1
        #     continue
        # p_dist = -current_link_times[path_i]
        # remaining_time = 3600/quarter_counts - p_dist
        trip_origin = nodes_origin[path_i]
        trip_destin = nodes_destin[path_i]
        agent_id = agent_ids[path_i]
        ### remove some agent (path too long)
        if agent_id in remove_agent_list:
            path_i += 1
            # no need to update trip info
            continue
        remaining_time = 3600/quarter_counts + agent_current_link_times[path_i]
        used_time = 0
        for edge_s, edge_e in zip(p, p[1:]):
            edge_str = "{}-{}".format(edge_s, edge_e)
            edge_travel_time = edge_travel_time_dict[edge_str]
            if (remaining_time > edge_travel_time) and (edge_travel_time < 36000):
                # all_paths.append(edge_str)
                # p_dist += edge_travel_time
                remaining_time -= edge_travel_time
                used_time += edge_travel_time
                edge_quarter_vol[edge_str] += (1 * sample_interval)
                trip_stop = edge_e
                if edge_str == agent_current_links[path_i]:
                    edge_current_vehicles[edge_str] -= (1 * sample_interval)
                # print('1: ', trip_origin, trip_destin, edge_s, edge_e, trip_stop)
            else:
                if edge_str != agent_current_links[path_i]:
                    edge_current_vehicles[edge_str] += (1 * sample_interval)
                new_current_link = edge_str
                new_current_link_time = remaining_time
                trip_stop = edge_s
                od_residual_ss_list.append([agent_id, trip_origin, trip_destin, trip_stop, new_current_link, new_current_link_time])
                # go_probability = remaining_time/edge_travel_time
                # if random.uniform(0, 1) < go_probability:
                #     all_paths.append(edge_str)
                #     od_residual_ss_list.append([agent_ids[path_i], edge_e, p[-1], None, None])
                # else:
                #     new_current_link = edge_str
                #     new_current_link_time = remaining_time
                #     od_residual_ss_list.append([agent_ids[path_i], edge_s, p[-1], new_current_link, new_current_link_time])
                # print('2: ', trip_origin, trip_destin, edge_s, edge_e, trip_stop)
                break
        trip_info[(agent_id, trip_origin, trip_destin)][0] += 3600/quarter_counts
        trip_info[(agent_id, trip_origin, trip_destin)][1] += used_time
        trip_info[(agent_id, trip_origin, trip_destin)][2] = trip_stop
        # print('ss: ', trip_origin, trip_destin, edge_s, edge_e, trip_stop)
        path_i += 1

    # edge_volume = pd.DataFrame(all_paths, columns=['edge_str']).groupby('edge_str').size().to_frame(name=['vol_ss'])
    
    new_edges_df = weighted_edges_df[['u', 'v', 'start_nid', 'end_nid', 'fft', 'capacity', 'normal_fft', 'normal_capacity', 'length', 'is_highway', 'vol_true', 'vol_tot', 'veh_current', 'geometry']].copy()
    # new_edges_df = new_edges_df.join(edge_volume, how='left')
    # new_edges_df['vol_ss'] = new_edges_df['vol_ss'].fillna(0)
    # new_edges_df['vol_true'] += new_edges_df['vol_ss']
    new_edges_df['vol_true'] = new_edges_df.index.map(edge_quarter_vol)
    new_edges_df['veh_current'] = new_edges_df.index.map(edge_current_vehicles)
    # new_edges_df['vol_tot'] += new_edges_df['vol_ss']
    new_edges_df['flow'] = (new_edges_df['vol_true']*quarter_demand/assigned_demand)*quarter_counts
    new_edges_df['t_avg'] = new_edges_df['fft'] * ( 1 + 0.3 * (new_edges_df['flow']/new_edges_df['capacity'])**4 )
    new_edges_df['t_avg'] = np.where(new_edges_df['t_avg']>36000, 36000, new_edges_df['t_avg'])
    new_edges_df['t_avg'] = new_edges_df['t_avg'].round(2)
    # bay_bridge_links = [76239, 285158, 313500, 425877]
    # print(new_edges_df.loc[new_edges_df['uniqueid'].isin(bay_bridge_links), ['capacity', 'flow', 't_avg']])
    # sys.exit(0)

    return new_edges_df, od_residual_ss_list, trip_info

def write_edge_vol(edges_df=None, simulation_outputs=None, quarter=None, hour=None, scen_nm=None):

    if 'flow' in edges_df.columns:
        if edges_df.shape[0]<10:
            edges_df[['start_nid', 'end_nid', 'veh_current', 'vol_true', 'vol_tot', 'flow', 't_avg']].to_csv(simulation_outputs+'/edge_vol/edge_vol_hr{}_qt{}_{}.csv'.format(hour, quarter, scen_nm), index=False)
        else:
            edges_df.loc[edges_df['vol_true']>0, ['start_nid', 'end_nid', 'veh_current', 'vol_true', 'vol_tot', 'flow', 't_avg']].to_csv(simulation_outputs+'/edge_vol/edge_vol_hr{}_qt{}_{}.csv'.format(hour, quarter, scen_nm), index=False)

def write_final_vol(edges_df=None, simulation_outputs=None, quarter=None, hour=None, scen_nm=None):
    
    edges_df.loc[edges_df['vol_tot']>0, ['start_nid', 'end_nid', 'vol_tot']].to_csv(simulation_outputs+'/edge_vol/final_edge_vol_hr{}_qt{}_{}.csv'.format(hour, quarter, scen_nm), index=False)

def plot_edge_flow(edges_df=None, simulation_outputs=None, quarter=None, hour=None, scen_nm=None, var='flow'):
    
    if var in edges_df.columns:
        fig, ax = plt.subplots(1,1, figsize=(20,20))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        # edges_df[edges_df[var]>0].to_crs(epsg=3857).plot(column=var, lw=0.5, ax=ax, cax=cax, cmap='magma_r', legend=True, vmin=10, vmax=200)
        edges_df[edges_df[var]>=0].plot(column=var, lw=5, ax=ax, cax=cax, cmap='magma_r', legend=True, vmin=10, vmax=90)
        # ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite, alpha=0.2)
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(0.7)
        ax.set_title('{} at {:02d}hr {:02d}div'.format(var, hour, quarter), font={'size': 30})
        plt.savefig(simulation_outputs+'/../visualization_outputs/map_hr{}_qt{}_{}.png'.format(hour, quarter, scen_nm), transparent=False)

def assignment(quarter_counts=4, substep_counts=15, substep_size=4000000, edges_df=None, nodes_df=None, od_all=None, demand_files=None, simulation_outputs=None, scen_nm=None, hour_list=None, quarter_list=None, cost_factor=None, closure_hours=[], closed_links=None, agent_time_limit=None, sample_interval=1):

    ### OD processing
    # od_all = read_od(demand_files=demand_files, nodes_df=nodes_df)
    od_all['current_nid'] = od_all['origin_nid']
    trip_info = {(getattr(od, 'agent_id'), getattr(od, 'origin_nid'), getattr(od, 'destin_nid')): [0, 0, getattr(od, 'origin_nid')] for od in od_all.itertuples()}
    ### Quarters and substeps
    ### probability of being in each division of hour
    quarter_ps = [1/quarter_counts for i in range(quarter_counts)]
    quarter_ids = [i for i in range(quarter_counts)]

    ### initial setup
    od_residual_list = []
    ### accumulator
    edges_df['vol_tot'] = 0
    edges_df['veh_current'] = 0
    # print(edges_df.head())
    # sys.exit(0)
    
    ### Loop through days and hours
    for day in ['na']:
        for hour in hour_list:
            gc.collect()
            if hour in closure_hours:
                for row in closed_links.itertuples():
                    edges_df.loc[(edges_df['u']==getattr(row, 'u')) & (edges_df['v']==getattr(row, 'v')), 'capacity'] = 1
                    edges_df.loc[(edges_df['u']==getattr(row, 'u')) & (edges_df['v']==getattr(row, 'v')), 'fft'] = 36000
            else:
                edges_df['capacity'] = edges_df['normal_capacity']
                edges_df['fft'] = edges_df['normal_fft']

            ### Read OD
            od_hour = od_all[od_all['hour']==hour].copy()
            if od_hour.shape[0] == 0:
                od_hour = pd.DataFrame([], columns=od_all.columns)
            od_hour['current_link'] = None
            od_hour['current_link_time'] = 0

            ### Divide into quarters
            if 'quarter' in od_all.columns:
                pass
            else:
                od_quarter_msk = np.random.choice(quarter_ids, size=od_hour.shape[0], p=quarter_ps)
                od_hour['quarter'] = od_quarter_msk

            if quarter_list is None:
                quarter_list = quarter_ids
            for quarter in quarter_list:
                ### New OD in assignment period
                od_quarter = od_hour.loc[od_hour['quarter']==quarter, ['agent_id', 'origin_nid', 'destin_nid', 'current_nid', 'current_link', 'current_link_time']]
                ### Add resudal OD
                od_residual = pd.DataFrame(od_residual_list, columns=['agent_id', 'origin_nid', 'destin_nid', 'current_nid', 'current_link', 'current_link_time'])
                od_residual['quarter'] = quarter
                ### Total OD in each assignment period is the combined of new and residual OD
                od_quarter = pd.concat([od_quarter, od_residual], sort=False, ignore_index=True)
                ### Residual OD is no longer residual after it has been merged to the quarterly OD
                od_residual_list = []
                od_quarter = od_quarter[od_quarter['current_nid'] != od_quarter['destin_nid']]

                quarter_demand = od_quarter.shape[0] ### total demand for this quarter, including total and residual demand
                residual_demand = od_residual.shape[0] ### how many among the OD pairs to be assigned in this quarter are actually residual from previous quarters
                assigned_demand = 0

                substep_counts = max(1, (quarter_demand // substep_size) + 1)
                logging.info('HR {} QT {} has {}/{} ods/residuals {} substeps'.format(hour, quarter, quarter_demand, residual_demand, substep_counts))
                substep_ps = [1/substep_counts for i in range(substep_counts)] 
                substep_ids = [i for i in range(substep_counts)]
                od_substep_msk = np.random.choice(substep_ids, size=quarter_demand, p=substep_ps)
                od_quarter['ss_id'] = od_substep_msk

                ### reset volume at each quarter
                edges_df['vol_true'] = 0

                for ss_id in substep_ids:
                    gc.collect()

                    time_ss_0 = time.time()
                    print(hour, quarter, ss_id)
                    od_ss = od_quarter[od_quarter['ss_id']==ss_id]
                    assigned_demand += od_ss.shape[0]
                    if assigned_demand == 0:
                        continue
                    ### calculate weight
                    weighted_edges_df = edges_df.copy()
                    ### weight by travel time
                    # weighted_edges_df['weight'] = (edges_df['t_avg'] - edges_df['fft']) * 0.5 + edges_df['length']*0.1 #+ cost_factor*edges_df['length']*0.1*(edges_df['is_highway']) ### 10 yen per 100 m --> 0.1 yen per m
                    weighted_edges_df['weight'] = edges_df['t_avg']
                    weighted_edges_df['weight'] = np.where(weighted_edges_df['weight']<0.1, 0.1, weighted_edges_df['weight'])
                    ### weight by emission
                    # weighted_edges_df['speed_mph'] = weighted_edges_df['length']/weighted_edges_df['t_avg']*2.237 ### m/s * 2.237 = mph
                    # weighted_edges_df['link_co2_g'] = barth_2008(weighted_edges_df['speed_mph'])*weighted_edges_df['length']/1609.34
                    # weighted_edges_df['weight'] = weighted_edges_df['link_co2_g']

                    ### traffic assignment with truncated path
                    edges_df, od_residual_ss_list, trip_info = substep_assignment(nodes_df=nodes_df, weighted_edges_df=weighted_edges_df, od_ss=od_ss, quarter_demand=quarter_demand, assigned_demand=assigned_demand, quarter_counts=quarter_counts, trip_info=trip_info, agent_time_limit=agent_time_limit, sample_interval=sample_interval)
                    od_residual_list += od_residual_ss_list
                    logging.info('HR {} QT {} SS {} finished, max vol {}, max hwy vol {}, time {}'.format(hour, quarter, ss_id, np.max(edges_df['vol_true']), np.max(edges_df.loc[edges_df['is_highway']==1, 'vol_true']), time.time()-time_ss_0))
                    print(hour, quarter, ss_id)
                    # print(trip_info)
                
                ### write quarterly results
                edges_df['vol_tot'] += edges_df['vol_true']
                if True: # hour >=16 or (hour==15 and quarter==3):
                    write_edge_vol(edges_df=edges_df, simulation_outputs=simulation_outputs, quarter=quarter, hour=hour, scen_nm=scen_nm)
                    # plot_edge_flow(edges_df=edges_df, simulation_outputs=simulation_outputs, quarter=quarter, hour=hour, scen_nm=scen_nm)
        # plot_edge_flow(edges_df=edges_df, simulation_outputs=simulation_outputs, quarter=quarter, hour=hour, scen_nm=scen_nm, var='vol_tot')
    ### output individual trip travel time and stop location
    # print(trip_info)
    trip_info_df = pd.DataFrame([[trip_key[0], trip_key[1], trip_key[2], trip_value[0], trip_value[1], trip_value[2]] for trip_key, trip_value in trip_info.items()], columns=['agent_id', 'origin_nid', 'destin_nid', 'travel_time', 'travel_time_used', 'stop_nid'])
    trip_info_df.to_csv(simulation_outputs+'/trip_info/trip_info_{}.csv'.format(scen_nm), index=False)

    write_final_vol(edges_df=edges_df, simulation_outputs=simulation_outputs, quarter=quarter, hour=hour, scen_nm=scen_nm)