from __future__ import print_function
import os.path
import sys
import time
import random
import numpy as np
import pandas as pd
import pandana.network as pdna

### dir
home_dir = '.' # os.environ['HOME']+'/residual_demand'
work_dir = '.' # os.environ['WORK']+'/residual_demand'
scratch_dir = '.' # os.environ['SCRATCH']+'/residual_demand'

### random seed
random.seed(0)
np.random.seed(0)

def substep_assignment(nodes_df=None, edges_df=None, od_ss=None, quarter_demand=None, assigned_demand=None, quarter_counts=4):

    # print(nodes_df.shape, edges_df.shape)
    # print(len(np.unique(nodes_df.index)), len(np.unique(edges_df.index)))
    # sys.exit(0)
    net = pdna.Network(nodes_df["lon"], nodes_df["lat"], edges_df["start_igraph"], edges_df["end_igraph"], edges_df[["t_avg"]], twoway=False)
    net.set(pd.Series(net.node_ids))

    nodes_origin = od_ss['node_id_igraph_O'].values
    nodes_destin = od_ss['node_id_igraph_D'].values
    agent_ids = od_ss['agent_id'].values
    paths = net.shortest_paths(nodes_origin, nodes_destin)

    all_path_vol = dict()
    edge_travel_time_dict = edges_df['t_avg'].T.to_dict()
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
            p_dist += edge_travel_time_dict[edge_str]
            if p_dist > 3600/quarter_counts:
                od_residual_ss_list.append([agent_ids[path_i], edge_e, p[-1]])
                break
        path_i += 1

    all_path_vol_df = pd.DataFrame.from_dict(all_path_vol, orient='index', columns=['vol_ss'])
    # od_residual_ss = pd.DataFrame(od_residual_ss_list, columns=['agent_id', 'node_id_igraph_O', 'node_id_igraph_D'])
    
    new_edges_df = edges_df[['start_igraph', 'end_igraph', 'fft', 'capacity', 'vol_true', 'vol_tot']].copy()
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
    od_all = od_all[['agent_id', 'node_id_igraph_O', 'node_id_igraph_D', 'hour']]
    # od_all = od_all.iloc[0:1000]

    t_od_1 = time.time()
    print('{} sec to read {} OD pairs'.format(t_od_1-t_od_0, od_all.shape[0]))
    return od_all

def assignment(quarter_counts=4, substep_counts=15, network_file_nodes=None, network_file_edges=None, demand_files=None, simulation_outputs=None):

    ### network processing
    edges_df = pd.read_csv( work_dir + network_file_edges )
    edges_df = edges_df.sort_values(by='fft', ascending=False).drop_duplicates(subset=['start_igraph', 'end_igraph'], keep='first')
    edges_df['edge_str'] = edges_df['start_igraph'].astype('str') + '-' + edges_df['end_igraph'].astype('str')
    edges_df['capacity'] = np.where(edges_df['capacity']<1, 1900, edges_df['capacity'])
    edges_df = edges_df.set_index('edge_str')
    nodes_df = pd.read_csv( work_dir + network_file_nodes )
    nodes_df = nodes_df.set_index('node_id_igraph')

    ### OD processing
    od_all = read_od(demand_files=demand_files)
    ### Quarters and substeps
    ### probability of being in each division of hour
    quarter_ps = [1/quarter_counts for i in range(quarter_counts)]
    quarter_ids = [i for i in range(quarter_counts)]
    ### probability of being in each substep
    substep_ps = [1/substep_counts for i in range(substep_counts)] 
    substep_ids = [i for i in range(substep_counts)]
    print('{} quarters per hour, {} substeps'.format(quarter_counts, substep_counts))

    ### initial setup
    edges_df['t_avg'] = edges_df['fft'] * 1.2
    od_residual_list = []
    ### accumulator
    edges_df['vol_tot'] = 0
    
    ### Loop through days and hours
    for day in ['na']:
        for hour in range(3,5):

            ### Read OD
            od_hour = od_all[od_all['hour']==hour].copy()
            if od_hour.shape[0] == 0:
                od_hour = pd.DataFrame([], columns=['agent_id', 'node_id_igraph_O', 'node_id_igraph_D', 'hour'])

            ### Divide into quarters
            od_quarter_msk = np.random.choice(quarter_ids, size=od_hour.shape[0], p=quarter_ps)
            od_hour['quarter'] = od_quarter_msk

            for quarter in range(quarter_counts):

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

                od_substep_msk = np.random.choice(substep_ids, size=quarter_demand, p=substep_ps)
                od_quarter['ss_id'] = od_substep_msk
                edges_df['vol_true'] = 0

                for ss_id in substep_ids:

                    od_ss = od_quarter[od_quarter['ss_id']==ss_id]
                    assigned_demand += od_ss.shape[0]
                    if assigned_demand == 0:
                        continue
                    edges_df, od_residual_ss_list = substep_assignment(nodes_df=nodes_df, edges_df=edges_df, od_ss=od_ss, quarter_demand=quarter_demand, assigned_demand=assigned_demand, quarter_counts=quarter_counts)
                    od_residual_list += od_residual_ss_list
                    print('hour {}, quarter {}, ss {} finished'.format(hour, quarter, ss_id))

def main():
    ### input files
    network_file_edges = '/projects/tokyo_residential_above/network_inputs/edges_residual_demand.csv'
    network_file_nodes = '/projects/tokyo_residential_above/network_inputs/nodes_residual_demand.csv'
    demand_files = ["/projects/tokyo_residential_above/demand_inputs/od_residual_demand_0.csv",
                    "/projects/tokyo_residential_above/demand_inputs/od_residual_demand_1.csv",
                    "/projects/tokyo_residential_above/demand_inputs/od_residual_demand_2.csv"]
    simulation_outputs = '/projects/tokyo_residential_above/simulation_outputs'
    
    ### run residual_demand_assignment
    assignment(network_file_edges=network_file_edges, network_file_nodes=network_file_nodes, demand_files=demand_files, simulation_outputs=simulation_outputs)

if __name__ == "__main__":
    main()