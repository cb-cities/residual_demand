import pandas as pd 

def main(project_folder=None, edges_df=None, hour=None, quarter=None):

    
    edge_vol_df = pd.read_csv(project_folder+'/simulation_outputs_c/edge_vol/edge_vol_h{}_q{}.csv'.format(hour, quarter))
    edge_vol_df = pd.merge(edge_vol_df, edges_df[['edge_id_igraph', 'geometry']], left_on='edgeid', right_on='edge_id_igraph', how='left')
    edge_vol_df = edge_vol_df[['edge_id_igraph', ' vol', 'geometry']]
    print(edge_vol_df.iloc[edge_vol_df[' vol'].idxmax()])
    edge_vol_df = edge_vol_df.to_csv(project_folder+'/simulation_outputs_c/edge_vol/edges_geom_h{}_q{}.csv'.format(hour, quarter))

if __name__ == '__main__':

    project_folder = 'projects/tokyo_residential_above'
    edges_df = pd.read_csv(project_folder+'/network_inputs/edges_residual_demand.csv')

    for hour in range(3,12):
        main(project_folder=project_folder, edges_df=edges_df, hour=hour, quarter=0)