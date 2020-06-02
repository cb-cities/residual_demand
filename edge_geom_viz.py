import pandas as pd 

project_folder = 'projects/los_angeles'

edge_df = pd.read_csv(project_folder+'/network_inputs/osm_edges.csv')
edge_vol_df = pd.read_csv(project_folder+'/simulation_outputs/edges_df/edges_df_scen1000_r0_DYna_HR1_QT3.csv')

edge_vol_df = pd.merge(edge_vol_df, edge_df[['edge_id_igraph', 'geometry']], on='edge_id_igraph', how='left')
print(edge_vol_df.iloc[0])
edge_vol_df = edge_vol_df.to_csv(project_folder+'/simulation_outputs/edges_df/edges_geom_df_scen1000_r0_DYna_HR1_QT3.csv')