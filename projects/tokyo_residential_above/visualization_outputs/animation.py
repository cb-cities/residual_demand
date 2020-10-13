import numpy as np 
import pandas as pd 
import imageio
import geopandas as gpd
import contextily as ctx
from shapely.wkt import loads
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable

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


def hourly_emission(hour, quarter, scen_nm):

    ### Get hour flow of a particular snapshot of the day
    edge_flow_df = pd.read_csv('../simulation_outputs/edge_vol/edge_vol_hr{}_qt{}_{}.csv'.format(hour, quarter, scen_nm))
    edge_flow_df['edge_str'] = edge_flow_df['start_igraph'].astype('str') + '-' + edge_flow_df['end_igraph'].astype('str')
    edge_flow_df = edge_flow_df.set_index('edge_str')

    ### Get attributes and geometry of each edge
    network_attr_df = pd.read_csv('../network_inputs/edges_residual_demand.csv')
    network_attr_df['edge_str'] = network_attr_df['start_igraph'].astype('str') + '-' + network_attr_df['end_igraph'].astype('str')
    network_attr_df = network_attr_df.set_index('edge_str')
    
    ### give the flow data a speed
    edge_emission_df = edge_flow_df.join(network_attr_df[['length', 'geometry']])
    edge_emission_df['speed_mph'] = edge_emission_df['length']/edge_emission_df['t_avg']*2.237 ### m/s * 2.237 = mph
    edge_emission_df['link_co2_g'] = barth_2008(edge_emission_df['speed_mph'])*edge_emission_df['flow']*edge_emission_df['length']/1609.34

    print(hour, quarter, np.sum(edge_emission_df['link_co2_g']/1e6), edge_emission_df['link_co2_g'].describe())
    return edge_emission_df


def plot_edge_flow(edges_df, hour, quarter, scen_nm):
    
    if 'link_co2_g' in edges_df.columns:

        edges_df = edges_df[edges_df['link_co2_g']>100].copy()
        edges_df['small_node_id'] = edges_df[['start_igraph','end_igraph']].min(axis=1)
        edges_df['large_node_id'] = edges_df[['start_igraph','end_igraph']].max(axis=1)
        # print(edges_df.head())
        undirected_edges_df = edges_df.groupby(['small_node_id', 'large_node_id']).agg({'link_co2_g': np.sum, 'geometry': 'first'}).reset_index()
        undirected_edges_gdf = gpd.GeoDataFrame(undirected_edges_df, crs='epsg:4326', geometry=undirected_edges_df['geometry'].map(loads))
        print(hour, quarter, np.sum(undirected_edges_df ['link_co2_g']/1e6), undirected_edges_df ['link_co2_g'].describe())

        fig, ax = plt.subplots(1,1, figsize=(20,20))
        ax.set_xlim([15.44e6, 15.70e6])
        ax.set_ylim([4.145e6, 4.355e6])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        undirected_edges_gdf.to_crs(epsg=3857).plot(column='link_co2_g', lw=0.5, ax=ax, cax=cax, cmap='magma_r', legend=True, vmin=0, vmax=5000)
        ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite, alpha=0.2)
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(0.7)
        ax.set_title('Emission by road (g) at {:02d}:{:02d}'.format(hour, quarter*15), font={'size': 30})
        plt.savefig('emission_map/emission_map_hr{}_qt{}_{}.png'.format(hour, quarter, scen_nm), transparent=False)
        # plt.show()
        plt.close()


for hour in range(17,23):
    for quarter in range(4):
        try:
            hourly_emission_df = hourly_emission(hour, quarter, 'ch_full')
            plot_edge_flow(hourly_emission_df, hour, quarter, 'ch_full')
        except Error:
            print(hour, quarter)