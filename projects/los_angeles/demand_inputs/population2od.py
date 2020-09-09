import pandas as pd 
import geopandas as gpd 
import shapely.wkt
import sys 
import json 
import matplotlib.path as mpltPath 
import shapely.vectorized as sv 
from multiprocessing import Pool
import random 
import itertools 
import csv 
import os
import numpy as np 
import igraph 
import math 
import datetime 
import re 
import haversine

absolute_path = os.path.dirname(os.path.abspath(__file__))

def find_in_nodes(row_id):
    polygon_geom = zone_gdf['geometry'].iloc[row_id]
    polygon_id = zone_gdf['PARTIAL_ID'].iloc[row_id]
    points = nodes_df[['lon', 'lat']].values
    if polygon_geom.type == 'MultiPolygon':
        exteriors = [p.exterior.xy for p in polygon_geom]
        in_index_lst = []
        for e in exteriors:
            path = mpltPath.Path(list(zip(*e)))
            in_index = path.contains_points(points)
            in_index_lst += nodes_df['node_osmid'].loc[in_index].tolist()
        return (polygon_id, in_index_lst)
    else:
        path = mpltPath.Path(list(zip(*polygon_geom.exterior.coords.xy)))
        in_index = path.contains_points(points)
        return (polygon_id, nodes_df['node_osmid'].loc[in_index].tolist())

def TAZ_nodes():
    global zone_gdf, nodes_df
    ### Find corresponding nodes for each TAZ
    ### Save as 'taz_nodes.geojson'
    zone_df = pd.read_csv(absolute_path+'/scag_tract_boundary_2010.csv')
    crs = {'init': 'epsg:4326'}
    zone_gdf = gpd.GeoDataFrame(zone_df, crs=crs, geometry=zone_df['WKT'].map(shapely.wkt.loads))
    # zone_gdf = zone_gdf.iloc[0:1000]

    ### read nodes file
    nodes_df = pd.read_csv(absolute_path+'/../network_inputs/osm_nodes_scag.csv')

    zone_nodes_dict = dict()
    pool = Pool(processes=15)
    res = pool.imap_unordered(find_in_nodes, [k for k in range(zone_gdf.shape[0])])
    pool.close()
    pool.join()
    for (zone_id, zone_in_nodes) in list(res):
        zone_nodes_dict[str(zone_id)] = zone_in_nodes
    
    with open('zone_nodes.json', 'w') as outfile:
        json.dump(zone_nodes_dict, outfile, indent=2)
        # for k, v in zone_nodes_dict.items():
        #     outfile.write(json.dumps({k: v}))
        #     outfile.write("\n")

def nodal_OD():
    ### Sample random OD pairs
    zone_node_dict = json.load(open('zone_nodes.json'))
    zone_node_dict = {k: v for k, v in zone_node_dict.items() if len(v)>0}
    tract_pop_df = pd.read_csv('scag_tract_boundary_2010.csv') ### String to list of strings
    tract_pop_df = tract_pop_df.loc[tract_pop_df['PARTIAL_ID'].isin(zone_node_dict.keys())].reset_index(drop=True)
    tot_od = int(np.floor(np.sum(tract_pop_df['TOTPOP']) * 0.05))
    print('total od {}'.format(tot_od))

    zone_o_list = np.random.choice(tract_pop_df['PARTIAL_ID'], size=tot_od, p=tract_pop_df['TOTPOP']/np.sum(tract_pop_df['TOTPOP']))
    zone_d_list = np.random.choice(tract_pop_df['PARTIAL_ID'], size=tot_od, p=tract_pop_df['TOTPOP']/np.sum(tract_pop_df['TOTPOP']))

    node_od_list = []
    for (zone_o, zone_d) in zip(zone_o_list, zone_d_list):
        node_o = random.choice(zone_node_dict[str(zone_o)])
        node_d = random.choice(zone_node_dict[str(zone_d)])
        node_od_list.append([node_o, node_d])

    node_od_df = pd.DataFrame(node_od_list, columns=['O', 'D'])
    node_od_df.to_csv('OD_scag_5pct.csv', index=False)

def port_od():

    zone_node_dict = json.load(open('zone_nodes.json'))
    port_to_outside_od = []
    origin_tract = '3743000980033'
    origin_node_list = zone_node_dict[origin_tract]

    ### ===================================== ###
    ### port to each zone
    # for tract, nodes in zone_node_dict.items():
    #     if tract != origin_tract:
    #         try:
    #             port_to_outside_od.append([np.random.choice(origin_node_list), np.random.choice(nodes)])
    #         except ValueError:
    #             # print(len(nodes))
    #             pass
    
    ### ===================================== ###
    ### fix number of trips
    zone_node_dict = {k: v for k, v in zone_node_dict.items() if len(v)>0}
    tract_pop_df = pd.read_csv('scag_tract_boundary_2010.csv') ### String to list of strings
    tract_pop_df = tract_pop_df.loc[tract_pop_df['PARTIAL_ID'].isin(zone_node_dict.keys())].reset_index(drop=True)
    tot_od = 200000 # int(np.floor(np.sum(tract_pop_df['TOTPOP']) * 0.05))
    print('total od {}'.format(tot_od))

    zone_d_list = np.random.choice(tract_pop_df['PARTIAL_ID'], size=tot_od, p=tract_pop_df['TOTPOP']/np.sum(tract_pop_df['TOTPOP']))

    for zone_d in zone_d_list:
        node_o = random.choice(origin_node_list)
        node_d = random.choice(zone_node_dict[str(zone_d)])
        port_to_outside_od.append([node_o, node_d])
    ### ===================================== ###

    port_od_df = pd.DataFrame(port_to_outside_od, columns=['port_origin_osmid', 'outside_destin_osmid'])

    ### read nodes file
    nodes_df = pd.read_csv(absolute_path+'/../network_inputs/osm_nodes_scag.csv')
    port_od_df = port_od_df.merge(nodes_df, how='left', left_on='port_origin_osmid', right_on='node_osmid')
    port_od_df = port_od_df.merge(nodes_df, how='left', left_on='outside_destin_osmid', right_on='node_osmid', suffixes=['_origin', '_destin'])
    port_od_df['straight_line_dist'] = haversine.haversine(port_od_df['lat_origin'], port_od_df['lon_origin'], port_od_df['lat_destin'], port_od_df['lon_destin'])
    port_od_df = port_od_df[['port_origin_osmid', 'outside_destin_osmid', 'straight_line_dist']]
    print(port_od_df.head())
    # port_od_df.to_csv('port_to_outside_zone_od.csv', index=False)
    port_od_df.to_csv('port_daily_od.csv', index=False)

def add_geometry():
    zone_df = pd.read_csv(absolute_path+'/scag_tract_boundary_2010.csv')
    crs = {'init': 'epsg:4326'}
    zone_gdf = gpd.GeoDataFrame(zone_df, crs=crs, geometry=zone_df['WKT'].map(shapely.wkt.loads))
    zone_gdf['c_x'] = zone_gdf['geometry'].centroid.x
    zone_gdf['c_y'] = zone_gdf['geometry'].centroid.y

    zone_node_dict = json.load(open('zone_nodes.json'))
    zone_node_dict = {k: v for k, v in zone_node_dict.items() if len(v)>0}
    tract_pop_df = pd.read_csv('scag_tract_boundary_2010.csv') ### String to list of strings
    tract_pop_df = tract_pop_df.loc[tract_pop_df['PARTIAL_ID'].isin(zone_node_dict.keys())].reset_index(drop=True)
    tot_od = int(np.floor(np.sum(tract_pop_df['TOTPOP']) * 0.05))
    print('total od {}'.format(tot_od))

    ### population-based OD
    # zone_o_list = np.random.choice(tract_pop_df['PARTIAL_ID'], size=tot_od, p=tract_pop_df['TOTPOP']/np.sum(tract_pop_df['TOTPOP']))
    # zone_d_list = np.random.choice(tract_pop_df['PARTIAL_ID'], size=tot_od, p=tract_pop_df['TOTPOP']/np.sum(tract_pop_df['TOTPOP']))
    # tract_to_tract_od = pd.DataFrame({'tract_o': zone_o_list, 'tract_d': zone_d_list}).groupby(['tract_o', 'tract_d']).size().reset_index()
    # tract_to_tract_od = tract_to_tract_od.rename(columns={0: 'tract_to_tract_flow'})
    # tract_to_tract_od = tract_to_tract_od.merge(zone_gdf[['PARTIAL_ID', 'c_x', 'c_y']], how='left', left_on='tract_o', right_on='PARTIAL_ID').merge(zone_gdf[['PARTIAL_ID', 'c_x', 'c_y']], how='left', left_on='tract_d', right_on='PARTIAL_ID', suffixes=['_O', '_D'])
    # tract_to_tract_od['geometry'] = tract_to_tract_od.apply(lambda x: 'LINESTRING ({} {}, {} {})'.format(x['c_x_O'], x['c_y_O'], x['c_x_D'], x['c_y_D']), axis=1)
    
    # tract_to_tract_od[['tract_o', 'tract_d', 'tract_to_tract_flow', 'geometry']].to_csv('flow_pattern_scag_5pct.csv', index=False)

    ### port OD
    tot_od = 200000
    zone_d_list = np.random.choice(tract_pop_df['PARTIAL_ID'], size=tot_od, p=tract_pop_df['TOTPOP']/np.sum(tract_pop_df['TOTPOP']))
    tract_to_tract_od = pd.DataFrame({'tract_d': zone_d_list})
    tract_to_tract_od['tract_o'] = 3743000980033
    tract_to_tract_od = tract_to_tract_od.groupby(['tract_o', 'tract_d']).size().reset_index()
    tract_to_tract_od = tract_to_tract_od.rename(columns={0: 'port_to_tract_flow'})
    tract_to_tract_od = tract_to_tract_od.merge(zone_gdf[['PARTIAL_ID', 'c_x', 'c_y']], how='left', left_on='tract_o', right_on='PARTIAL_ID').merge(zone_gdf[['PARTIAL_ID', 'c_x', 'c_y']], how='left', left_on='tract_d', right_on='PARTIAL_ID', suffixes=['_O', '_D'])
    tract_to_tract_od['geometry'] = tract_to_tract_od.apply(lambda x: 'LINESTRING ({} {}, {} {})'.format(x['c_x_O'], x['c_y_O'], x['c_x_D'], x['c_y_D']), axis=1)
    
    tract_to_tract_od[['tract_o', 'tract_d', 'port_to_tract_flow', 'geometry']].to_csv('flow_pattern_port.csv', index=False)


if __name__ == '__main__':

    # TAZ_nodes()
    # nodal_OD()
    # port_od()
    add_geometry()
