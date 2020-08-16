import os 
import sys
import fiona 
import pandas as pd 
import geopandas as gpd

absolute_path = os.path.dirname(os.path.abspath(__file__))

def main():
    ca_counties = gpd.read_file(absolute_path+'/ca-county-boundaries/CA_Counties/CA_Counties_TIGER2016.shp')
    # print(ca_counties['NAME'])
    # return

    selected_counties = ca_counties[ca_counties['NAME'].isin(['Los Angeles', 'Orange', 'Ventura', 'San Bernardino', 'Riverside', 'Imperial'])].copy().reset_index(drop=True)
    print(selected_counties['NAME'])
    # return

    selected_polygon_buffer = selected_counties.buffer(distance=8000).simplify(tolerance=8000).to_crs({'init': 'epsg:4326'}).unary_union#.convex_hull
    selected_polygon_buffer = max(selected_polygon_buffer, key=lambda a: a.area)
    print( " ".join("{} {}".format(y, x) for (x, y) in list(selected_polygon_buffer.exterior.coords)) )
    print('\n', " ".join("{} {},".format(x, y) for (x, y) in list(selected_polygon_buffer.exterior.coords)) )
    ### use the output for query OSM data

if __name__ == '__main__':
    main()