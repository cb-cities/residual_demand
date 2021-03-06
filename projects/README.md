# Example project folder

This folder contains all the data inputs needed to run the test of this residual demand simulation code. The corresponding scripts can be found [here](../scripts/run_simulation_template.py).

### File structure
```
+-- network_inputs
|   +-- test_edges.csv (road network links geometry and properties)
|   +-- test_nodes.csv (road network nodes geometry and properties)
+-- demand_inputs
|   +-- test_od.csv (hourly OD trips)
+-- simulation_outputs
|   +-- link_weights (folder containing time-stepped link-level traffic volume)
|   +-- log (folders of log files)
|   +-- trip_info (folder containing individual trip travel)
```

### Road network
Bay Area road network retrieved from the [OSMnx](https://github.com/gboeing/osmnx). It has 224,224 nodes and 549,009 links.

### Demand
Bay Area commuting trips retrieved from the [Census Transportation Planning Package (CTPP)](https://ctpp.transportation.org/). It has 1,707,041 origin-destination (OD) pairs assigned to start from 6 AM to 9 AM.

### Assignment with residual demand
Trips are assigned to the instantaneous shortest path at the time of the assignment. The assignment interval is 15 minutes. The shortest path calculation uses the [sp](https://github.com/cb-cities/sp) package based on the Dijkstra's algorithm, or the [contraction hierarchy](https://github.com/UDST/pandana/blob/dev/examples/shortest_path_example.py) implementation by [pandana](https://github.com/UDST/pandana).

### Outputs
The outputs include the traffic volume in a particular assignment interval and the cumulative volume for each road link.