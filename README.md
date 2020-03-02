# Semi-dynamic traffic assignment with residual demand
* Trips that did not finish in the previous time step will be considered as residual demand and re-assigned in the next time step.
* Assignment time interval can be as small as a few minutes to as big as a few hours. The larger the time interval, the lesser the residual trips.

### Road network
Since the assignment is still based on volume-delay functions (i.e., static), detailed road information (e.g., traffic signals, lanes, turn restrictions) is not needed. The code is based on [OSMnx](https://github.com/gboeing/osmnx) format. Other road geometry inputs need to be converted to this format.

### Demand
The demand is specified by a CSV file, including at least the origin node colume and the destination node column (node index corresponds to the road network input). Optionally, departure hour can be given.

### Assignment with residual demand
Trips are assigned to the instantaneous shortest path at the time of the assignment. The shortest path calculation uses the [sp](https://github.com/cb-cities/sp) package based on the Dijkstra's algorithm.

### Outputs
The outputs include the traffic volume in a particular assignment interval and the cumulative volume for each road link.