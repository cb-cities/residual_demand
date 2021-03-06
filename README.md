# Semi-dynamic traffic assignment with residual demand

![Tokyo emission](images/tokyo_animation_emission_crop.gif)

### Features
* Quasi-equilibrium traffic assignment
* Efficient routing for millions of trips using [contraction hierarchy](https://github.com/UDST/pandana/blob/dev/examples/shortest_path_example.py) and priority-queue based Dijkstra algorithm [sp](https://github.com/cb-cities/sp)
* Temporal dynamics with residual demand, with time step of a few minutes
* Compatible with road network retrieved from [OSMnx](https://github.com/gboeing/osmnx)

### Use cases
* Calculating network traffic flow for small and large road networks (10 to 1,000,000 links) at sub-hourly time steps
* Visualizing traffic congestion dynamics throughout the day
* Analyzing traffic-induced carbon emissions (emission factor model)
* Assessing regional mobility and resilience with disruptions (e.g., road closure, seismic damages)

### Getting started
1. Clone the repository `git clone https://github.com/cb-cities/residual_demand.git`
2. Create a new Python 3.8 virtual environment and install dependencies `conda env create -f environment.yml`
    * Active the environment `conda activate residual_demand`
    * Install pandana [from Github](http://udst.github.io/pandana/installation.html). This is the contraction hierarchy code.
3. Run the test example `python scripts/run_simulation_template.py`
4. Examine the outputs in the output data (projects/test/simulation_outputs) and  visualization (projects/test/visualization_outputs) folders
5. Run for your own problem by following the test example