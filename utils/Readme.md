# GNN4RC utils - Python scripts to convert circuit schematic to a DGL graph
This folder contains several useful python scripts that can handle the SPICE netlist into a DGL graph. We wish them can help with circuits netlist conversion, and reduce time cost of developing a EDA tool.

## subckt.py
### class Subckt
Definition of the subckt class, contianing a list of devices and a list of nets.
### class Net
Definition of the net class, statisitcal information of the net node.
### class Device
Definition of the device instance class, statisitcal information of the device instance. The instance could be a device or a instantiation of a subckt.

## read_cdl.py 
It reads a cdl file, and save the design as a list of class Subckt.

## build_graph.py
The input is a list of Subckt. This script has a "constructGraph" function that assigns each net/device/subckt instance a ID, and creates edges and extracts the feature vectors iteratively. Finally, the node IDs are used for DGL graph construction.

## cdl2graph_main.py
The main py file of the netlist conversion. 
### read_spf(filename)
This function use 'grep' command to find the net capacitance, and push the net name and cap pair into a dict
### findInterNetName(net_cap_dict, subcktList, topSubIdx, '')
This function record the names of each net node, and quary the net_cap_dict to find the parasitic capacitance.