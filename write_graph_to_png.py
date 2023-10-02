import pydot
import sys
import networkx as nx

if len(sys.argv)!=3:
    sys.exit(f'Usage: python write_graph_to_png.py $inp_dot/gml $out_png')

inp_file = sys.argv[1]
out_file = sys.argv[2]

if '.dot' in inp_file:
    graphs = pydot.graph_from_dot_file(inp_file)
    graph = graphs[0]
    graph.write_png(out_file)

if '.gml' in inp_file:
    graph = nx.nx_pydot.to_pydot(nx.read_gml(inp_file, destringizer=None))
    graph.write_png(out_file)
