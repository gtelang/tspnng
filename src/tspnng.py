
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import scipy as sp
import numpy as np
import random
import networkx as nx
from prettytable import PrettyTable

from sklearn.cluster import KMeans
import argparse, os, sys, time
from colorama import init, Fore, Style, Back
init() # this line does nothing on Linux/Mac,
       # but is important for Windows to display
       # colored text. See https://pypi.org/project/colorama/
import yaml

def get_names_of_all_euclidean2D_instances(dirpath=\
         "./sym-tsp-tsplib/instances/euclidean_instances_yaml/" ):
     
     inst_names = []
     for name in os.listdir(dirpath):
         full_path = os.path.join(dirpath, name)
         if os.path.isfile(full_path):
             inst_names.append(name)
     return inst_names

def tsplib_instance_points(instance_file_name,\
                           dirpath="./sym-tsp-tsplib/instances/euclidean_instances_yaml/"):

        print(Fore.GREEN+"Reading " + instance_file_name, Style.RESET_ALL)
        with open(dirpath+instance_file_name) as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            points = np.asarray(data['points'])
        
        return points 
           
def uniform_points(numpts):
     return  sp.rand(numpts, 2).tolist()

def non_uniform_points(numpts):

    cluster_size = int(np.sqrt(numpts)) 
    numcenters   = cluster_size
    centers      = sp.rand(numcenters,2).tolist()
    scale, points = 4.0, []

    for c in centers:
        cx, cy = c[0], c[1]
        sq_size      = min(cx,1-cx,cy, 1-cy)

        loc_pts_x    = np.random.uniform(low  = cx-sq_size/scale, 
                                         high = cx+sq_size/scale, 
                                         size = (cluster_size,))
        loc_pts_y    = np.random.uniform(low = cy-sq_size/scale, 
                                         high = cy+sq_size/scale, 
                                         size = (cluster_size,))

        points.extend(zip(loc_pts_x, loc_pts_y))

    num_remaining_pts = numpts - cluster_size * numcenters
    remaining_pts = sp.rand(num_remaining_pts, 2).tolist()
    points.extend(remaining_pts)
    return points
def write_to_yaml_file(data, dir_name, file_name):
   with open(dir_name + '/' + file_name, 'w') as outfile:
          yaml.dump( data, outfile, default_flow_style = False)
class TSPNNGInput:
      def __init__(self, points=[]):
          self.points            = points

      def clearAllStates (self):
          self.points = []

      def generate_geometric_graph(self,graph_code):
           pass
def hamilton(G):
    F = [(G,[list(G.nodes())[0]])]
    n = G.number_of_nodes()
    while F:
        graph,path = F.pop()
        confs = []
        neighbors = (node for node in graph.neighbors(path[-1]) 
                     if node != path[-1]) #exclude self loops
        for neighbor in neighbors:
            conf_p = path[:]
            conf_p.append(neighbor)
            conf_g = nx.Graph(graph)
            conf_g.remove_node(path[-1])
            confs.append((conf_g,conf_p))
        for g,p in confs:
            if len(p)==n:
                return p
            else:
                F.append((g,p))
    return None
def perturb_points(points, alpha=0.01):
     points = np.asarray(points)

     for i in range(len(points)):
         theta     = random.uniform(0,2*np.pi)
         unitvec   = np.asarray([np.cos(theta), np.sin(theta)])
         points[i] = points[i] + alpha * unitvec

     return points
def run_handler():
    fig, ax =  plt.subplots()
    run = TSPNNGInput()
    
    ax.set_xlim([xlim[0], xlim[1]])
    ax.set_ylim([ylim[0], ylim[1]])
    ax.set_aspect(1.0)
    ax.set_xticks([])
    ax.set_yticks([])
      
    mouseClick   = wrapperEnterRunPointsHandler(fig,ax, run)
    fig.canvas.mpl_connect('button_press_event' , mouseClick )
      
    keyPress     = wrapperkeyPressHandler(fig,ax, run)
    fig.canvas.mpl_connect('key_press_event', keyPress   )
    plt.show()
xlim, ylim = [0,1], [0,1]
def wrapperEnterRunPointsHandler(fig, ax, run):
    def _enterPointsHandler(event):
        if event.name      == 'button_press_event'     and \
           (event.button   == 1)                       and \
            event.dblclick == True                     and \
            event.xdata  != None                       and \
            event.ydata  != None:

             newPoint = np.asarray([event.xdata, event.ydata])
             run.points.append( newPoint  )
             print("You inserted ", newPoint)

             patchSize  = (xlim[1]-xlim[0])/130.0
                   
             ax.clear()

             for pt in run.points:
                  ax.add_patch( mpl.patches.Circle( pt, radius = patchSize,
                                                    facecolor='blue', edgecolor='black'  ))

             ax.set_title('Points Inserted: ' + str(len(run.points)), \
                           fontdict={'fontsize':25})
             applyAxCorrection(ax)
             fig.canvas.draw()

    return _enterPointsHandler
def wrapperkeyPressHandler(fig,ax, run): 
       def _keyPressHandler(event):
               if event.key in ['n', 'N', 'u', 'U']: 
                     numpts = int(input("\nHow many points should I generate?: ")) 
                     run.clearAllStates()
                     ax.cla()
                     applyAxCorrection(ax)

                     ax.set_xticks([])
                     ax.set_yticks([])
                     fig.texts = []
                                      
                     if event.key in ['n', 'N']: 
                             run.points = non_uniform_points(numpts)
                     else : 
                             run.points = uniform_points(numpts)

                     patchSize  = (xlim[1]-xlim[0])/140.0

                     for site in run.points:      
                         ax.add_patch(mpl.patches.Circle(site, radius = patchSize, \
                                      facecolor='blue',edgecolor='black' ))

                     ax.set_title('Points generated: ' + str(len(run.points)), fontdict={'fontsize':25})
                     fig.canvas.draw()                   
               elif event.key in ['t' or 'T']:
                     tsp_graph = get_concorde_tsp_graph(run.points)
                     graph_fns = [(get_delaunay_tri_graph, 'Delaunay Triangulation (D)'), \
                                  (get_mst_graph         , 'Minimum Spanning Tree (M)'), \
                                  (get_onion_graph       , 'Onion'),\
                                  (get_gabriel_graph     , 'Gabriel'),\
                                  (get_urquhart_graph    , 'Urquhart') ]

                     from functools import partial
                     for k in range(1,5): 
                        graph_fns.append((partial(get_knng_graph, k=k), str(k)+'_NNG'))

                     tbl             = PrettyTable()
                     tbl.field_names = ["Spanning Graph (G)", "G", "G \cap T", "T", "(G \cap T)/T"]
                     num_tsp_edges   = len(tsp_graph.edges)

                     for ctr, (fn_body, fn_name) in zip(range(1,1+len(graph_fns)), graph_fns):
                          geometric_graph = fn_body(run.points)
                          num_graph_edges = len(geometric_graph.edges)
                          common_edges    = list_common_edges(tsp_graph, geometric_graph)
                          num_common_edges_with_tsp = len(common_edges)

                          tbl.add_row([fn_name,                 \
                                     num_graph_edges,           \
                                     num_common_edges_with_tsp, \
                                     num_tsp_edges,             \
                                     "{perc:3.2f}".format(perc=1e2*num_common_edges_with_tsp/num_tsp_edges)+ ' %' ])
                                     
                     print("Table of number of edges in indicated graph")
                     print(tbl)
                     render_graph(tsp_graph,fig,ax)
                     fig.canvas.draw()
               elif event.key in ['i', 'I']:                     
                     algo_str = input(Fore.YELLOW + "Enter code for the graph you need to span the points:\n" + Style.RESET_ALL  +\
                                          "(knng)   k-Nearest Neighbor Graph        \n"            +\
                                          "(mst)    Minimum Spanning Tree           \n"            +\
                                          "(onion)  Onion                           \n"            +\
                                          "(gab)    Gabriel Graph                 \n"            +\
                                          "(urq)    Urquhart Graph                    \n"            +\
                                          "(dt)     Delaunay Triangulation         \n"             +\
                                          "(conc)   TSP computed by the Concorde TSP library \n" +
                                          "(pytsp)  TSP computed by the pure Python TSP library \n")
                     algo_str = algo_str.lstrip()

                     if algo_str == 'knng':
                           k_str = input('===> What value of k do you want? ')
                           k     = int(k_str)
                           geometric_graph = get_knng_graph(run.points,k)

                     elif algo_str == 'mst':
                          geometric_graph = get_mst_graph(run.points)

                     elif algo_str == 'onion':
                          geometric_graph = get_onion_graph(run.points)

                     elif algo_str == 'gab':
                          geometric_graph = get_gabriel_graph(run.points)

                     elif algo_str == 'urq':
                          geometric_graph = get_urquhart_graph(run.points)

                     elif algo_str == 'dt':
                           geometric_graph = get_delaunay_tri_graph(run.points)

                     elif algo_str == 'conc':
                          geometric_graph = get_concorde_tsp_graph(run.points)

                     elif algo_str == 'pytsp':
                          geometric_graph = get_py_tsp_graph(run.points)

                     else:
                           print(Fore.YELLOW, "I did not recognize that option.", Style.RESET_ALL)
                           geometric_graph = None

                     tsp_graph = get_concorde_tsp_graph(run.points)
                     common_edges = list_common_edges( tsp_graph, geometric_graph)
                     print(Fore.YELLOW+"------------------------------------------------------------------------------"+Style.RESET_ALL)
                     print("Number of edges in " + algo_str + " graph                                :", len(geometric_graph.edges))
                     print(Fore.YELLOW, "\nNumber of edges in " + algo_str + " graph which are also in Concorde TSP :", len(common_edges))
                     print("Number of edges in " + "Concorde TSP                              :", len(tsp_graph.edges))
                     print("------------------------------------------------------------------------------", Style.RESET_ALL)


                     ax.set_title("Graph Type: " + geometric_graph.graph['type'] + '\n Number of nodes: ' + str(len(run.points)), fontdict={'fontsize':25})
                     render_graph(geometric_graph,fig,ax)
                     fig.canvas.draw()    
               elif event.key in ['x', 'X']:
                     print(Fore.GREEN, 'Removing network edges from canvas' ,Style.RESET_ALL)
                     ax.lines=[]
                     applyAxCorrection(ax)
                     fig.canvas.draw()
               elif event.key in ['c', 'C']: 
                     run.clearAllStates()
                     ax.cla()
                                                      
                     applyAxCorrection(ax)
                     ax.set_xticks([])
                     ax.set_yticks([])
                                                         
                     fig.texts = []
                     fig.canvas.draw()
       return _keyPressHandler
def applyAxCorrection(ax):
      ax.set_xlim([xlim[0], xlim[1]])
      ax.set_ylim([ylim[0], ylim[1]])
      ax.set_aspect(1.0)

def clearPatches(ax):
    for index , patch in zip(range(len(ax.patches)), ax.patches):
        if isinstance(patch, mpl.patches.Polygon) == True:
            patch.remove()
    ax.lines[:]=[]
    applyAxCorrection(ax)

def clearAxPolygonPatches(ax):

    for index , patch in zip(range(len(ax.patches)), ax.patches):
        if isinstance(patch, mpl.patches.Polygon) == True:
            patch.remove()
    ax.lines[:]=[]
    applyAxCorrection(ax)
def render_graph(G,fig,ax):
     if G is None:
            return
     edgecol = None
     if G.graph['type'] == 'mst':
          edgecol = 'g'
     elif G.graph['type'] == 'onion':
          edgecol = 'gray'
     elif G.graph['type'] == 'gabriel':
          edgecol = (153/255, 102/255, 255/255)
     elif G.graph['type'] == 'urq':
          edgecol = (255/255, 102/255, 153/255)
     elif G.graph['type'] in ['conc','pytsp']:
          edgecol = 'r'
     elif G.graph['type'] == 'dt':
          edgecol = 'b'
     elif G.graph['type'][-3:] == 'nng':
          edgecol = 'm'
     if G.graph['type'] not in ['conc', 'pytsp']:
          
          #for elt in list(G.nodes(data=True)):
          #     print(elt)

          for  (nidx1, nidx2) in G.edges:
              x1, y1 = G.nodes[nidx1]['coods']
              x2, y2 = G.nodes[nidx2]['coods']
              ax.plot([x1,x2],[y1,y2],'-', color=edgecol)
     else:
          from networkx.algorithms.traversal.depth_first_search import dfs_edges
          node_coods = []
          for (nidx1, nidx2) in dfs_edges(G):
                 node_coods.append(G.nodes[nidx1]['coods'])
                 node_coods.append(G.nodes[nidx2]['coods'])

          node_coods = np.asarray(node_coods)
          from matplotlib.patches import Polygon
          from matplotlib.collections import PatchCollection

          polygon = Polygon(node_coods, closed=True, \
                            facecolor=(255/255, 255/255, 102/255,0.5), \
                            edgecolor='k', linewidth=1)
          ax.add_patch(polygon)
          
     ax.axis('off') # turn off box surrounding plot
     fig.canvas.draw()
def get_knng_graph(points,k):
     from sklearn.neighbors import NearestNeighbors
     points     = np.array(points)
     coords     = [{"coods":pt} for pt in points]
     knng_graph = nx.Graph()
     knng_graph.add_nodes_from(zip(range(len(points)), coords))
     nbrs = NearestNeighbors(n_neighbors=(k+1), algorithm='ball_tree').fit(points)
     distances, indices = nbrs.kneighbors(points)
     edge_list = []

     for nbidxs in indices:
          nfix = nbidxs[0]
          edge_list.extend([(nfix,nvar) for nvar in nbidxs[1:]])

     knng_graph.add_edges_from(  edge_list  )
     knng_graph.graph['type']   = str(k)+'nng'
     knng_graph.graph['weight'] =  None # TODO, also edge weights for each edge!!!
     return knng_graph
def get_delaunay_tri_graph(points):
     from scipy.spatial import Delaunay
     points       = np.array(points)
     coords       = [{"coods":pt} for pt in points]
     tri          = Delaunay(points)
     deltri_graph = nx.Graph()

     deltri_graph.add_nodes_from(zip(range(len(points)), coords))

     edge_list = []
     for (i,j,k) in tri.simplices:
         edge_list.extend([(i,j),(j,k),(k,i)])    
     deltri_graph.add_edges_from(  edge_list  )
     
     total_weight_of_edges = 0.0
     for edge in deltri_graph.edges:
           n1, n2 = edge
           pt1 = deltri_graph.nodes[n1]['coods'] 
           pt2 = deltri_graph.nodes[n2]['coods']
           edge_wt = np.linalg.norm(pt1-pt2)

           deltri_graph.edges[n1,n2]['weight'] = edge_wt
           total_weight_of_edges = total_weight_of_edges + edge_wt 
     
     deltri_graph.graph['weight'] = total_weight_of_edges
     deltri_graph.graph['type']   = 'dt'

     return deltri_graph

def get_mst_graph(points):

     points = np.array(points)
     deltri_graph = get_delaunay_tri_graph(points)
     mst_graph = nx.algorithms.tree.mst.minimum_spanning_tree(deltri_graph, \
                                                              algorithm='kruskal')
     mst_graph.graph['type']   = 'mst'
     return mst_graph

def get_onion_graph(points):
     from scipy.spatial import ConvexHull
     points      = np.asarray(points)     
     points_tmp  = points.copy()
     numpts      = len(points)
     onion_graph = nx.Graph()
     numpts_proc = -1

     def circular_edge_zip(xs):
         xs = list(xs) # in the event that xs is of the zip or range type 
         if len(xs) in [0,1] :
              zipl = []
         elif len(xs) == 2 :
              zipl = [(xs[0],xs[1])]
         else:
              zipl = list(zip(xs,xs[1:]+xs[:1]))
         return zipl

     while len(points_tmp) >= 3:
           hull            = ConvexHull(points_tmp)
           pts_on_hull     = [points_tmp[i] for i in hull.vertices]
           coords          = [{"coods":pt} for pt in pts_on_hull]
           new_node_idxs   = range(numpts_proc+1, numpts_proc+len(hull.vertices)+1)
           onion_graph.add_nodes_from(zip(new_node_idxs, coords))
           onion_graph.add_edges_from(circular_edge_zip(new_node_idxs))
           numpts_proc  = numpts_proc + len(hull.vertices)
           rem_pts_idxs = list(set(range(len(points_tmp)))-set(hull.vertices)) 
           points_tmp   = [ points_tmp[idx] for idx in rem_pts_idxs ]
           coords       = [{"coods":pt} for pt in points]

     if len(points_tmp) == 2:
          p, l = numpts_proc+1, numpts_proc+2
          onion_graph.add_node(p)
          onion_graph.add_node(l)
          onion_graph.nodes[p]['coods'] = points_tmp[0]
          onion_graph.nodes[l]['coods'] = points_tmp[1]
          onion_graph.add_edge(p,l)
     elif len(points_tmp) == 1:
          l = numpts_proc+1 
          onion_graph.add_node(l)
          onion_graph.nodes[l]['cood'] = points_tmp[0]
 
     onion_graph.graph['type'] = 'onion'
     return onion_graph
def get_gabriel_graph(points):
    from scipy.spatial import Voronoi
    
    def ccw(A,B,C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    def intersect(A,B,C,D):
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
    
    points = np.array(points)
    coords = [{"coods":pt} for pt in points]
    gabriel = nx.Graph()
    gabriel.add_nodes_from(zip(range(len(points)), coords))

    vor = Voronoi(points)
    diam = np.sqrt( (vor.min_bound[0] - vor.max_bound[0])**2 +
                    (vor.min_bound[1] - vor.max_bound[1])**2 )
    center = vor.points.mean(axis=0)
    
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        if v2<0:
            v1, v2 = v2, v1
        if v1 >= 0: # bounded Voronoi edge
            if intersect(vor.points[p1], vor.points[p2],
                         vor.vertices[v1], vor.vertices[v2]):
                gabriel.add_edge(p1,p2)
            continue
        else: # unbounded Voronoi edge
            # compute "unbounded" edge
            p1p2 = vor.points[p2] - vor.points[p1]
            p1p2 /= np.linalg.norm(p1p2)
            normal = np.array([-p1p2[1], p1p2[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = vor.vertices[v2] + direction * diam
            
            if intersect(vor.points[p1], vor.points[p2],
                         vor.vertices[v2], far_point):
                gabriel.add_edge(p1,p2)
    gabriel.graph['type'] = 'gabriel'
    return gabriel
def get_urquhart_graph(points):
     from scipy.spatial import Delaunay
     points       = np.array(points)
     coords       = [{"coods":pt} for pt in points]
     tri          = Delaunay(points)
     urq_graph = nx.Graph()

     urq_graph.add_nodes_from(zip(range(len(points)), coords))

     edge_list = []
     longest_edge_list = []
     for (i,j,k) in tri.simplices:
         edges = [(i,j),(j,k),(k,i)]
         norms = [np.linalg.norm(points[j]-points[i]),
                  np.linalg.norm(points[k]-points[j]),
                  np.linalg.norm(points[i]-points[k])]
         zipped = zip(edges,norms)
         sorted_edges = sorted(zipped, key = lambda t: t[1])
         longest_edge_list.append(sorted_edges[2][0])
         edge_list.extend([(i,j),(j,k),(k,i)])    
     urq_graph.add_edges_from( edge_list )
     urq_graph.remove_edges_from( longest_edge_list )
     
     total_weight_of_edges = 0.0
     for edge in urq_graph.edges:
           n1, n2 = edge
           pt1 = urq_graph.nodes[n1]['coods'] 
           pt2 = urq_graph.nodes[n2]['coods']
           edge_wt = np.linalg.norm(pt1-pt2)

           urq_graph.edges[n1,n2]['weight'] = edge_wt
           total_weight_of_edges = total_weight_of_edges + edge_wt 
     
     urq_graph.graph['weight'] = total_weight_of_edges
     urq_graph.graph['type']   = 'urq'

     return urq_graph
def get_py_tsp_graph(points):
     import tsp
     points = np.array(points)
     coords = [{"coods":pt} for pt in points]
     t              = tsp.tsp(points)
     idxs_along_tsp = t[1]
     tsp_graph      = nx.Graph()

     tsp_graph.add_nodes_from(zip(range(len(points)), coords))
     edge_list = list(zip(idxs_along_tsp, idxs_along_tsp[1:])) + \
                       [(idxs_along_tsp[-1],idxs_along_tsp[0])]
     tsp_graph.add_edges_from(  edge_list  )
     total_weight_of_edges = 0.0
     for edge in tsp_graph.edges:

           n1, n2 = edge
           pt1 = tsp_graph.nodes[n1]['coods'] 
           pt2 = tsp_graph.nodes[n2]['coods']
           edge_wt = np.linalg.norm(pt1-pt2)

           tsp_graph.edges[n1,n2]['weight'] = edge_wt
           total_weight_of_edges = total_weight_of_edges + edge_wt 
     tsp_graph.graph['weight'] = total_weight_of_edges
     tsp_graph.graph['type']   = 'pytsp'     
     return tsp_graph
def get_concorde_tsp_graph(points, scaling_factor=1000):
     from concorde.tsp import TSPSolver
     points = np.array(points)
     coords = [{"coods":pt} for pt in points]

     xs = [int(scaling_factor*pt[0]) for pt in points]
     ys = [int(scaling_factor*pt[1]) for pt in points]
     solver = TSPSolver.from_data(xs, ys, norm='EUC_2D', name=None)
     print(Fore.GREEN)
     solution = solver.solve()
     print(Style.RESET_ALL)

     concorde_tsp_graph=nx.Graph()
          
     idxs_along_tsp = solution.tour
     concorde_tsp_graph.add_nodes_from(zip(range(len(points)), coords))
     edge_list = list(zip(idxs_along_tsp, idxs_along_tsp[1:])) + \
                    [(idxs_along_tsp[-1],idxs_along_tsp[0])]
     concorde_tsp_graph.add_edges_from(  edge_list  )

     concorde_tsp_graph.graph['type']   = 'conc'
     concorde_tsp_graph.graph['found_tour_p'] = solution.found_tour
     concorde_tsp_graph.graph['weight'] = None ### TODO!! 
     return concorde_tsp_graph
def edge_equal_p(e1,e2):
     e1 = sorted(list(e1))
     e2 = sorted(list(e2))
     return (e1==e2)
def list_common_edges(g1, g2):
     common_edges = []
     for e1 in g1.edges:
          for e2 in g2.edges:
             if  edge_equal_p(e1,e2):
                  common_edges.append(e1)
     return common_edges
def graphs_intersect_p(g1,g2):
     flag = False
     if list_common_edges(g1,g2):     
          flag = True 
     return flag
def expt_intersection_behavior():
     expt_name           = 'expt_intersection_behavior'
     expt_plot_file_extn = '.pdf'

     ptsmin   = 10
     ptsmax   = 200
     skipval  = 10

     numrunsper    = 20
     cols_1nng     = {}
     cols_mst      = {}
     cols_gabriel      = {}
     cols_urquhart      = {}
     cols_delaunay = {}
     for numpts in range(ptsmin,ptsmax,skipval):
          cols_1nng[numpts]     = []
          cols_mst[numpts]      = []
          cols_gabriel[numpts]      = []
          cols_urquhart[numpts]      = []
          cols_delaunay[numpts] = []
          for runcount in range(numrunsper):
               pts            = uniform_points(numpts)
               nng1_graph     = get_knng_graph(pts,k=1)
               mst_graph      = get_mst_graph(pts)
               gabriel_graph  = get_gabriel_graph(pts)
               urquhart_graph = get_urquhart_graph(pts)
               del_graph     = get_delaunay_tri_graph(pts)
               conctsp_graph = get_concorde_tsp_graph(pts)
               
               cols_1nng[numpts].append(100*len(list_common_edges(nng1_graph,conctsp_graph))/len(conctsp_graph.edges) )
               cols_mst[numpts].append(100*len(list_common_edges(mst_graph,conctsp_graph)) /len(conctsp_graph.edges) )
               cols_gabriel[numpts].append(100*len(list_common_edges(gabriel_graph,conctsp_graph)) /len(conctsp_graph.edges) )
               cols_urquhart[numpts].append(100*len(list_common_edges(urquhart_graph,conctsp_graph)) /len(conctsp_graph.edges) )
               cols_delaunay[numpts].append(100*len(list_common_edges(del_graph,conctsp_graph)) /len(conctsp_graph.edges) )

     fig, ax = plt.subplots()
     ax.set_title(r"Intersection \% between Euc. 2D TSP and Various Graphs" "\n"  r"on Random Uniform points in $[0,1]^2$", fontdict={'fontsize':15})
     ax.set_xlim([ptsmin,ptsmax-skipval])
     ax.set_ylim([0,110])
     ax.set_xlabel("Number of points in point-cloud")
     ax.set_ylabel("Percentage")
     ax.set_xticks(range(ptsmin,ptsmax,skipval))
     
     def arithmetic_mean(nums):
          return sum(nums)/len(nums)
     

     cols_1nng_min = [ min(cols_1nng[key]) for key in sorted(cols_1nng)]
     cols_1nng_max = [ max(cols_1nng[key]) for key in sorted(cols_1nng)]
     cols_1nng_am  = np.asarray([ arithmetic_mean(cols_1nng[key]) for key in sorted(cols_1nng)])
     cols_1nng_std = np.asarray([ np.std(cols_1nng[key]) for key in sorted(cols_1nng)])
     
     cols_mst_min = [ min(cols_mst[key]) for key in sorted(cols_mst)]
     cols_mst_max = [ max(cols_mst[key]) for key in sorted(cols_mst)]
     cols_mst_am  = np.asarray([ arithmetic_mean(cols_mst[key]) for key in sorted(cols_mst)])
     cols_mst_std = np.asarray([ np.std(cols_mst[key]) for key in sorted(cols_mst)])
   
     cols_gabriel_min = [ min(cols_gabriel[key]) for key in sorted(cols_gabriel)]
     cols_gabriel_max = [ max(cols_gabriel[key]) for key in sorted(cols_gabriel)]
     cols_gabriel_am  = np.asarray([ arithmetic_mean(cols_gabriel[key]) for key in sorted(cols_gabriel)])
     cols_gabriel_std = np.asarray([ np.std(cols_gabriel[key]) for key in sorted(cols_gabriel)])
     
     cols_urquhart_min = [ min(cols_urquhart[key]) for key in sorted(cols_urquhart)]
     cols_urquhart_max = [ max(cols_urquhart[key]) for key in sorted(cols_urquhart)]
     cols_urquhart_am  = np.asarray([ arithmetic_mean(cols_urquhart[key]) for key in sorted(cols_urquhart)])
     cols_urquhart_std = np.asarray([ np.std(cols_urquhart[key]) for key in sorted(cols_urquhart)])
  
     cols_delaunay_min = [ min(cols_delaunay[key]) for key in sorted(cols_delaunay)]
     cols_delaunay_max = [ max(cols_delaunay[key]) for key in sorted(cols_delaunay)]
     cols_delaunay_am = np.asarray([ arithmetic_mean(cols_delaunay[key]) for key in sorted(cols_delaunay)])
     cols_delaunay_std = np.asarray([ np.std(cols_delaunay[key]) for key in sorted(cols_delaunay)])

     xs = range(ptsmin,ptsmax,skipval)

     # colors of lines corresponding to various graphs
     nng1col='r'
     mstcol='b'
     gabcol='g'
     urqcol='k'
     dtcol='m'
 
     ax.plot(xs,cols_delaunay_am,'o-', label='Delaunay',color=dtcol)
     ax.fill_between(xs, cols_delaunay_am-cols_delaunay_std, \
                         cols_delaunay_am+cols_delaunay_std ,  color=dtcol, alpha=0.3)     
  
     ax.plot(xs,cols_gabriel_am,'o-', label='Gabriel',color=gabcol)
     ax.fill_between(xs, cols_gabriel_am-cols_gabriel_std, \
                         cols_gabriel_am+cols_gabriel_std ,  color=gabcol, alpha=0.3)     

     ax.plot(xs,cols_urquhart_am,'o-', label='Urquhart',color=urqcol)
     ax.fill_between(xs, cols_urquhart_am-cols_urquhart_std, \
                         cols_urquhart_am+cols_urquhart_std ,  color=urqcol, alpha=0.3)     


     ax.plot(xs,cols_mst_am,'o-', label='MST',color=mstcol)
     ax.fill_between(xs, cols_mst_am-cols_mst_std, \
                         cols_mst_am+cols_mst_std ,  color=mstcol, alpha=0.3)     

     ax.plot(xs,cols_1nng_am,'o-', label='1-NNG',color=nng1col)
     ax.fill_between(xs, cols_1nng_am-cols_1nng_std, \
                         cols_1nng_am+cols_1nng_std ,  color=nng1col, alpha=0.3)     

     ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
     ax.grid(color='gray',linestyle='--',linewidth=0.5)
     plt.savefig(expt_name+expt_plot_file_extn, bbox_inches='tight')
     print("Plot File written to disk")
