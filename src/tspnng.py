
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
import shutil

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

def multimodal_points(numpts, nummodes=4, sigma=0.05):
     
     modes  = sp.rand(nummodes, 2).tolist()
     points = [] 
     for mode in modes:
          mx, my = mode
          samp_x = np.random.normal(mx, sigma, int(numpts/nummodes))
          samp_y = np.random.normal(my, sigma, int(numpts/nummodes))
          points.extend([np.asarray(pt) for pt in zip(samp_x, samp_y)])
          
     num_points_rem = numpts%nummodes
     if num_points_rem != 0:
          xrs = np.random.rand(num_points_rem)
          yrs = np.random.rand(num_points_rem)
          points.extend([np.asarray(pt) for pt in zip(xrs,yrs)])

     points = shift_and_scale_to_unit_square(points)
     return points
def shift_and_scale_to_unit_square(points):
     
     # make all coordinates positive by shifting origin
     points = [np.asarray(pt) for pt in points]
     min_x  = min([x for (x,_) in points])
     min_y  = min([y for (_,y) in points])
     m      = min(min_x, min_y)
     origin = np.asarray([m,m])

     # scale to unit-square
     points = [pt - origin for pt in points]
     max_x  = max([x for (x,_) in points])
     max_y  = max([y for (_,y) in points])
     scale  = max(max_x,max_y)
     points = [pt/scale for pt in points]

     return points
def concentric_circular_points(numpts, numrings):
     numpts_per_ring = int(numpts/numrings)
     points          = []
     center          = np.asarray([0.5,0.5])
     for ring in range(numrings):
          radius = (ring+1)*0.5/(numrings+1)
          print("Radius computed is ", radius)
         
          angles = [idx * 2*np.pi/numpts_per_ring for idx in range(numpts_per_ring)]
          xs     = [center[0] + radius * np.cos(theta) for theta in angles ]
          ys     = [center[1] + radius * np.sin(theta) for theta in angles ]
          points.extend([np.asarray(pt) for pt in zip(xs,ys)])
     
     num_points_rem = numpts%numrings
     if num_points_rem != 0:
          xrs = np.random.rand(num_points_rem)
          yrs = np.random.rand(num_points_rem)
          points.extend([np.asarray(pt) for pt in zip(xrs,yrs)])

     return points
def rectangular_grid_points(numpts, numrows):
     numcols = int(numpts/numrows)
     
     points = []
     for i in range(numrows):
         for j in range(numcols):
              print([i,j])
              points.append(np.asarray([j,i]))

     points = shift_and_scale_to_unit_square(points)

     num_points_rem = numpts-numrows*numcols
     if num_points_rem != 0:
          xrs = np.random.rand(num_points_rem)
          yrs = np.random.rand(num_points_rem)
          points.extend([np.asarray(pt) for pt in zip(xrs,yrs)])
     
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
def length_poly_chain(points, cycle_p=False):

     points     = [np.asarray(pt) for pt in points]
     seglengths = [np.linalg.norm(p1-p2) for (p1,p2) in zip(points, points[1:])]

     if cycle_p :
          seglengths.append(np.linalg.norm(points[-1]-points[0]))

     return sum(seglengths)
def error_intervals(xss):
     means           = [np.mean(xs) for xs in xss]
     stds            = [np.std(xs)  for xs in xss]
     error_intervals =  [ (means[i]-stds[i], \
                           means[i]        , \
                           means[i]+stds[i])   for i in len(xss)]
     return error_intervals
def exists_disp_vertex_cover_more_than_bin_guess(graph,points,d):

     from satispy import Variable, Cnf
     from satispy.solver import Minisat

     numpts = len(points)
     exp    = Cnf()

     xs = []
     for idx in range(numpts):
         xs.append(Variable( 'x_'+str(idx) ))
     print("     Created Variables....")
     for (i,j) in  list(graph.edges):
           c = Cnf()
           c = xs[i] | xs[j]
           exp &= c
     print("     Created Edge Clauses...")
     for i in range(numpts):
         for j in range(i+1, numpts):
             if np.linalg.norm(points[i]-points[j]) < d:
                    c = Cnf()
                    c = -xs[i] | -xs[j]
                    exp &= c
     print("     Created All Pair Clauses.....")
     solver = Minisat()
     solution = solver.solve(exp)

     if solution.success:
             print(Fore.CYAN, "    2-SAT Solution found!", Style.RESET_ALL)
             return (True, [idx for idx in range(numpts) if solution[xs[idx]] == True])
     else: 
              print(Fore.RED, "    2-SAT Solution cannot be found", Style.RESET_ALL)
              return (False, [])
def get_max_disp_vc_nodes(graph, points):

      tol               = 1e-6     # configurable
      bin_low, bin_high = 0.0, 2.0 # initial interval limits for binary search
      old_disp_idxs     = range(len(points)) 

      while abs(bin_high - bin_low) > tol:

          assert bin_high >= bin_low 
          bin_guess = (bin_high + bin_low)/2.0
          
          print ("\nBinary search Interval Limits are ", "[", bin_low, " , ", bin_high, "]")
          print ("Guessed dispersion value currently is ", bin_guess)
          print ("Checking for vertex cover with ", bin_guess, " dispersion.....")

          [exist_flag_p, disp_idxs] = exists_disp_vertex_cover_more_than_bin_guess(graph,points,bin_guess)

          if exist_flag_p:
                bin_low       = bin_guess
                old_disp_idxs = disp_idxs
          else:
                bin_high = bin_guess 

      return (old_disp_idxs, bin_guess)
def list_points_in_node_order(graph):
     return [graph.nodes[idx]['coods'] for idx in range(len(graph.nodes))]
class Forest:
    def __init__(self,rootNodes):
         self.rootNodes = rootNodes

class Node:
    def __init__(self, point, children=[]):
          self.point    = np.asarray(point)
          self.children = children
def render_max_disp_graph_hierarchy(points, graphfn, stopnum=10):
     
          dirName = './max_disp_vc_graphs/'
          shutil.rmtree(dirName, ignore_errors=True)
     
          try:
              os.mkdir(dirName)
              print("Directory " , dirName ,  " Created ") 
          except FileExistsError:
               print("Directory " , dirName ,  " already exists")
               sys.exit()    


          points          = np.asarray(points)
          original_points = points.copy()
          original_numpts = len(original_points)

          filenum = 0
          
          fig, ax = plt.subplots()

          ax.set_aspect(aspect=1.0)
          ax.set_xlim([0.0,1.0])
          ax.set_ylim([0.0,1.0])

          plt.cla()
          ax.set_title("Number of red points: " + str(len(points)))
          render_graph(graphfn(points),fig,ax)
          ax.plot(points[:,0], points[:,1], 'o', markerfacecolor='r', markersize=10)
          plt.savefig(dirName+'myplot_' + str(filenum).zfill(4) + '.png', bbox_inches='tight', dpi=200)
          

          while len(points) >= stopnum :
              print("ITERATION ", filenum)
              filenum += 1

              plt.cla()
              plt.grid(linestyle='--')
              ax.set_aspect(aspect=1.0)
              ax.scatter(original_points[:,0], original_points[:,1],alpha=0.2)

              mygraph                       = graphfn(points)     
              points                        = list_points_in_node_order(mygraph)     
              max_disp_vc_nodes, dispersion = get_max_disp_vc_nodes(mygraph, points)    
              points                        = np.asarray([points[idx] for idx in max_disp_vc_nodes])

              render_graph(graphfn(points),fig,ax)
              ax.plot(points[:,0], points[:,1], 'o', markerfacecolor='r', markersize=10)
              ax.set_title("Number of red points: " + str(len(points)))
              plt.savefig(dirName+'myplot_' + str(filenum).zfill(4) + '.png', bbox_inches='tight', dpi=200)
          
          return None
def build_delaunay_forest(points,stopnum=10):
     from sklearn.neighbors import NearestNeighbors     
     points = np.asarray([np.asarray(pt) for pt in points])
     while len(points) >= stopnum :
              del_graph                     = get_delaunay_tri_graph(points)
              max_disp_vc_nodes, dispersion = get_max_disp_vc_nodes(del_graph, points) 
              vcpoints                      = np.asarray([points[idx] for idx in max_disp_vc_nodes])
              nbrs                          = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(points)
              distances, indices            = nbrs.kneighbors(vcpoints)
              print(distances)
              print(indices)
              print(len(vcpoints))
              points             = vcpoints

     sys.exit()
     return delaunay_forest
def get_tsp_incr_graph(points):

     points     = [np.asarray(pt) for pt in points]
     numpts     = len(points)
     
     print(Fore.YELLOW, "Closest Pair Calculation Started",  Style.RESET_ALL)
     dmin     = np.inf
     dminpair = None
     del_idxs = None
     for i in range(numpts):
         for j in range(i+1, numpts):
             dist = np.linalg.norm(points[i]-points[j])
             if dist < dmin:
                 dmin     = dist
                 dminpair = [points[i], points[j]]
                 del_idxs = [i,j]

     points_rem = [ points[i] for i in range(numpts) if i not in del_idxs ]     
 
     print(Fore.GREEN, "......Closest Pair Calculation Finished", Style.RESET_ALL)

     
     print(Fore.YELLOW, "Setting initial graph", Style.RESET_ALL)
     tsp_incr_graph = nx.Graph()
     tsp_incr_graph.add_nodes_from([
                                    (0,{'coods':dminpair[0]}),\
                                    (1,{'coods':dminpair[1]})
                                   ])
     tsp_incr_graph.add_edge(0, 1, weight=dmin)
     print(Fore.GREEN, ".......Initial graph set ", Style.RESET_ALL)
  
     print(Fore.YELLOW, "Beginning Incremental TSP calculation", Style.RESET_ALL)
     itcount = 0 

     while points_rem:

          print(Fore.YELLOW, "---> Iteration Count ",itcount, Style.RESET_ALL) ; itcount += 1; 
               

          ptidx_remove = None
          edge_remove  = None
     
          ins_cost_min = np.inf

          for edge in list(tsp_incr_graph.edges()):
               u, v = edge
               for ptidx in range( len(points_rem) ):
                   
                   ins_cost_ptidx = np.linalg.norm( tsp_incr_graph.nodes[u]['coods']-points_rem[ptidx]  ) +\
                                    np.linalg.norm( tsp_incr_graph.nodes[v]['coods']-points_rem[ptidx]  )   - tsp_incr_graph[u][v]['weight']
                   
                   if ins_cost_ptidx < ins_cost_min:
                       ins_cost_min = ins_cost_ptidx
                       edge_remove  = edge
                       ptidx_remove = ptidx
         
          ur, vr = edge_remove
          if len(list(tsp_incr_graph.edges)) > 1:
               tsp_incr_graph.remove_edge(ur,vr)

          N      = len( tsp_incr_graph.nodes )
          tsp_incr_graph.add_node(N, coods=points_rem[ptidx_remove])   
          tsp_incr_graph.add_edge(N, ur, weight = np.linalg.norm( tsp_incr_graph.nodes[ur]['coods']-points_rem[ptidx] )   )
          tsp_incr_graph.add_edge(N, vr, weight = np.linalg.norm( tsp_incr_graph.nodes[vr]['coods']-points_rem[ptidx] )   )

          points_rem.pop(ptidx_remove)
          print("TSP tour weight: ", tsp_incr_graph.size(weight="weight"))

     
     print(Fore.GREEN, ".......Finished Incremental TSP calculation", Style.RESET_ALL)
     assert( len(list(tsp_incr_graph.edges)) == len(points) )  # to ensure it is a cycle 
        
     tsp_incr_graph.graph['type']   = 'tspincr'
     tsp_incr_graph.graph['weight'] = tsp_incr_graph.size(weight="weight")
     return tsp_incr_graph
def run_handler(points=[]):
    fig, ax =  plt.subplots()
    run = TSPNNGInput(points=points)
    
    ax.set_xlim([xlim[0], xlim[1]])
    ax.set_ylim([ylim[0], ylim[1]])
    ax.set_aspect(1.0)
    ax.set_xticks([])
    ax.set_yticks([])
 
    patchSize  = (xlim[1]-xlim[0])/130.0

    for pt in run.points:
       ax.add_patch( mpl.patches.Circle( pt, radius = patchSize,
                           facecolor='blue', edgecolor='black'  ))

    ax.set_title('Points Inserted: ' + str(len(run.points)), \
                   fontdict={'fontsize':25})
    applyAxCorrection(ax)
    fig.canvas.draw()

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
               if event.key in ['n', 'N', 'u', 'U','m','M','o','O','g','G']: 
                     numpts = int(input("\nHow many points should I generate?: ")) 
                     run.clearAllStates()
                     ax.cla()
                     applyAxCorrection(ax)

                     ax.set_xticks([])
                     ax.set_yticks([])
                     fig.texts = []
                                      
                     if event.key in ['n', 'N']: 
                             run.points = non_uniform_points(numpts)
                     elif event.key in ['u', 'U'] : 
                             run.points = uniform_points(numpts)
                     elif event.key in ['m', 'M']:
                             nummodes   = int(input(Fore.YELLOW+"How many modes do you want in the distribution?"+Style.RESET_ALL))
                             sigma      = float(input(Fore.YELLOW+"What do you want the standard deviation of the local distribution around each mode to be?"+Style.RESET_ALL))
                             run.points = multimodal_points(numpts,nummodes=nummodes,sigma=sigma)
                     elif event.key in ['o', 'O']:
                             numrings   = int(input(Fore.YELLOW+"How many rings do you want?"+Style.RESET_ALL))
                             run.points = concentric_circular_points(numpts,numrings)
                     elif event.key in ['g', 'G']:
                             numrows    = int(input(Fore.YELLOW+"How many rows do you want?"+Style.RESET_ALL))
                             run.points = rectangular_grid_points(numpts,numrows)
                     else:
                            print("I did not understand that option. Please type one of `n`, `u`, `m`, `o`, `g`")

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
                     algo_str = input(Fore.YELLOW + "Enter code for the graph you need to span the points: \n" + Style.RESET_ALL  +\
                                          "(knng)      k-Nearest Neighbor Graph                            \n" +\
                                          "(mst)       Minimum Spanning Tree                               \n" +\
                                          "(onion)     Onion                                               \n" +\
                                          "(gab)       Gabriel Graph                                       \n" +\
                                          "(urq)       Urquhart Graph                                      \n" +\
                                          "(dt)        Delaunay Triangulation                              \n" +\
                                          "(bitonic)   Bitonic tour                                        \n" +\
                                          "(l1bitonic) L1 Bitonic tour                                     \n" +\
                                          "(tspincr)   Incremental TSP-approx                              \n" +\
                                          "(conc)      TSP computed by the Concorde TSP library            \n" +\
                                          "(pytsp)     TSP tour computed by the pure Python TSP library    \n" +\
                                          "(pypath)    TSP path computed by the pure Python TSP library    \n" +\
                                          "(l1pytsp)   L1 TSP tour computed by the pure Python TSP library \n" +\
                                          "(concorde)  Tour or path in any metric using Concorde           \n")
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

                     elif algo_str == 'tspincr':
                          geometric_graph = get_tsp_incr_graph(run.points)

                     elif algo_str == 'bitonic':
                          geometric_graph = get_bitonic_tour(run.points)

                     elif algo_str == 'l1bitonic':
                          geometric_graph = get_l1_bitonic_tour(run.points)

                     elif algo_str == 'pypath':
                          geometric_graph = get_pytsp_path(run.points)

                     elif algo_str == 'l1pytsp':
                          geometric_graph = get_L1_tsp_tour(run.points)

                     elif algo_str == 'concorde':
                          mode = input('Enter tour or path: ')
                          metric = int(input('Enter metric (inf for L_infty metric): '))
                          geometric_graph = get_tsp_graph(run.points, metric=metric, mode=mode)

                     elif algo_str in ['d','D']:

                          build_delaunay_forest(run.points,stopnum=10) 
                          sys.exit()

                     elif algo_str in ['v','V']:
                          render_max_disp_graph_hierarchy(run.points,graphfn=get_mst_graph ,stopnum=4)


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
     elif G.graph['type'] == 'bitonic':
          edgecol = (153/255, 0/255, 0/255)
     elif G.graph['type'] == 'pypath':
          edgecol = (255/255, 0/255, 0/255)
     elif G.graph['type'] == 'concorde':
          edgecol = (127/255 , 255/255, 212/255)
     if G.graph['type'] not in ['conc', 'pytsp','tspincr']:
          
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

          # mark the nodes
          xs = [pt[0] for pt in node_coods ]
          ys = [pt[1] for pt in node_coods ]
          ax.scatter(xs,ys)

          polygon = Polygon(node_coods, closed=True, \
                            facecolor=(255/255, 255/255, 102/255,0.4), \
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
            length = max(2*np.linalg.norm(vor.points[p1]-vor.vertices[v2]),
                         2*np.linalg.norm(vor.points[p2]-vor.vertices[v2]))
            far_point = vor.vertices[v2] + direction * length
            
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
def get_concorde_tsp_graph(points, scaling_factor=10000):
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

def get_pytsp_path(points):
     import tsp
     points = np.array(points)
     coords = [{"coods":pt} for pt in points]

     n = len(points)
     edge_weights = np.zeros((n+1,n+1))
     for i in range(0,n-1):
          for j in range(i+1,n):
               edge_weights[(i,j)] = np.linalg.norm(points[i]-points[j])
               edge_weights[(j,i)] = np.linalg.norm(points[i]-points[j])
     #for i in range(0,n):
     #     edge_weights[i,n] = 0
     #     edge_weights[n,i] = 0
     r = range(n+1)
     dist = {(i, j): edge_weights[i][j] for i in r for j in r}

     t              = tsp.tsp(r, dist)
     idxs_along_tsp = t[1]
     tsp_path       = nx.Graph()

     tsp_path.add_nodes_from(zip(range(len(points)), coords))
     dummy_node_ind = idxs_along_tsp.index(n)
     if dummy_node_ind == 0:
           path = idxs_along_tsp[1:]
     else:
           path = idxs_along_tsp[dummy_node_ind+1:] + \
                  idxs_along_tsp[:dummy_node_ind]
     for i in range(0,n-1):
           tsp_path.add_edge(path[i], path[i+1])

     total_weight_of_edges = 0.0
     for edge in tsp_path.edges:

           n1, n2 = edge
           pt1 = tsp_path.nodes[n1]['coods'] 
           pt2 = tsp_path.nodes[n2]['coods']
           edge_wt = np.linalg.norm(pt1-pt2)

           tsp_path.edges[n1,n2]['weight'] = edge_wt
           total_weight_of_edges = total_weight_of_edges + edge_wt 
     tsp_path.graph['weight'] = total_weight_of_edges
     tsp_path.graph['type']   = 'pypath'     
     return tsp_path

def get_L1_tsp_tour(points):
    import tsp
    points = np.array(points)
    coords = [{"coods":pt} for pt in points]
    tsp_L1_tour = nx.Graph()
    tsp_L1_tour.add_nodes_from(zip(range(len(points)), coords))
    n = len(points)
    
    edge_weights = np.zeros((n,n))
    for i in range(0,n-1):
        for j in range(i+1,n):
            edge_weights[(i,j)] = np.linalg.norm(points[i]-points[j], ord=1)
            edge_weights[(j,i)] = np.linalg.norm(points[i]-points[j], ord=1)
     #for i in range(0,n):
     #     edge_weights[i,n] = 0
     #     edge_weights[n,i] = 0
    r = range(n)
    dist = {(i, j): edge_weights[i][j] for i in r for j in r}

    t = tsp.tsp(r, dist)
    tsp_idxs = t[1]
    edge_list = [[tsp_idxs[i],tsp_idxs[i+1]] for i in range(-1,n-1)]
    tsp_L1_tour.add_edges_from(edge_list)

    total_weight_of_edges = 0.0
    for edge in tsp_L1_tour.edges:
        n1, n2 = edge
        pt1 = tsp_L1_tour.nodes[n1]['coods'] 
        pt2 = tsp_L1_tour.nodes[n2]['coods']
        edge_wt = np.linalg.norm(pt1-pt2)
        tsp_L1_tour.edges[n1,n2]['weight'] = edge_wt
        total_weight_of_edges = total_weight_of_edges + edge_wt 
    tsp_L1_tour.graph['weight'] = total_weight_of_edges
    tsp_L1_tour.graph['type']   = 'pytsp'     
    return tsp_L1_tour

def get_bitonic_tour(points):
    points = np.array(points)
    points = points[np.lexsort((points[:,1], points[:,0]))]
    coords = [{"coods":pt} for pt in points]
    bitonic_tour = nx.Graph()
    bitonic_tour.add_nodes_from(zip(range(len(points)), coords))
    n = len(points)
    
    min_lengths = [0, np.linalg.norm(points[0]-points[1])]
    partial_bitonic_path_edges = {1:[[1,0]]}
    for l in range(2,n):
        path_values = []
        for i in range(2,l+1):
            path_values.append(np.linalg.norm(points[l]-points[i-2]) + \
                               min_lengths[i-1] + \
                               sum( [np.linalg.norm(points[k]-points[k-1]) for k in range(i,l)] ) )
        path_lngth, idx = min((val, idx) for (idx, val) in enumerate(path_values))
        min_lengths = min_lengths + [path_lngth]
        partial_bitonic_path_edges[l] = partial_bitonic_path_edges[idx+1] + [[l,idx]] + \
                                    [[k-1,k] for k in range(idx+2,l)]
    bitonic_tour_edges = partial_bitonic_path_edges[n-1] + [[n-2,n-1]]
    bitonic_tour.add_edges_from(bitonic_tour_edges)

    total_weight_of_edges = 0.0
    for edge in bitonic_tour.edges:
          n1, n2 = edge
          pt1 = bitonic_tour.nodes[n1]['coods'] 
          pt2 = bitonic_tour.nodes[n2]['coods']
          edge_wt = np.linalg.norm(pt1-pt2)

          bitonic_tour.edges[n1,n2]['weight'] = edge_wt
          total_weight_of_edges = total_weight_of_edges + edge_wt 
    bitonic_tour.graph['weight'] = total_weight_of_edges
    bitonic_tour.graph['type']   = 'bitonic'

    return bitonic_tour

def get_l1_bitonic_tour(points):
    points = np.array(points)
    points = points[np.lexsort((points[:,1], points[:,0]))]
    coords = [{"coods":pt} for pt in points]
    bitonic_tour = nx.Graph()
    bitonic_tour.add_nodes_from(zip(range(len(points)), coords))
    n = len(points)
    
    min_lengths = [0, np.linalg.norm(points[0]-points[1], ord=1)]
    partial_bitonic_path_edges = {1:[[1,0]]}
    for l in range(2,n):
        path_values = []
        for i in range(2,l+1):
            path_values.append(np.linalg.norm(points[l]-points[i-2], ord=1) + \
                               min_lengths[i-1] + \
                               sum( [np.linalg.norm(points[k]-points[k-1], ord=1) for k in range(i,l)] ) )
        path_lngth, idx = min((val, idx) for (idx, val) in enumerate(path_values))
        min_lengths = min_lengths + [path_lngth]
        partial_bitonic_path_edges[l] = partial_bitonic_path_edges[idx+1] + [[l,idx]] + \
                                    [[k-1,k] for k in range(idx+2,l)]
    bitonic_tour_edges = partial_bitonic_path_edges[n-1] + [[n-2,n-1]]
    bitonic_tour.add_edges_from(bitonic_tour_edges)

    total_weight_of_edges = 0.0
    for edge in bitonic_tour.edges:
          n1, n2 = edge
          pt1 = bitonic_tour.nodes[n1]['coods'] 
          pt2 = bitonic_tour.nodes[n2]['coods']
          edge_wt = np.linalg.norm(pt1-pt2, ord=1)

          bitonic_tour.edges[n1,n2]['weight'] = edge_wt
          total_weight_of_edges = total_weight_of_edges + edge_wt 
    bitonic_tour.graph['weight'] = total_weight_of_edges
    bitonic_tour.graph['type']   = 'bitonic'

    return bitonic_tour


##### Generate Distance matrix ##### 
def generate_distance_matrix(pts, metric, mode='tour'):
    N   = len(pts)
    t = 0 if mode=='tour' else 1
    if metric=='inf':
        D   = np.zeros((N+t,N+t))
        for i in range(N):
            for j in range(N):
                D[i,j] = max(abs(pts[i]-pts[j]))
        return D
    else:
        D   = np.zeros((N+t,N+t))
        for i in range(N):
            for j in range(N):
                D[i,j] = np.linalg.norm(pts[i]-pts[j], ord=metric)
    return D

##### Write distance matrix to file #####
def write_distance_matrix_to_file(D,fname, dscale = 10000):
  with open(fname, 'w') as file:
    numrows, numcols = D.shape[0], D.shape[1]
    assert numrows == numcols, "Number of rows and columns in distance matrix must be equal, as matrix of distances is square"

    file.write('NAME: sampleinstance\n')
    file.write('TYPE: TSP\n')
    file.write('COMMENT: An explicit distance matrix between given set of points. Scaling factor (dscale) used for getting integer distance below = {dscale}\n'.format(dscale=dscale))
    file.write('DIMENSION: {dim}\n'.format(dim=numrows))
    file.write('EDGE_WEIGHT_TYPE: EXPLICIT\n')
    file.write('EDGE_WEIGHT_FORMAT: FULL_MATRIX\n')
    file.write('EDGE_WEIGHT_SECTION\n')
    
    # it is essential that distances are integers, otherwise concorde crashes. 
    # this is also the reason for the scaling factor that has been introduced, 
    # to make sure the original distances between the data are preserved as much as possible
    for i in range(numrows):
        for j in range(numcols):
            file.write('{value} \t'.format(value=int(dscale*D[i,j])) )
        file.write('\n')
    file.write('EOF') 

##### Solve with Concorde #####
def solve_tsp_from_file(fname):
    from concorde.tsp import TSPSolver
    solver   = TSPSolver.from_tspfile(fname)
    solution = solver.solve()
    return solution

def get_tsp_graph(points, metric=2, mode='tour'):
    import sys
    from concorde.tsp import TSPSolver
    points = np.array(points)
    n = len(points)
    coords = [{"coods":pt} for pt in points]
    tsp_graph = nx.Graph()

    # Solve correct problem
    if metric==2 and mode=='tour':
        xs = [int(scaling_factor*pt[0]) for pt in points]
        ys = [int(scaling_factor*pt[1]) for pt in points]
        solver = TSPSolver.from_data(xs, ys, norm='EUC_2D', name=None)
        print(Fore.GREEN)
        solution = solver.solve()
        print(Style.RESET_ALL)
    else:
        D = generate_distance_matrix(points, metric=metric, mode=mode)
        write_distance_matrix_to_file(D,fname='instance.tsp',  dscale = 10000)
        solution = solve_tsp_from_file('instance.tsp')

    # get solution inds and add nodes
    idxs_along_tsp = list(solution.tour)
    tsp_graph.add_nodes_from(zip(range(len(points)), coords))

    # add correct edges to graph
    if mode=='tour':
        edge_list = list(zip(idxs_along_tsp, idxs_along_tsp[1:])) + \
                    [(idxs_along_tsp[-1],idxs_along_tsp[0])]
        tsp_graph.add_edges_from(edge_list)
    elif mode=='path':
        dummy_node_ind = idxs_along_tsp.index(n)
        if dummy_node_ind == 0:
            path = idxs_along_tsp[1:]
        else:
            path = idxs_along_tsp[dummy_node_ind+1:] + \
                   idxs_along_tsp[:dummy_node_ind]
        for i in range(0,n-1):
            tsp_graph.add_edge(path[i], path[i+1])

    total_weight_of_edges = 0.0
    for edge in tsp_graph.edges:
        n1, n2 = edge
        pt1 = tsp_graph.nodes[n1]['coods'] 
        pt2 = tsp_graph.nodes[n2]['coods']
        edge_wt = np.linalg.norm(pt1-pt2)
        tsp_graph.edges[n1,n2]['weight'] = edge_wt
        total_weight_of_edges = total_weight_of_edges + edge_wt 
    tsp_graph.graph['weight'] = total_weight_of_edges
    tsp_graph.graph['type']   = 'concorde'
    return tsp_graph


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
def expt_tsplib_intersection_behavior():
 
     dirName = './tsplib-expts/'
     shutil.rmtree(dirName, ignore_errors=True)
     try:
         os.mkdir(dirName)
         print("Directory " , dirName ,  " Created ") 
     except FileExistsError:
         print("Directory " , dirName ,  " already exists")
    
     maxpts_bound        = 1100
     instance_file_names = get_names_of_all_euclidean2D_instances()

     f = open("./slides-beamer/slidecode.tex", "w")

     for fname in instance_file_names:

        inst_points = shift_and_scale_to_unit_square(tsplib_instance_points(fname))

        if len(inst_points) > maxpts_bound:
              print(Fore.RED, 'Skipping file', fname, ' which has ', len(inst_points) , '>', maxpts_bound , ' points', Style.RESET_ALL)
        else:
              print(fname , 'has ', len(inst_points), ' points')
              print('...Testing for intersections')
              tsp_graph = get_concorde_tsp_graph(inst_points)
              
              graph_fns = [(get_mst_graph         , 'MST'), \
                           (get_urquhart_graph    , 'Urquhart'), \
                           (get_gabriel_graph     , 'Gabriel'),\
                           (get_delaunay_tri_graph, 'Delaunay')]

              from functools import partial
              for k in range(1,4): 
                     graph_fns.append((partial(get_knng_graph, k=k), str(k)+'-NNG'))

              num_tsp_edges        = len(tsp_graph.edges)
              percentage_intersecn = []  
     
              print(Fore.YELLOW,'....COMPUTING INTERSECTIONS OF TSP WITH VARIOUS GRAPHS')
              for ctr, (fn_body, fn_name) in zip(range(1,1+len(graph_fns)), graph_fns):
                      print(Fore.CYAN+'---------> Intersecting with '+fn_name+Style.RESET_ALL)
                      geometric_graph           = fn_body(inst_points)
                      num_graph_edges           = len(geometric_graph.edges)
                      common_edges              = list_common_edges(tsp_graph, geometric_graph)
                      num_common_edges_with_tsp = len(common_edges)
                      percentage_intersecn.append(100*num_common_edges_with_tsp/num_tsp_edges)

              # plot the tour
              fig,ax = plt.subplots()
              ax.set_xlim([xlim[0], xlim[1]])
              ax.set_ylim([ylim[0], ylim[1]])
              ax.set_aspect(1.0)
              ax.set_xticks([])
              ax.set_yticks([])
 
              render_graph(tsp_graph,fig,ax)
              inst_name = fname[:-4]
              plt.savefig(dirName+inst_name+'-concorde-tsp.pdf',bbox_inches='tight')
              plt.close()
              
              # plot the intersection percentage bar-chart
              plt.style.use('ggplot')
              x              = [grtype for (_,grtype) in graph_fns]
              x_pos          = [i for i, _ in enumerate(x)]
     
              plt.bar(x_pos, percentage_intersecn, color='blue')
              plt.xlabel("Graph Type")
              plt.ylabel("Percentage of Intersections")
              plt.title("Percentage of intersections of TSP with various graphs on `" + fname[:-4] + "`")
 
              plt.xticks(x_pos, x)   
              plt.savefig(dirName+inst_name+'-intersection-chart.pdf',bbox_inches='tight')
                   
              # write slide code to file
             
              slidecode="%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
              slidecode+="\\begin{frame}{Intersections of Concorde TSP with TSPLIB instances of size $\\leq 1100$}\n"
              slidecode+="\\begin{figure}[ht]\n"
              slidecode+="  \\centering\n"
              slidecode+="  \\includegraphics[width=4cm]{../tsplib-expts/" + inst_name+"-concorde-tsp.pdf}\n"
              slidecode+="  \\includegraphics[width=8cm]{../tsplib-expts/" + inst_name+"-intersection-chart.pdf}\n"
              slidecode+="\\end{figure}\n"
              slidecode+="\\end{frame}\n"
              f.write(slidecode)


              plt.close()

     f.close()
def expt_intersection_behavior():
     expt_name           = 'expt_non_uniform_intersection_behavior'
     expt_plot_file_extn = '.pdf'

     ptsmin   = 10
     ptsmax   = 300
     skipval  = 20

     numrunsper    = 15
     cols_1nng     = {}
     cols_mst      = {}
     cols_gabriel  = {}
     cols_urquhart = {}
     cols_delaunay = {}
     for numpts in range(ptsmin,ptsmax,skipval):
          print(Fore.CYAN, "\n ========== \n Tackling numpts = ", numpts, "\n==============\n", Style.RESET_ALL)
          time.sleep(0.50)
          cols_1nng[numpts]     = []
          cols_mst[numpts]      = []
          cols_gabriel[numpts]      = []
          cols_urquhart[numpts]      = []
          cols_delaunay[numpts] = []
          for runcount in range(numrunsper):
               print(Fore.CYAN, "......Run number:", runcount, " for numpts = ", numpts, Style.RESET_ALL)
               time.sleep(0.25)
               #pts            = uniform_points(numpts)
               pts            = multimodal_points(numpts, nummodes=4, sigma=0.05)     
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
     ax.set_title(r"Intersection \% between Euc. 2D TSP and Various Graphs" "\n"  r"on Random Non Uniform points in $[0,1]^2$", fontdict={'fontsize':15})
     ax.set_xlim([ptsmin,ptsmax-skipval])
     ax.set_ylim([0,110])
     ax.set_xlabel("Number of points in point-cloud")
     ax.set_ylabel("Percentage")
     ax.set_xticks(np.arange(ptsmin,ptsmax,step=3*skipval))
     
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
