
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import scipy as sp
import numpy as np
import random
import networkx as nx

from sklearn.cluster import KMeans
import argparse, sys, time
from colorama import Fore, Style, Back
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
   import yaml
   with open(dir_name + '/' + file_name, 'w') as outfile:
          yaml.dump( data, outfile, default_flow_style = False)
class TSPNNGInput:
      def __init__(self, points=[]):
          self.points            = points

      def clearAllStates (self):
          self.points = []

      def generate_geometric_graph(self,graph_code):
           pass
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

             patchSize  = (xlim[1]-xlim[0])/140.0
                   
             ax.add_patch( mpl.patches.Circle( newPoint, radius = patchSize,
                                               facecolor='blue', edgecolor='black'  ))
             ax.set_title('Points Inserted: ' + str(len(run.points)), \
                           fontdict={'fontsize':25})
             applyAxCorrection(ax)
             fig.canvas.draw()

    return _enterPointsHandler
def wrapperkeyPressHandler(fig,ax, run): 
       def _keyPressHandler(event):
               if event.key in ['i', 'I']:                     
                     algo_str = input(Fore.YELLOW + "Enter code for the graph you need to span the points:\n" + Style.RESET_ALL  +\
                                          "(knng) k-Nearest Neighbor Graph        \n"            +\
                                          "(mst)  Minimum Spanning Tree           \n"            +\
                                          "(dt)   Delaunay Triangulation         \n"             +\
                                          "(tsp)  TSP\n")
                     algo_str = algo_str.lstrip()

                     if algo_str == 'knng':
                           k_str = input('===> What value of k do you want? ')
                           k     = int(k_str)
                           geometric_graph = get_knng_graph(run.points,k)

                     elif algo_str == 'mst':
                          geometric_graph = get_mst_graph(run.points)

                     elif algo_str == 'dt':
                           geometric_graph = get_delaunay_tri_graph(run.points)

                     elif algo_str == 'tsp':
                          geometric_graph = get_tsp_graph(run.points)

                     else:
                           print(Fore.YELLOW, "I did not recognize that option.", Style.RESET_ALL)
                           geometric_graph = None

                     render_graph(geometric_graph,fig,ax)
                     fig.canvas.draw()    
               elif event.key in ['n', 'N', 'u', 'U']: 
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

                     ax.set_title('Points : ' + str(len(run.points)), fontdict={'fontsize':40})
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
def render_graph(geometric_graph,fig,ax):
     if geometric_graph is None:
            return

     t = np.arange(0.0, 2.0, 0.01)
     s = 1 + np.sin(2 * np.pi * t)
     ax.plot(t, s)
     fig.canvas.draw()

def get_knng_graph(points,k):
     points = np.array(points)

     ### --> Make the graph here
     knng_graph = None
     return knng_graph

def get_mst_graph(points):
     points = np.array(points)

     ### --> Make the graph here
     mst_graph = None
     return mst_graph

def get_delaunay_tri_graph(points):
     points = np.array(points)
     tri    = sp.spatial.Delaunay(points)

     ### --> Make the graph here
     deltri_graph = None
     return deltri_graph
def get_tsp_graph(points):

     import tsp
     points = np.array(points)
     coords = [{"coords":pt} for pt in points]
     t      = tsp.tsp(points)
     idxs_along_tsp = t[1]


     tsp_graph = nx.Graph()
     tsp_graph.add_nodes_from(zip(range(len(points)), coords))

     edge_list = zip(idxs_along_tsp, idxs_along_tsp[1:]) + [(idxs_along_tsp[-1],idxs_along_tsp[0])]
     tsp_graph.add_edges_from(  edge_list  )

     print(Fore.RED, list_edges(tsp_graph), Style.RESET_ALL)

     return tsp_graph


