
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import scipy as sp
import numpy as np
import random
from colorama import Fore, Back, Style

from scipy.optimize import minimize
from sklearn.cluster import KMeans
import argparse, sys, time

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
            event.xdata  != None                       and\ 
            event.ydata  != None:

             newPoint = (event.xdata, event.ydata)
             run.sites.append( newPoint  )
             patchSize  = (xlim[1]-xlim[0])/140.0
                   
             ax.add_patch( mpl.patches.Circle( newPoint, radius = patchSize,
                                               facecolor='blue', edgecolor='black'  ))
             ax.set_title('Points Inserted: ' + str(len(run.sites)), \
                           fontdict={'fontsize':40})
             applyAxCorrection(ax)
             fig.canvas.draw()

    return _enterPointsHandler
def wrapperkeyPressHandler(fig,ax, run): 
       def _keyPressHandler(event):
               if event.key in ['i', 'I']:                     
                     algo_str = raw_input(Fore.YELLOW                             +\
                                    "Enter code for the graph you need to span the points:\n"  +\
                                    "(dt)   Delaunay Triangulation           \n"  +\
                                    "(knng) k-Nearest Neighbor Graph           \n"  +\
                                    Style.RESET_ALL)
                     algo_str = algo_str.lstrip()

                     if algo_str == 'dt':
                           geometric_graph = pass
                           
                     elif algo_str == 'knng'
                           k_str = raw_input(Fore.YELLOW + '--> What value of k do you want? ')
                           k     = int(k_str)
                           geometric_graph = pass

                     else:
                           print("Unknown option! ")
                           sys.exit()

                     clearAxPolygonPatches(ax)
                     applyAxCorrection(ax)

                     ## --> Plot spanning graph onto ax
                     fig.canvas.draw()    
               elif event.key in ['n', 'N', 'u', 'U']: 
                     numpts = int(raw_input("\n" + Fore.YELLOW+\
                                            "How many points should I generate?: "+\
                                            Style.RESET_ALL)) 
                     run.clearAllStates()
                     ax.cla()
                                    
                     applyAxCorrection(ax)
                     ax.set_xticks([])
                     ax.set_yticks([])
                     fig.texts = []
                                      
                     if event.key in ['n', 'N']: 
                             run.sites = non_uniform_points(numpts)
                     else : 
                             run.sites = uniform_points(numpts)

                     patchSize  = (xlim[1]-xlim[0])/140.0

                     for site in run.sites:      
                         ax.add_patch(mpl.patches.Circle(site, radius = patchSize, \
                                      facecolor='blue',edgecolor='black' ))

                     ax.set_title('Points : ' + str(len(run.sites)), fontdict={'fontsize':40})
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



