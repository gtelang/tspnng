import tspnng
import yaml
import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Style

def main2():
    d="./sym-tsp-tsplib/instances/euclidean_instances_yaml/"
    picdir = './tsplib_euc2d_pictures_of_instances/'
    
    # delete picdir if it exists
    import shutil
    shutil.rmtree(picdir, ignore_errors=True)

    # create picdir 
    import os
    try:
        os.makedirs(picdir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    inst_names = tspnng.get_names_of_all_euclidean2D_instances(dirpath=d)
    for inst_name in inst_names:

        print(Fore.GREEN+"Reading " + inst_name, Style.RESET_ALL)
        with open(d+inst_name) as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            points = np.asarray(data['points'])
            
            # plot the data-points and save plot to disk
            xs = [pt[0] for pt in points]
            ys = [pt[1] for pt in points]
            
            fig, ax = plt.subplots()
            ax.set_aspect(1.0)
            ax.plot(xs,ys,'bs', markersize=1.5)
            ax.set_title(inst_name[:-4], fontdict={'fontsize':25})
            plt.savefig(picdir+inst_name[:-4]+'.png', dpi=250, bbox_inches='tight')
            plt.close('all')
            print('....rendering finished!')

def main1():
    tspnng.run_handler()

if __name__=='__main__':
    main1()
