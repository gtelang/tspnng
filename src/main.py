import tspnng
import yaml
import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Style
import sys

def main4():
    if len(sys.argv)>=2 and sys.argv[1] == '--file':
      filename = sys.argv[2]
      with open(filename, 'r') as stream:
         try:
            file_data = yaml.safe_load(stream)
            points    = [np.asarray(pt) for pt in file_data['points']]
            
            for pt in points:
                if pt[0]<0 or pt[0]>1 or pt[1]<0 or pt[1]>1:
                    print(Fore.RED,"One of your input points is ", pt)
                    print(Fore.RED,"Please adjust the coordinates of your points so that ALL of them lie inside [0,1]x[0,1]",Style.RESET_ALL)
                    sys.exit()

            print("\nPoints read from the input file are ")
            for pt in points:
                print(" ",pt)
            print("\nOpening interactive canvas with provided input points")
            tspnng.run_handler(points=points)

         except yaml.YAMLError as exc:
            print(exc)

    elif len(sys.argv)>=2 and sys.argv[1] == '--tsplibinstance':
        
        # this variable is added as a prefix to the filename so that the user does not have to type in a big-ass filename into the terminal
        tsplibfiledir='/home/gaurish/Dropbox/MyWiki/research-projects/TSPNNG/sym-tsp-tsplib/instances/euclidean_instances_yaml/'
        filename = sys.argv[2]
        filename = tsplibfiledir + filename + '.yml'
        with open(filename, 'r') as stream:
            try:
               file_data = yaml.load(stream,Loader=yaml.Loader)
               points=file_data['points']
               points=[np.asarray(pt) for pt in points]
               
               xmax = max([pt[0] for pt in points])
               ymax = max([pt[1] for pt in points])

               print("Scaling TSPLIB points to lie in unit-square")

               # The eps is for increasing the scaling factor slightly so 
               # that all points in the data-set falls inside unit box
               eps    = 1.0
               M      = max(xmax,ymax) + eps
               points = [pt/M for pt in points]

               print("\nPoints read from the input file are ")
               for pt in points:
                     print(" ",pt)
               print("\nOpening interactive canvas with scaled input points from TSPLIB")
               tspnng.run_handler(points=points)

            except yaml.YAMLError as exc:
               print(exc)

    elif len(sys.argv)>=2 and sys.argv[1] == '--interactive':
         tspnng.run_handler()
    else:
         print("Please run as one of:")
         print(Fore.GREEN)
         print("-->   python src/main.py --interactive")
         print("-->   python src/main.py --file <file.yaml>")
         print("-->   python src/main.py --tsplibinstance <instancename>")
         print(Style.RESET_ALL)
         sys.exit()
         

def main3():
    tspnng.expt_intersection_behavior()


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
