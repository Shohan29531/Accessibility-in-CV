from a11y_utils import utility 
import sys
import glob
import pandas as pd
from natsort import natsorted


root_dir = "/home/touhid/Downloads/acss_videos_elena_outputs_by_group/"

csv_filenames = []
csv_filenames_full_path = []
for filename in glob.iglob(root_dir + '**/*.csv', recursive=True):
        csv_filenames_full_path.append( filename )
        tokens = filename.split('/')
        file = tokens[ len(tokens) - 1 ]
        csv_filenames.append( file )



for k in csv_filenames:
        print( 'Rscript heatmap.R ' + '"' + k + '"' )