from a11y_utils import utility 
import sys
import glob
import pandas as pd
from natsort import natsorted
from PIL import Image


root_dir = "/home/touhid/Desktop/GPV-1 outputs (names fixed)/"

all_image_files = []
all_image_files_full_path = []
for filename in glob.iglob(root_dir + '**/*.csv', recursive=True):
        all_image_files_full_path.append( filename )
        tokens = filename.split('/')
        file = tokens[ len(tokens) - 1 ]
        all_image_files.append( file )


new_root_dir = "/home/touhid/Desktop/GPV-1 outputs (names fixed)/"


# all_image_files = natsorted( all_image_files )

# print (csv_filenames)


for i in range( len( all_image_files) ):

    image_file = all_image_files[i]
    image_file_path = all_image_files_full_path[i]

    identity = utility.get_video_identity_from_name( image_file )

    output_file_name = identity['video_name'] + '.csv'

#     img = Image.open( image_file_path )

#     img = img.save( new_root_dir + output_file_name )

    # print(output_file_name, identity, image_file)

    csv_file = pd.read_csv( image_file_path )

#     csv_file.columns = ['Object', 'BLIP Prediction', 'Ground Truth']

    csv_file.drop('Ground Truth', inplace=True, axis=1)

    csv_file.to_csv( new_root_dir + output_file_name, sep =',', index = False )  

    print(i)






