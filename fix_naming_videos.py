from a11y_utils import utility 
import sys
import glob
import pandas as pd
from PIL import Image
import shutil
import os



root_dir = "/home/touhid/Downloads/accss_videos_elena/"

all_video_files = []
all_video_files_full_path = []
for filename in glob.iglob(root_dir + '**/*.mp4', recursive=True):
        all_video_files_full_path.append( filename )
        tokens = filename.split('/')
        file = tokens[ len(tokens) - 1 ]
        all_video_files.append( file )


new_root_dir = "/home/touhid/Desktop/Video Segments/"


# all_image_files = natsorted( all_image_files )

# print (csv_filenames)

print( len(all_video_files) )


for i in range( len( all_video_files) ):

    video_file = all_video_files[i]
    video_file_path = all_video_files_full_path[i]

    if os.path.isdir( video_file_path ):  
        continue 

    identity = utility.get_video_identity_from_name_only_segments( video_file )

    print( video_file )

    if identity == False:
         continue

    output_file_name = identity['video_name'] + '.mp4'

#     img = Image.open( image_file_path )

#     img = img.save( new_root_dir + output_file_name )

    # print(output_file_name, identity, image_file)

#     csv_file = pd.read_csv( video_file_path )

# #     csv_file.columns = ['Object', 'BLIP Prediction', 'Ground Truth']

#     csv_file.drop('Ground Truth', inplace=True, axis=1)

#     csv_file.to_csv( new_root_dir + output_file_name, sep =',', index = False )  


    shutil.copyfile( video_file_path, new_root_dir + output_file_name)

    print(i)






