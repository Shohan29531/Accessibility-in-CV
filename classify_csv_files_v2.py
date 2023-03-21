from a11y_utils import utility 
import sys
import glob
import pandas as pd
from natsort import natsorted


root_dir = "/home/touhid/Downloads/acss_videos_elena_outputs/"

csv_filenames = []
csv_filenames_full_path = []
for filename in glob.iglob(root_dir + '**/*.csv', recursive=True):
        csv_filenames_full_path.append( filename )
        tokens = filename.split('/')
        file = tokens[ len(tokens) - 1 ]
        csv_filenames.append( file )


new_root_dir = "/home/touhid/Downloads/acss_videos_elena_outputs_by_group/"


csv_filenames = natsorted( csv_filenames )

# print (csv_filenames)


video_id = 1
segment_id = 1

while True:


    output_file = "video-" + str( video_id ) + "-segment-" + str( segment_id ) + ".csv"
    first_file_name = ""

    first_file_found = False
    first_merge_next = False

    for csv_filename in csv_filenames:
        identity = utility.get_video_identity_from_name( root_dir + csv_filename )

        if identity['video'] == str( video_id ) and identity['segment'] == str( segment_id ):
            if first_file_found == False:
                first_file_name = csv_filename
                first_file_found = True
                first_merge_next = True

            else:
                if first_merge_next:
                    first_merge_next = False
                    utility.join_csv_files_first_merge( 
                        root_dir + first_file_name, 
                        root_dir + csv_filename, 
                        new_root_dir + output_file)    
                else:
                    utility.join_csv_files( new_root_dir + output_file,
                                            root_dir + csv_filename )

    if first_file_found == False:
         video_id += 1
         if video_id == 17:
              break
         segment_id = 1
         continue
                                 
    df = pd.read_csv( new_root_dir + output_file )

    df = df.drop( df[ 
        (df.Question == "What type of disability does the person have, if any?" ) |
        (df.Question == "What is the weather like in the scene?") |
        (df.Question == "How many males are there in the scene?") |
        (df.Question == "How many females are there in the scene?")
        ].index)

    df.to_csv( new_root_dir + output_file, sep =',', index = False )  

    segment_id += 1









