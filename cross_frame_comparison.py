from a11y_utils import utility 
import sys
import glob
import pandas as pd
from natsort import natsorted
import csv


root_dir = "/home/touhid/Downloads/acss_videos_elena_outputs/"
output_dir = "/home/touhid/Downloads/measures/"

csv_filenames = []
csv_filenames_full_path = []
for filename in glob.iglob(root_dir + '**/*.csv', recursive=True):
        csv_filenames_full_path.append( filename )
        tokens = filename.split('/')
        file = tokens[ len(tokens) - 1 ]
        csv_filenames.append( file )

csv_filenames = natsorted( csv_filenames )



video_id = 1
segment_id = 1

rows_in_comparison_file = []


with open( output_dir + 'measures.csv', 'a', newline='') as csvfile:
    csvwriter = csv.writer( csvfile, delimiter = ',')
    csvwriter.writerow(['video', 'segment', 'first-frame', 'seond-frame', 'similarity-score', 'yes-to-no-ratio']) 


for i in range( 0, len(csv_filenames) -1 ):
        
        first_file_name = csv_filenames[i]
        second_file_name = csv_filenames[i+1]

        identity_first = utility.get_video_identity_from_name( root_dir + first_file_name )   
        identity_second = utility.get_video_identity_from_name( root_dir + second_file_name )  

        if str( identity_first['video'] ) == '17':
               break 

        if identity_first['video'] != identity_second['video'] or identity_first['segment'] != identity_second['segment']:
                continue
        
        first_df = pd.read_csv( root_dir + first_file_name )

        first_df = first_df.drop( first_df[ 
        (first_df.Question == "What type of disability does the person have, if any?" ) |
        (first_df.Question == "What is the weather like in the scene?") |
        (first_df.Question == "How many males are there in the scene?") |
        (first_df.Question == "How many females are there in the scene?")
        ].index)


        second_df = pd.read_csv( root_dir + second_file_name )

        second_df = second_df.drop( second_df[ 
        (second_df.Question == "What type of disability does the person have, if any?" ) |
        (second_df.Question == "What is the weather like in the scene?") |
        (second_df.Question == "How many males are there in the scene?") |
        (second_df.Question == "How many females are there in the scene?")
        ].index)



        first_yes = 0
        second_yes = 0
        both_yes = 0

        first_no = 0
        second_no = 0
        both_no = 0

        same = 0

        total_questions = first_df.shape[0]

        for i in range( total_questions ):
            if str( first_df.iat[i, 1] ) == "1" and str( second_df.iat[i, 1] ) == "1":
                   both_yes += 1

            if str( first_df.iat[i, 1] ) == "1":
                   first_yes += 1

            if str( second_df.iat[i, 1] ) == "1":
                   second_yes += 1

            if str( first_df.iat[i, 1] ) == "0" and str( second_df.iat[i, 1] ) == "0":
                   both_no += 1

            if str( first_df.iat[i, 1] ) == "0":
                   first_no += 1

            if str( second_df.iat[i, 1] ) == "0":
                   second_no += 1 

            if str( first_df.iat[i, 1] ) == str( second_df.iat[i, 1] ):
                   same += 1                                
        
        metric = same / total_questions

        rows_in_comparison_file.append( [identity_first['frame'], identity_second['frame'], metric ] ) 

        # print( identity_first['frame'], identity_second['frame'], metric, both_yes / both_no )   

        with open( output_dir + 'measures.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer( csvfile, delimiter = ',')
            csvwriter.writerow([ identity_first['video'], identity_first['segment'],
                                 identity_first['frame'], identity_second['frame'],
                                 metric, both_yes / both_no ] ) 



                                  





        
        
                






        
        


    
