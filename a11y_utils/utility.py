import re
import pandas as pd

def read_lines(file_name):
    file_object = open(file_name, 'r')
    lines = file_object.readlines()

    return lines


def make_a11y_questions():
    lines = read_lines('../a11y_objects_of_interest.txt')
    questions = []

    for item in lines:
        question_str = "Is there a/an " + item.strip() + " in the scene?\n"
        questions.append( question_str )

    file_object = open('../a11y_questions_of_interest.txt', 'w')
    file_object.writelines( questions )
    file_object.close()



def get_a11y_questions( a11y_filename ):
    lines = read_lines( a11y_filename )

    questions = []

    for line in lines:
        questions.append( line.strip() )

    return questions    



def get_video_identity_from_name( filename ): 
  
    temp = re.findall( r'\d+', filename )
    res = list( map( int, temp ) )

    ## for videos 17 and beyond
    if( len( res ) ) == 2:
        res.append( 0 )
        temp = res[1]
        
        #set the segment as 0
        res[1] = 1

        #set the frame
        res[2] = temp

    return {
        'video' : str( res[0] ),
        'segment' : str( res[1] ),
        'frame': str( res[2] ),
        'video_name': 
                "video-" +  str( res[0] ) 
            + "-segment-" + str( res[1] ) 
            + "-frame-" + str( res[2] )
        }




def join_csv_files_first_merge( filename1, filename2, merged_filename ):

    data1 = pd.read_csv( filename1 )
    data2 = pd.read_csv( filename2 )

    identity1 = get_video_identity_from_name( filename1 )
    identity2 = get_video_identity_from_name( filename2 )


    data1.columns.values[1] =  "Frame-" + identity1[ "frame" ]
    data2.columns.values[1] =  "Frame-" + identity2[ "frame" ]

    if 'Ground Truth' in data1: 
        data1.pop('Ground Truth')
    if 'Ground Truth' in data2:    
        data2.pop('Ground Truth')

    data1 = data1.drop_duplicates()
    data2 = data2.drop_duplicates()

    merged_file = pd.merge( data1, data2, on ='Question', how ='inner' )
    merged_file.to_csv( merged_filename, sep =',', index = False )

    

def join_csv_files( merged_filename, new_filename ):

    data1 = pd.read_csv( merged_filename )
    data2 = pd.read_csv( new_filename)

    identity2 = get_video_identity_from_name( new_filename )
    data2.columns.values[1] =  "Frame-" + identity2[ "frame" ]

    if 'Ground Truth' in data2:    
        data2.pop('Ground Truth')
    
    data1 = data1.drop_duplicates()
    data2 = data2.drop_duplicates()

    merged_file = pd.merge( data1, data2, on ='Question', how ='inner' )
    merged_file.to_csv( merged_filename, sep =',', index = False )   