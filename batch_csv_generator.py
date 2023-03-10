import glob

root_dir = '/home/touhid/Downloads/accss_videos_elena/'

i = 0
# root_dir needs a trailing slash (i.e. /root/dir/)
for filename in glob.iglob(root_dir + '**/*.jpeg', recursive=True):
     tokens = filename.split('/')
     file = tokens[ len(tokens) - 1 ]
     print ( i )
     i += 1
     print( file )

