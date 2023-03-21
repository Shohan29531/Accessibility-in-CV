import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


new_root_dir = "/home/touhid/Downloads/acss_videos_elena_outputs_by_group/"
file_name = "merged.csv"

df = pd.read_csv( new_root_dir + file_name )

sns.set()

df = df.set_index('Question')
plt.figure(figsize=(25, 15))
sns.heatmap(df)
plt.show()