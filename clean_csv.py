import os
import pandas as pd

input_folder = 'E:\\Ubuntu Backups\\Downloads\\Imran_new_dataset\\gpv-1\\'
output_folder = 'E:\\Ubuntu Backups\\Downloads\\Imran_new_dataset\\gpv-1\\Cleaned\\'  


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        df = pd.read_csv(input_path)

        df_cleaned = df.dropna(how='all')

        df_cleaned.to_csv(output_path, index=False)

        print(f"Cleaned {filename} and saved to {output_path}")

print("Cleaning completed.")
