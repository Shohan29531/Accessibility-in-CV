This project is created directly from another repository (https://github.com/allenai/gpv-1/)

# Running the Project

1. 
A Clean installation of Ubuntu 22.04 is preferred.
2. 
Install git (sudo apt install git-all).
3. 
Intsall NVIDIA and CUDA drivers from the "Software & Updates" app (I used v525 Proprietary), 
DO NOT PERFORM THIS INSTALLATION ANY OTHER WAY (FROM TERMINAL OR SOMETHING)
Restart the computer

4. 
Install Anaconda 5.3.1 (you can find the version here. Download the file and then use bash to install.) 
(DO NOT USE ANY OTHER VERSION)
Restart all of your Terminals

    
5.
Change your directory to where you wish to download/place your GPV project.  
git clone https://github.com/allenai/gpv-1/

6. 
conda create -n gpv python=3.6 -y
conda activate gpv


7.
conda install -c pytorch pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.2.89 -y
conda install scikit-image=0.17.2 
conda install spacy=2.3.2 
pip install spacy-lookups-data=0.3.0

conda install -c cyclus java-jdk=8.45.14 -y
conda install h5py=2.10.0 -y

python -m spacy download en_core_web_sm


8.
pip install -r requirements.txt 
(IF THIS ONE FAILS AT ANY POINT, INSTALL ALL THE MODULES IN THE REQUIREMNETS.TXT FILE ONE-BY-ONE)



9.
Decide two asbolute paths for <data_dir> and <output_dir>.

For example, if you decide your <data_dir> to be ‘home/user/Desktop/gpv/data_gpv/’, use ‘home/user/Desktop/gpv/data_gpv/’ in place of <data_dir> for all the following commands. Same goes for <output_dir>.

(MAKE SURE THE DIRECTORY <data_dir> EXISTS BEFORE GOING FORWARD)

bash setup_data.sh <data_dir> 
(THIS IS GOING TO TAKE A WHILE, YOU CAN COMMENT OUT THE LAST LINE IN THE setup_data.sh FILE TO SAVE SOME SPACE AND TIME)



10.
(MAKE SURE THE DIRECTORIES <output_dir>/coco/ckpts/ and <output_dir>/coco_sce/ckpts/ EXIST BEFORE GOING FORWARD)

wget https://ai2-prior-gpv.s3-us-west-2.amazonaws.com/public/trained_models/gpv_all_original_split/ckpts/model.pth -P <output_dir>/coco/ckpts/

wget https://ai2-prior-gpv.s3-us-west-2.amazonaws.com/public/trained_models/gpv_all_gpv_split/ckpts/model.pth -P <output_dir>/coco_sce/ckpts/



11.
update your <data_dir> and <output_dir> addresses in four files inside the project (they are invalid as of now):

1. configs/exp/gpv_ft.yaml
2. configs/exp/gpv_inference_cmdline.yaml
3. configs/exp/gpv_inference.yaml
4. configs/exp/gpv.yaml


12.
Open the inference.ipynb file and copy all the codes (in the order they appear in the .ipynb file) into a python file (say, inference_notebook.py)


13.
python inference_notebook.py 
(this should generate outputs using the inputs of the street and horse examples as shown in the python notebook)
