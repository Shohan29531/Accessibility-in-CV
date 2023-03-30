import glob
import hydra
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf
import torch
import torchvision.transforms as T
import numpy as np
from nltk.tokenize.treebank import TreebankWordDetokenizer

import time
import math
import sys
import csv

from exp.gpv.models.gpv import GPV
from utils.detr_misc import collate_fn
from inference_util import *
from a11y_utils import utility



def preprocess(inputs,transforms):
    proc_inputs = []
    for img_path, query in inputs:
        img,_ = read_image(img_path,resize_image=True)
        proc_img = (255*img).astype(np.uint8)
        proc_img = transforms(proc_img).cuda()
        proc_inputs.append((proc_img,query))
    
    return collate_fn(proc_inputs)


def decode_outputs(outputs):
    detokenizer = TreebankWordDetokenizer()
    relevance = outputs['pred_relevance_logits'].softmax(-1).detach().cpu().numpy()
    pred_boxes = outputs['pred_boxes'].detach().cpu().numpy()
    topk_answers = torch.topk(outputs['answer_logits'][-1],k=1,dim=-1)
    topk_answer_ids = topk_answers.indices.detach().cpu().numpy()
    pred_answers = model.token_ids_to_words(topk_answer_ids[:,:,0])
    decoded_outputs = []
    for b in range(len(pred_answers)):
        scores, boxes = zip(*sorted(zip(
            relevance[b,:,0].tolist(),pred_boxes[b].tolist()),
            key=lambda x: x[0],reverse=True))
        scores = np.array(scores,dtype=np.float32)
        boxes = np.array(boxes,dtype=np.float32)
        answer = []
        for token in pred_answers[b]:
            if token in ['__stop__','__pad__']:
                break
            answer.append(token)
        answer = detokenizer.detokenize(answer)
        decoded_outputs.append({
            'answer': answer,
            'boxes': boxes,
            'relevance': scores})
    
    return decoded_outputs




if __name__=="__main__":

    start_time = time.time()

    root_dir = '/home/touhid/Downloads/accss_videos_elena/'

    input_names = []
    input_names_full_path = []
    for filename in glob.iglob(root_dir + '**/*.jpeg', recursive=True):
        input_names_full_path.append( filename )
        tokens = filename.split('/')
        file = tokens[ len(tokens) - 1 ]
        input_names.append( file ) 

    my_output_dir = "/home/touhid/Downloads/acss_videos_elena_outputs_v2/"

    batch_size = 4

    ###################### Load configuration and model###################

    with initialize(config_path='configs',job_name='inference'):
        cfg = compose(config_name='exp/gpv_inference')


    model = GPV(cfg.model).cuda().eval()
    loaded_dict = torch.load(cfg.ckpt, map_location='cuda:0')['model']
    state_dict = model.state_dict()
    for k,v in state_dict.items():
        state_dict[k] = loaded_dict[f'module.{k}']
        state_dict[k].requires_grad = False
    model.load_state_dict(state_dict)


    transforms = T.Compose([
        T.ToPILImage(mode='RGB'),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    #######################################################################

    for item in range( len(input_names) ):
        print( "Currently Processing " + str( item + 1 ) )  
        input_file = input_names[ item ]

        a11y_questions = utility.get_a11y_questions( 'a11y_questions_of_interest.txt' )

        a11y_objects = utility.get_a11y_objects( 'a11y_objects_of_interest.txt' )

        with open( my_output_dir + input_file.split('.')[0] + ".csv", "a") as csv_file:
            csvwriter = csv.writer( csv_file, delimiter=',')
            csvwriter.writerow( [ 'Object', 'GPV-1 Prediction', 'Ground Truth'] )

        inputs = []

        for a11y_question in a11y_questions:
            inputs.append( ( input_names_full_path[ item ], a11y_question ) )

        required_iterations = math.ceil( len( inputs ) / batch_size )

        for i in range( required_iterations ):
            input_batch = inputs[ i * batch_size : ( i + 1 ) * batch_size ]
            images, queries = preprocess( input_batch, transforms )
            output_batch = model( images, queries, None)
            predictions = decode_outputs( output_batch )

            for j in range( len(input_batch) ):
                img_path, query = input_batch[j]
                prediction = predictions[j]

                vis_img = vis_sample( img_path, prediction, 5 )
                with open(my_output_dir + input_file.split('.')[0] + ".csv", "a") as csv_file:
                    csvwriter = csv.writer( csv_file, delimiter=',')

                    flag = prediction[ 'answer' ]
                    if prediction[ 'answer' ] == 'yes' or prediction[ 'answer' ] == 'Yes':
                        flag = "1"
                    elif prediction[ 'answer' ] == 'no' or prediction[ 'answer' ] == 'No':
                        flag = "0"                       
                        

                    csvwriter.writerow( [ a11y_objects[ i * batch_size + j ], flag, "0"] )

        print( time.time() - start_time, end = "")
        print( " seconds" ) 
