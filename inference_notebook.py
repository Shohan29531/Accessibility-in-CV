import hydra
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf
import torch
import torchvision.transforms as T
import numpy as np
from nltk.tokenize.treebank import TreebankWordDetokenizer

import time
import math

from exp.gpv.models.gpv import GPV
from utils.detr_misc import collate_fn
from inference_util import *
import a11y_utils.utility



my_output_dir = "/home/touhid/Desktop/gpv-1/a11y_testing_outputs/"
batch_size = 4

start_time = time.time()

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




# img1,_ = read_image('assets/busy_street.png',resize_image=True)
# img2,_ = read_image('assets/white_horse.png',resize_image=True)

# imshow((255*img1[:,:,::-1]).astype(np.uint8)) # scale pixel values, RGB to BGR (because imshow uses opencv), and convert to uint8
# imshow((255*img2[:,:,::-1]).astype(np.uint8))



inputs = [
    ('assets/2.jpeg','is there a wall?'),
    ('assets/2.jpeg','is there a sign?'),
    ('assets/2.jpeg','is there a parallel parking spot?'),
    ('assets/ex.jpg','is there a stop sign?'),
    ('assets/blind.jpeg','is there a person with disability?'),
    ('assets/blind.jpeg','is there a white cane?'),
    ('assets/blind.jpeg','is there a blind person?'),
    ('assets/blind.jpeg','is there a white cane?'),
]




number_of_inputs = len( inputs )

required_iterations = math.ceil( number_of_inputs / batch_size )

for i in range( required_iterations ):
    input_batch = inputs[ i * batch_size : ( i + 1 ) * batch_size ]
    images, queries = preprocess( input_batch, transforms )
    output_batch = model( images, queries, None)
    predictions = decode_outputs( output_batch )

    for j in range( len(input_batch) ):
        img_path, query = input_batch[j]
        prediction = predictions[j]

        vis_img = vis_sample( img_path, prediction, 5 )
        print( '-' * 80 )
        print( f'Query { i * batch_size + j + 1 }:', query )
        print( f'Ans:', prediction[ 'answer' ] )

        cv2.imwrite( my_output_dir + "output_" + str(j) + ".png", vis_img )


print( time.time() - start_time, end = "")
print( " seconds" )    