######################################################
# Use these package versions
#!pip install torchtext==0.6.0 torch==1.13.1
######################################################


import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] =':16:8' #This is a command to reduce non-deterministic behavior in CUDA
import warnings
warnings.simplefilter("ignore", UserWarning)
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data import get_tokenizer
import sys
import argparse
from LanguageModel import LanguageModel
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

def main():
  chkpt = "got_language_model"

  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  logging.info('Using device: {}'.format(dev))

  logging.info("Loading tokenizer and vocab from vocab.pkl")  
  text_field = pickle.load(open("vocab.pkl", "rb"))
  vocab_size = len(text_field.vocab.itos)

  logging.info("Loading checkpoint {}".format(chkpt))
  lm = LanguageModel(vocab_size).to(dev)
  lm.load_state_dict(torch.load(chkpt, map_location=torch.device('cpu')))
  lm.eval()


  p = "the night is dark and full of terrors"

  # Torch is a bit frustrating at times and some things that ought to be deterministic are not.
  # This is an attempt to resolve that, but it doesn't work 100% of the time
  torch.use_deterministic_algorithms(True)
  seed = 42
  mlen = 150

  # torch.manual_seed(seed); np.random.seed(seed)
  # print("\n----------- Vanilla Sampling -----------")
  # print(sample(lm, text_field, prompt=p, max_len=mlen))

  # torch.manual_seed(seed); np.random.seed(seed)
  # print("\n------- Temp-Scaled Sampling 0.0001 -------")
  # print(sample(lm, text_field, prompt=p, temp=0.0001, max_len=mlen))

  # torch.manual_seed(seed); np.random.seed(seed)
  # print("\n------- Temp-Scaled Sampling 100 --------")
  # print(sample(lm, text_field, prompt=p, temp=100, max_len=mlen))

  # torch.manual_seed(seed); np.random.seed(seed)
  # print("\n----------- Top-k Sampling 1 -----------")
  # print(sample(lm, text_field, prompt=p, k=1, max_len=mlen))

  # torch.manual_seed(seed); np.random.seed(seed)
  # print("\n----------- Top-k Sampling 20 -----------")
  # print(sample(lm, text_field, prompt=p, k=20, max_len=mlen))

  # torch.manual_seed(seed); np.random.seed(seed)
  # print("\n----------- Top-p Sampling 0.001 -----------")
  # print(sample(lm, text_field, prompt=p, p=0.001, max_len=mlen))

  # torch.manual_seed(seed); np.random.seed(seed)
  # print("\n----------- Top-p Sampling 0.75 -----------")
  # print(sample(lm, text_field, prompt=p, p=0.75, max_len=mlen))

  # torch.manual_seed(seed); np.random.seed(seed)
  # print("\n----------- Top-p Sampling 1 -----------")
  # print(sample(lm, text_field, prompt=p, p=1, max_len=mlen))


  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Beam Search B=1 -----------")
  print(beamsearch(lm, text_field, prompt=p, beams=1, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Beam Search B=10 -----------")
  print(beamsearch(lm, text_field, prompt=p, beams=10, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Beam Search B=50 -----------")
  print(beamsearch(lm, text_field, prompt=p, beams=50, max_len=mlen))

  print()

############################################################################################
# TASK 2.1
############################################################################################

def beamsearch(model, text_field, beams=5, prompt="", max_len=50):
  decodedString = "Not implemented"

  p_tokens = text_field.process([text_field.tokenize(prompt.lower())])

  p_tokens_b = []
  h_b = torch.zeros(beams, 3, 1, 512)
  c_b = torch.zeros(beams, 3, 1, 512)
  probs_b = torch.zeros(beams)

  top_k_t_probs = torch.zeros(beams, beams, 3)

  #First beam
  h = torch.zeros(3, 1, 512)
  c = torch.zeros(3, 1, 512)

  out, h, c = model(p_tokens, h=h, c=c)
  out = out[-1,0,:]

  s_out = F.log_softmax(out) #Grab last output

  _, top_k_idx = torch.topk(s_out, beams)

  for beam, idx in enumerate(top_k_idx):
    h_b[beam] = h.clone()
    c_b[beam] = c.clone()
    probs_b[beam] += s_out[idx]
    p_tokens_b.append(idx.view(-1,1))

  #Beam Search
  for t in range(max_len-1):

    for beam in range(beams):

      out, h_b[beam], c_b[beam] = model(p_tokens_b[beam][-1].view(-1,1), h=h_b[beam], c=c_b[beam])

      s_out = F.log_softmax(out[-1,0,:]) #Grab last output

      top_k, top_k_idx = torch.topk(s_out, beams)

      b_tag = torch.ones((beams)) * beam

      top_k_t_probs[beam] = torch.stack((b_tag, top_k_idx, top_k), dim=1)


    flat_top_k_t_probs = torch.flatten(top_k_t_probs, start_dim=0, end_dim=1)
    _, flat_top_k_idx = torch.topk(flat_top_k_t_probs[:,-1], beams)

    temp_p_tokens = p_tokens_b[:]
    temp_probs = probs_b.clone()
    temp_h_b = h_b.clone()
    temp_c_b = c_b.clone()

    for i, idx in enumerate(flat_top_k_idx):
      beam, w_t1, prob = flat_top_k_t_probs[idx.item()]

      probs_b[i] = temp_probs[int(beam.item())] + prob
      p_tokens_b[i] = torch.cat((temp_p_tokens[int(beam.item())], w_t1.int().view(-1,1)))
      h_b[i] = temp_h_b[int(beam.item())].clone()
      c_b[i] = temp_c_b[int(beam.item())].clone()

  decodedString = reverseNumeralize(p_tokens_b[torch.argmax(probs_b)], text_field)

  return prompt + " " + decodedString

############################################################################################
# TASK 1.1
############################################################################################

def sample(model, text_field, prompt="", max_len=50, temp=1.0, k=0, p=10):
  assert (k==0 or p==10), "Cannot combine top-k and top-p sampling"
  decodedString = "Not implemented"

  p_tokens = text_field.process([text_field.tokenize(prompt.lower())])

  h = torch.zeros(3, 1, 512)
  c = torch.zeros(3, 1, 512)

  for _ in range(max_len):

    out, h, c = model(p_tokens, h=h, c=c)

    out = out[-1,0,:]

    s_out = F.softmax(out/temp)

    if k > 0:
      #Top-k Sampling
      s_out = F.softmax(out) #Grab last output

      # Create mask
      mask = torch.zeros_like(s_out)
      _, idx = torch.topk(s_out, k)
      mask[idx] = 1

      s_out_masked = torch.mul(mask, s_out) #Leave only top k values, set rest to zero
      s_out_masked_norm =  s_out_masked/torch.sum(s_out_masked) #Normalize probabilities
      w_t1 = torch.multinomial(s_out_masked_norm, 1)
    
    elif p<=1:
      #Top-p Sampling
      s_out = F.softmax(out) #Grab last output

      sorted_s_out, indices = torch.sort(s_out, dim=0, descending=True)

      cum_sum_probs = torch.cumsum(sorted_s_out, dim=0)

      nucleus = cum_sum_probs < p

      if not nucleus.any():
        nucleus[0] = True

      idx = nucleus.nonzero()

      # Create mask
      mask = torch.zeros_like(s_out)
      mask[indices[idx]] = 1

      s_out_masked = torch.mul(mask, s_out) #Leave only top p values, set rest to zero
      s_out_masked_norm =  s_out_masked/torch.sum(s_out_masked) #Normalize probabilities
      w_t1 = torch.multinomial(s_out_masked_norm, 1)

    else:
      #Vanilla and Temp Sampling
      s_out = F.softmax(out/temp) #Grab last output
      w_t1 = torch.multinomial(s_out, 1)

    w_t1 = torch.unsqueeze(w_t1, 1)

    p_tokens = torch.cat((p_tokens, w_t1))
  
  decodedString = reverseNumeralize(p_tokens, text_field)

  return decodedString

############################################################################################

def reverseNumeralize(numeralized_string, text_field):
  strings = [text_field.vocab.itos[i] for i in numeralized_string]
  return " ".join(strings)

if __name__ == "__main__":
  main()
