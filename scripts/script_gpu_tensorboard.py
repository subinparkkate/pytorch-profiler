import sys
import os
import torch
import torch.profiler as profiler
from torch.profiler import ProfilerActivity

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from attention import Attention

# Hyperparameter
d_model = 512  
num_heads = 8  
seq_length = 10   
batch_size = 2    

# input tokens
input_tokens = torch.randint(0, 10000, (batch_size, seq_length)).cuda() 

attention_layer = Attention(d_model=d_model, num_heads=num_heads).cuda()

output, attention_weights = attention_layer(input_tokens)

#result
with profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True, on_trace_ready=profiler.tensorboard_trace_handler( '../log/attention_gpu')) as prof:
   output, attention_weights = attention_layer(input_tokens)

print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=10))
