import sys
import os
import torch
import torch.profiler as profiler
from torch.profiler import ProfilerActivity

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from attention import ScaledDotProductAttention

# Hyperparameter
d_model = 512  #임베딩 차원
num_heads = 8  
seq_length = 10  #입력 문장 길이  
batch_size = 2  #한번에 처리하는 입력 데이터 샘플 수  

# input tokens
input_tokens = torch.randint(0, 10000, (batch_size, seq_length)).cuda() # 행렬 2*10이고 각 값은 0~10000 사이값임. 즉, 두 문장이고 한 문장은 10개의 토큰?

# 행: 2, 열 : 10 

# make ScaledDotProductAttention Class instance
attention_layer = ScaledDotProductAttention(d_model=d_model, num_heads=num_heads).cuda()

# forward operation
output, attention_weights = attention_layer(input_tokens)

#result
with profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory = True, record_shapes=True, on_trace_ready=profiler.tensorboard_trace_handler( '../log/attention_gpu')) as prof:
   output, attention_weights = attention_layer(input_tokens)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
