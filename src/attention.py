import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.profiler as profiler

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(ScaledDotProductAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        # token -> embeding vector
        self.embedding_layer = nn.Embedding(10000, d_model) #10000개의 token을 처리할 수 있고 d_model 차원의 벡터로 변환됨.
        # self.embedding_layer.weight.size() = ([10000, 512])

        # X * weight 
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        ## self.query, key, value의 weight, bias가 랜덤하게 초기화됨.

        # final output layer
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, input_tokens):
        """이미 토큰화된 입력을 받아서 어텐션을 수행하는 함수"""
        with profiler.record_function("attention Forward"):
            batch_size = input_tokens.shape[0]

            # 1. token -> embedding vector
            X_embedded = self.embedding_layer(input_tokens)  # 내가 아는 input shape임. 3*10*512

            # 2. 위에 초기화된 weight, bias를 이용해서 input embedding vector를 변환해서 Q,K,V만듬.
            # 2. 즉, X * 모든 헤드에 대한 weight
            Q = self.query(X_embedded)  #2*10*512
            K = self.key(X_embedded)    
            V = self.value(X_embedded)  
            ## 모든 헤드에 대한 weight를 먼저 다 곱하기. 그러면 다 곱하고 나중에 나눠도 각 헤드별로 weight 다른 걸로 곱한걸로 생각할 수 있지.

            # 3. Q, K, V를 여러 헤드로 나누기 (Multi-head Attention)
            Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # 2 * 8 * 10 * 64
            K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  
            V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  
            # view = shape를 변경하는 명령어임.
            # -1 : 차원이 유지되도록 -1로 표시하면 해당 부분 알아서 계산해 //여기서는 10임. 
            # num_head = 8
            # head_dim = 64
            # transpose = 0, 1, 2, 3 중 1, 2의 순서 바꿔

            # 4. Q * K^
            energy = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5) # ** 0.5 = 제곱근
            # -1 : 마지막 차원, -2: 마지막에서 두번째 차원
            # 즉, 마지막 두 차원의 위치를 바꿈. 
            # 2*8*10*64 -> 2*8*64*10
            # energy.size는 2*8*10*10이 됨.

            # 5. softmax
            attention = torch.softmax(energy, dim=-1)  
            # -1 : 마지막 차원에 softmax를 적용해라 //10임

            # 6. S * V 
            out = torch.matmul(attention, V)  #2*8*10*64

            # 7. concat - shape 다시 크게 하나 
            out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
            # contiguous : 메모리 상에서 텐서의 저장 형태가 연속적이도록 보장
            # size : 2, 10, 512

            # 8. concat 후 weight 곱하는 과정임.
            out = self.fc_out(out)  

            # 9. weight 저장
            #torch.save(self.query.state_dict(), 'query_weights.pt')
            #torch.save(self.key.state_dict(), 'key_weights.pt')
            #torch.save(self.value.state_dict(), 'value_weights.pt')

            #.pt 파일로 가중치 저장

            #torch.save({
            #    'query_state_dict': self.query.state_dict(),
            #    'key_state_dict': self.key.state_dict(),
            #    'value_state_dict': self.value.state_dict(),
            #}, 'attention_weights.pt')

            return out, attention  # 최종 결과, attention score