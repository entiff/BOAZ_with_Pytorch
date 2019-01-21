import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
torch.manual_seed(777);
import numpy as np

class CNN_Clf(nn.Module):
    def __init__(self, vocab_size, embed_size, output_size, embedding_matrix, out_chs, DR_rate, filter_sizes):
        super(CNN_Clf, self).__init__()
        self.embed = nn.Embedding.from_pretrained(embedding_matrix)
        
        self.conv_1d_1 = nn.Sequential(
                            nn.Conv1d(1, out_chs, embed_size*filter_sizes[0]),
                            nn.Tanh()
                        )
        
        self.conv_1d_2 = nn.Sequential(
                            nn.Conv1d(1, out_chs, embed_size*filter_sizes[1]),
                            nn.Tanh()
                        ) 
        self.conv_1d_3 = nn.Sequential(
                            nn.Conv1d(1, out_chs, embed_size*filter_sizes[2]),
                            nn.Tanh()
                        )
        
        self.dropout = nn.Dropout(DR_rate)
        
        self.fc_layer = nn.Linear(out_chs*len(filter_sizes), output_size)
            
    def forward(self, inputs):
        batch_size = inputs.size(0)
        embed = self.embed(inputs)
        embed_cat = embed.reshape(batch_size, 1, -1)
        x = [self.conv_1d_1(embed_cat), self.conv_1d_2(embed_cat), self.conv_1d_3(embed_cat)]
        x = [F.max_pool1d(conv, (conv.size(2), )).squeeze(2) for conv in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.fc_layer(x)
        x = F.softmax(x ,dim=1)
        return x
      
    def predict(self, inputs, test_batch_size):
        embed = self.embed(inputs)
        embed_cat = embed.reshape(test_batch_size, 1, -1)        
        x = [self.conv_1d_1(embed_cat), self.conv_1d_2(embed_cat), self.conv_1d_3(embed_cat)]
        x = [F.max_pool1d(conv, (conv.size(2), )).squeeze(2) for conv in x]

        x = torch.cat(x, 1)
        x = self.fc_layer(x)
        x = F.softmax(x, dim=1)
        
        return x