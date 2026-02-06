import os

import torch.nn as nn
import torch

# AudioMLP: audio mlp
class AudioMLP(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=384):
        super().__init__()
        self.embed_to_whispers = nn.ModuleList([
            self._build_mlp(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim),
            self._build_mlp(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim),
            self._build_mlp(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim),
            self._build_mlp(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim),
            self._build_mlp(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        ])
    # _build_mlp: build mlp
    def _build_mlp(self, input_dim, hidden_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
    # forward: forward pass
    def forward(self, x):
        return [net(x) for net in self.embed_to_whispers]  # 返回5个输出张量的列表

# AudioEmbedding: audio embedding
class AudioEmbedding(nn.Module):
    def __init__(self, input_dim=384 * 5, hidden_dim=2048, output_dim=8194):
        super().__init__()
        self.whispers_to_token = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, output_dim)
        )
        self.codec_embed = nn.Embedding(8194, 512)
    # forward: forward pass
    def forward(self, x):   
        l=0
        if len(x.shape)==3:
            b,l,d=x.shape
            x=x.view(-1,d)
        audio_token = self.whispers_to_token(x)  # [(B T), 8192]
        indices = audio_token.argmax(dim=-1)  # [(B T)]
        audio_embedding = self.codec_embed(indices)  # [(B T), embed_dim]
        if l!=0:
            audio_embedding=audio_embedding.view(b,l,-1)
        return audio_embedding


