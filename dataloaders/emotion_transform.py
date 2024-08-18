# emotion_transform.py
import torch
import torch.nn as nn

class EmotionTransform(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EmotionTransform, self).__init__()
        self.transform = nn.Linear(input_dim, output_dim)

    def forward(self, emotion_embedding):
        # 假设情感嵌入是一个形状为 [batch_size, input_dim] 的张量
        # 输出将是形状为 [batch_size, output_dim]，这里 output_dim 设为224*224
        transformed_emotion = self.transform(emotion_embedding)
        return transformed_emotion
