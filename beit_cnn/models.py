import timm
import torch
import torch.nn as nn


class TimmNetWrapper(nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super(TimmNetWrapper, self).__init__()

        self.model = timm.create_model(model_name, pretrained=True)
        

        if hasattr(self.model, 'classifier'):
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, num_classes)
        
        elif hasattr(self.model, 'fc'):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        
        elif hasattr(self.model, 'head'):
            in_features = self.model.head.in_features
            self.model.head = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
