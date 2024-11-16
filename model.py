import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(1, 1)  # Simple linear regression example

    def forward(self, x):
        return self.fc(x)

# Save a pre-trained model for deployment
model = SimpleModel()
torch.save(model.state_dict(), "model.pth")
