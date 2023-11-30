import torch
import torch.nn as nn

class MassModel(nn.Module):
    def __init__(self, CMRPConfig):
        super(MassModel, self).__init__()
        self.filters = CMRPConfig.mr1_filters
        self.window = CMRPConfig.mr1_window
        self.num_channels = CMRPConfig.mr1_num_channels
        self.stride = CMRPConfig.mr1_stride

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=self.num_channels, out_channels=self.filters, kernel_size=self.window, padding=self.window//2, stride=self.stride),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.filters, out_channels=self.filters//2, kernel_size=self.window, padding=self.window//2, stride=self.stride),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.filters//2, out_channels=self.filters//4, kernel_size=self.window, padding=self.window//2, stride=self.stride),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.filters//4, out_channels=self.filters//32, kernel_size=self.window, padding=self.window//2, stride=self.stride),
            nn.ReLU()
            # Conv1D(filters/16, window, padding='same', strides=2, input_shape=input_shape, activation='relu'),
        )
    def forward(self, inputs):
        inputs = inputs.float()
        out = self.encoder(inputs)
        out = out.squeeze(1)  # remove channel dim to match with CMRP model
        return out
