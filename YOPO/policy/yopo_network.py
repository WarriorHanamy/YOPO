"""
YOPO Network
forward, prediction, pre-processing, post-processing
"""

import torch
from torch import nn

from .models.backbone import YopoBackbone
from .models.head import YopoHead
from .state_transform import StateTransform


class YopoNetwork(nn.Module):
    def __init__(
        self,
        observation_dim=9,
        output_dim=10,
        hidden_state=64,
    ):
        super().__init__()
        self.state_transform = StateTransform()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.image_backbone = YopoBackbone(hidden_state)
        self.state_backbone = nn.Sequential()
        self.yopo_head = YopoHead(hidden_state + observation_dim, output_dim)

    def forward(self, depth: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """
        forward propagation of neural network
        """
        depth_feature = self.image_backbone(depth)
        obs_feature = self.state_backbone(obs)
        input_tensor = torch.cat((obs_feature, depth_feature), 1)
        output = self.yopo_head(input_tensor)
        endstate = torch.tanh(output[:, :9])
        score = torch.nn.functional.softplus(output[:, 9])
        return endstate, score

    def inference(self, depth: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """
        For network training:
        (1) normalize the input state and transform to primitive frame
        (2) forward propagation
        (3) convert the prediction to endstate in body frame.
        obs: current state in the body frame.
        return: end state in the body frame
        """
        obs = self.state_transform.normalize_obs(obs)
        obs = self.state_transform.prepare_input(obs)
        endstate_pred, score_pred = self.forward(depth, obs)
        endstate = self.state_transform._pred_to_endstate(endstate_pred)
        return endstate, score_pred

    def _print_grad(self, grad):
        print('grad of hook: ', grad)
