"""
Asymmetric feature extraction. Via defining a list of keys in the original observation space,
the extractor processes only the selected keys while the rest of the observation is discarded.

This is used for giving less information to the policy, while full information
to the Q function when training policies with access to privileged information.
"""
from typing import Dict, List

from gymnasium import spaces
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.preprocessing import is_image_space, get_flattened_obs_dim

TensorDict = Dict[str, th.Tensor]

class PartialInfoExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_mask: List[str],
        observation_space: spaces.Dict,
        cnn_output_dim: int = 256,
        normalized_image: bool = False,
    ) -> None:
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        extractors: dict[str, nn.Module] = {}
        self.keys_to_keep = observation_mask

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if key in self.keys_to_keep:
                if is_image_space(subspace, normalized_image=normalized_image):
                    extractors[key] = NatureCNN(subspace, features_dim=cnn_output_dim, normalized_image=normalized_image)
                    total_concat_size += cnn_output_dim
                else:
                    # The observation key is a vector, flatten it if needed
                    extractors[key] = nn.Flatten()
                    total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            if key in self.keys_to_keep:
                encoded_tensor_list.append(extractor(observations[key]))
        
        return th.cat(encoded_tensor_list, dim=1)

