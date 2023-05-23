import os

import torch

from ...ml_decoder.ml_decoder import add_ml_decoder_head


from ..tresnet import TResnetL



# Specifically for Stanford model

def create_model():
    """Create a model
    """
    model_params = {'num_classes': 196}

    model = TResnetL(model_params)

    ####################################################################################
   
    model = add_ml_decoder_head(model,num_classes=196,num_of_groups=-1,
                                decoder_embedding=768, zsl=0)
    ####################################################################################
    # loading pretrain model
    model_path = './tresnet_l_stanford_card_96.41.pth'
   
    state = torch.load(model_path, map_location='cpu')
    if 'model' in state:
        key = 'model'
    else:
        key = 'state_dict'
    model.load_state_dict(state[key], strict=True)

    return model
