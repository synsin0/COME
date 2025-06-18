from collections import OrderedDict
import torch

def revise_ckpt(state_dict):
    tmp_k = list(state_dict.keys())[0]
    if not tmp_k.startswith('module.'):
        state_dict = OrderedDict(
            {('module.' + k): v
                for k, v in state_dict.items()})
    return state_dict

def revise_ckpt_1(state_dict):
    tmp_k = list(state_dict.keys())[0]
    if tmp_k.startswith('module.'):
        state_dict = OrderedDict(
            {k[7:]: v
                for k, v in state_dict.items()})
    return state_dict
        
def revise_ckpt_2(state_dict):
    param_names = list(state_dict.keys())
    for param_name in param_names:
        if 'img_neck.lateral_convs' in param_name or 'img_neck.fpn_convs' in param_name:
            del state_dict[param_name]
    return state_dict

def load_checkpoint(raw_model, state_dict, strict=True) -> None:
    # state_dict = checkpoint["state_dict"]
    model_state_dict = raw_model.state_dict()
    is_changed = False
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                # process embedding
                if k=='class_embeds.weight':
                    l1=state_dict[k].shape[0]
                    l2=model_state_dict[k].shape[0]
                    if l1>l2:
                        state_dict[k] = state_dict[k][:l2]
                    else:
                        state_dict_k_new=torch.zeros_like(model_state_dict[k])
                        state_dict_k_new[:l1]=state_dict[k]
                        state_dict[k]=state_dict_k_new
                else:
                    print(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                is_changed = True

        else:
            print(f"Dropping parameter {k}")
            is_changed = True

    # if is_changed:
    #     checkpoint.pop("optimizer_states", None)
    
    print(raw_model.load_state_dict(state_dict, strict=strict))