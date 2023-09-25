from collections import OrderedDict

# define function to clean up dataparallel config for loading
def clean_state_dict(state_dict):
    '''
    Removes the "module." from the state_dict keys
    '''
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if "module." in k:
            name = k.replace("module.", "")
        if "dronet." in name:
            name = name.replace("dronet.", "encoder.")
        new_state_dict[name] = v
    return new_state_dict

