import torch
import torch.nn as nn

def export_layers_to_txt(module, filename='./model_layers.txt'):
    """
    Export all the layers and their details to a text file.
    
    Args:
    - module (nn.Module): The PyTorch model to inspect.
    - filename (str): The name of the output text file.
    """
    with open(filename, 'w') as f:
        f.write("List of layers in the model:\n")
        f.write("=" * 40 + "\n")
        
        # Iterate over all modules in the model, including nested modules
        for name, submodule in module.named_modules():
            # Writing the module name and type to the file
            f.write(f"Layer name: {name}, Layer type: {type(submodule)}\n")
            
            # Optional: If you want to write more details about each layer (like parameters), you can add:
            if isinstance(submodule, nn.Module):
                for param_name, param in submodule.named_parameters():
                    f.write(f"  - Parameter: {param_name}, Shape: {param.shape}\n")
        
        f.write("=" * 40 + "\n")
    print(f"Model layers have been exported to {filename}")
