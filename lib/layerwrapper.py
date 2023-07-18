import torch
import torch.nn as nn


# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples
        
    def free(self):
        self.scaler_row = None
        torch.cuda.empty_cache()
    
    
# Define WrappedGPT class
class SkillGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_inp = torch.zeros((self.columns), device=self.dev)
        self.skill_inp = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.scaler_inp *= self.nsamples / (self.nsamples+tmp)
        self.skill_inp *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp
        
        inp = inp.type(torch.float32)
        self.skill_inp += inp.mean(axis=1) / self.nsamples
        self.scaler_inp += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples
        
    def free(self):
        self.scaler_inp = None
        self.skill_inp = None
        torch.cuda.empty_cache()       
        

# Define WrappedGPT class
class WrappedPlusGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_inp = torch.zeros((self.columns), device=self.dev)
        self.scaler_out = torch.zeros((self.rows), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
            out = out.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
                out = out.reshape((-1, out.shape[-1]))
            inp = inp.t()
            out = out.t()

        self.scaler_inp *= self.nsamples / (self.nsamples+tmp)
        self.scaler_out *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        out = out.type(torch.float32)
        self.scaler_inp += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples
        self.scaler_out += torch.norm(out, p=2, dim=1) ** 2  / self.nsamples
        
    def free(self):
        self.scaler_inp = None
        self.scaler_out = None
        torch.cuda.empty_cache()