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
class Skill3GPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.W = layer.weight.data.unsqueeze(-1)
        self.dev = layer.weight.device
        self.out_dim = self.W.shape[0] # out_dim
        self.in_dim = self.W.shape[1] # in_dim

        self.mean_inp = torch.zeros((self.in_dim), device=self.dev)
        self.inp_metric = torch.zeros((self.in_dim), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        batch_size = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()   # (dim, seqlen)
          
        # update mean_inp(基准值), 无偏差
        self.mean_inp *= self.nsamples / (self.nsamples + batch_size)
        self.mean_inp += torch.mean(inp, dim=1) / (self.nsamples + batch_size)
        
        # update △yj均值
        X_expanded = inp.unsqueeze(0)  # shape (1, in_dim, seq)
        self.norms_squared = torch.zeros((self.in_dim,), device=self.dev)
        print('1')
        blocksize = 204
        num_batches = (self.in_dim + blocksize - 1) // blocksize  # Equivalent to ceil(in_dim / batch_size)
        for i in range(num_batches):
            start = i * blocksize
            end = min((i+1) * blocksize, self.in_dim)
            
            # Compute Y for this batch
            X_batch = X_expanded[:, start:end, :]  # shape (1, batch_size, seq)
            W_batch = self.W[:, start:end, :]  # shape (out_dim, batch_size, 1)
            Y_squared = (X_batch * W_batch) ** 2  # shape (out_dim, batch_size, seq)
            
            # Compute and store norms for this batch
            self.norms_squared[start:end] = torch.sum(Y_squared, dim=(0, 2))
            torch.cuda.empty_cache()  
        
        # TODO:可以除||y||2
        self.inp_metric *= self.nsamples / (self.nsamples + batch_size)
        self.inp_metric += self.norms_squared / (self.nsamples + batch_size)

        self.nsamples += batch_size

        
    def free(self):
        self.mean_inp = None
        self.norms_squared = None
        self.inp_metric = None
        torch.cuda.empty_cache()  
    



# Define WrappedGPT class
class Skill2GPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.mean_inp = torch.zeros((self.columns), device=self.dev)
        self.l2_inp = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        batch_size = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()   # (dim, seqlen)
        # update mean_inp
        self.mean_inp *= self.nsamples / (self.nsamples + batch_size)
        self.mean_inp += torch.mean(inp, dim=1) / (self.nsamples + batch_size)
        
        # update l2_loss
        self.l2_inp *= self.nsamples / (self.nsamples + batch_size)
        self.l2_inp += torch.sum((inp - self.mean_inp.unsqueeze(1))**2, dim=1) / (self.nsamples + batch_size)   # a²+b²+c²...没开根号

        self.nsamples += batch_size

        
    def free(self):
        self.mean_inp = None
        self.l2_inp = None
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