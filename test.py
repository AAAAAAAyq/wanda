import torch 
import torch.nn.functional as F

########## 验证LLM-int8()的去冗余通道的假设 ##########
# x_in = torch.tensor([[0, 1, -60, 4],
#                 [3, 0, -50, -2],
#                 [-1, 0, -55, 1]])
# w_q = torch.tensor([[-1, -1, -1, -1],
#                     [-1, -1, -1, -1],
#                     [1, -1, -1, 1],
#                     [-1, -1, -1, -1]])
# w_k = torch.tensor([[-1, -1, -1, -1],
#                     [-1, -1, -1, -1],
#                     [1, -1, -1, -1],
#                     [-1, -1, -1, -1]])
# x_q = x_in @ w_q
# x_k = x_in @ w_k

# scores = torch.matmul(x_q, x_k.transpose(0, 1))
# outs = F.softmax(scores.float(), dim=1)
# print(outs)

########## 推导SparseGPT的公式 ##########
# x_o = torch.tensor([[1, 1, 1, 1],
#                     [1, 1, 1, 1],
#                     [1, 1, 1, 1]]).T.float()  # (in=4, seq=3)

# x_q = torch.tensor([[1, 0, 1, 1],
#                     [1, 0, 1, 1],
#                     [1, 0, 1, 1]]).T.float()  # (in=4, seq=3)

# w_q = torch.tensor([[1, 1, 1, 1]]).float()
# Hessian = x_q @ x_q.T
# w_max = torch.linalg.inv(Hessian.float() + torch.eye(4) * 0.001) @ x_q @ (w_q @ x_o).T
# import pdb;pdb.set_trace()

########## 验证Skill Neurons的剪枝思想 ##########
w_q = torch.tensor([[1, 3, 1, -2, 1],
                    [1, 5, 1, -1, 6],
                    [4, 3, 2, 5, -1],
                    [2, 6, 1, -3, -4]]).T  # [out=5, in=4]
x_1 = torch.tensor([[1, 3, -2, 1],
                    [1, 5, -1, 6],
                    [2, 6, -3, -4]]).T  # [in=4, seq=3]
x_2 = torch.tensor([[6, 3, -2, 1],
                    [2, 5, -1, 6],
                    [-5, 6, -3, -4]]).T  # [in=4, seq=3]
import pdb;pdb.set_trace()
y_1 = w_q @ x_1
y_2 = w_q @ x_2
y_2_by_1 = w_q[:,1:] @ x_1[1:] + w_q[:,:1] @ x_2[:1,:]
import pdb;pdb.set_trace()