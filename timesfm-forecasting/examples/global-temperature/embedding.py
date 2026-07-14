import torch
import torch.nn as nn

x = torch.tensor([[1.0, 2.0, 3.0, 4.0,
                    5.0, 6.0, 7.0, 8.0]]) # tiny time series data 
#patch embedding (patch_size=4, d_model=6)
# ✅ define the linear layer first
d_model = 6

# ✅ patch must be nn.Linear, not a number
patch = nn.Linear(4, d_model)   # ← fix this line    # 6 # a vector itself does not have a dimension. Rather, it belongs to a "6-dimensional vector space," meaning a space that requires 6 independent basis vectors to define it.


patches = x.unfold(-1, 4, 4)#-1 (Dimension): Specifies the dimension to operate on (the last one), extracts non-overlapping chunks of size 4 along the last dimension of the tensor 
p_emb   = patch(patches)  #  2 patches of size 4

# patch 0 = [1,2,3,4]  → 6 numbers
# patch 1 = [5,6,7,8]  → 6 numbers  
# postitional embedding  
pos_enc = nn.Embedding(2, 6)   # 2 positions, each 6-dim

pos     = torch.arange(2)      # [0, 1]
pos_emb = pos_enc(pos)          # [2, 6]  
#add patch embedding and positional embedding together
out = p_emb + pos_emb   # [1, 2, 6]

# "patch 0 is at position 0"
# "patch 1 is at position 1"

print(out.shape)
# torch.Size([1, 2, 6])  
# 
# torch.Size([1, 2, 6])
             # ↑  ↑  ↑
              #│  │  └── 6 = d_model  (each patch projected to 6 numbers)
             # │  └───── 2 = patches  (8 steps ÷ 4 patch_size = 2)
             # └──────── 1 = batch    (how many series you fed in)                                  

#seq_len = 8 time steps
        #  ↓ split into patches
         # 2 patches

#each patch → goes into a bag of size d_model
           #  bag size = YOUR CHOICE (6, 64, 128, 512...)
            # has NOTHING to do with 8
         