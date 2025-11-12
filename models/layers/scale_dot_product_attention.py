"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import math

from torch import nn


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Query(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose ===> [batch_size, head, d_tensor, length]
        # [batch_size, head, length, d_tensor] x [batch_size, head, d_tensor, length]
        # q @ k_t ==> [batch_size, head, length, length]
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        # [batch_size, head, length]
        score = self.softmax(score)

        # 4. multiply with Value
        # element-wise multiplication
        # [batch_size, head, length, d_tensor] x [batch_size, head, length, d_tensor]
        # [batch_size, head, length, d_tensor]
        v = score @ v

        return v, score




# img   (h, w, 3=rgb)      
# 10 images    (10=batch size,   h,   w,   3=rgb)


# I love you          (length=3,  d=1024)      (3, 1024)
# 10 sentences                               (10=batch_size, 1=head,  3=length, 1024=d)


# 8=   depend on model design
#      (10=batch_size, 8=head,  3=length, 1024=d)

# I love you === batch index = 0
# I hate you === batch index = 1
# you are good === batch index = 2
# 
#                               9  



# (a, b)   @  (b, 1)  ==> (a, 1)


# 自回归  auto-regressive
# 1. [SOS]      seen
# 2. I          seen
# 3. love       seen
# 4. you  <---- predicted 
# 5. if       ?????      prob=0.0    -99999999
# .......


# softmax -999999  ===>  prob=0





# (,  3=length,)

# I love you       --> A  B  C
#                      .....
#                      A'   B'    C'

# classification
# 


# mnist   28x28      R^784    VECTORS
#                                            R^1024  R^1024  
# 784-dimensionl vectors   784 tokens         A_1      A_2   A_3  ...  A_784
# each token ==> 1024
# 
#                                             B_1 ....              B_784
#                                             B_1 [ discarded 2-784]
#                                            R^1024
#                                            -->FC--> R^10
#                                             10-classification

# I LOVE YOU
# 


# [SOS] == (0, 0, 0, 0, ....., 0)    1024-d 
#  I


# [SOS]    I
#        LOVE 


# [SOS]   I    LOVE 
#               YOU

# [SOS]    I    LOVE   YOU
#                      AND

# [SOS]    I    LOVE   YOU  AND
#                            ME

# [SOS]    I    LOVE   YOU  AND   ME
#                                [EOS]


# [SOS]    I    LOVE   YOU  AND   ME  [EOS]
# end of inference