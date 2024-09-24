import time

# # Copyright (c) Facebook, Inc. and its affiliates.
# #
# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.

# import numpy as np
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '6'
# d = 768                           # dimension
# nb = 100000                      # database size
# nq = 10000                       # nb of queries
# np.random.seed(1234)             # make reproducible
# xb = np.random.random((nb, d)).astype('float32')
# xb[:, 0] += np.arange(nb) / 1000.
# xq = np.random.random((nq, d)).astype('float32')
# xq[:, 0] += np.arange(nq) / 1000.

# import faiss                     # make faiss available

# ngpus = faiss.get_num_gpus()

# print("number of GPUs:", ngpus)

# cpu_index = faiss.IndexFlatL2(d)

# gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
#     cpu_index
# )

# gpu_index.add(xb)              # add vectors to the index
# # print(gpu_index.ntotal)
# # cpu_index.add(xb)
# k = 32                          # we want to see 4 nearest neighbors
# # import ipdb; ipdb.set_trace()
# times = time.time()
# for i in range(100):
#     D, I = gpu_index.search(xq, k) # actual search
#     # D, I = cpu_index.search(xq, k) # actual search
# timee = time.time()
# timea = timee-times
# print(timea)
# # print(I[:5])                   # neighbors of the 5 first queries
# # print(I[-5:])                  # neighbors of the 5 last queries
import torch
# from external.memorizing_transformers_pytorch import KNNMemory
from external.memorizing_transformers_pytorch.memorizing_transformers_pytorch import KNNMemory
memory = KNNMemory(
    dim = 768,                   # dimension of key / values
    max_memories = 64000,       # maximum number of memories to keep (will throw out the oldest memories for now if it overfills)
    num_indices = 2             # this should be equivalent to batch dimension, as each batch keeps track of its own memories, expiring when it sees a new document
)

memory.add(torch.randn(2, 512, 2, 768))  # (batch, seq, key | value, feature dim)
memory.add(torch.randn(2, 512, 2, 768))

memory.clear([0]) # clear batch 0, if it saw an <sos>

memory.add(torch.randn(2, 512, 2, 768))
memory.add(torch.randn(2, 512, 2, 768))
times = time.time()
for i in range(100):
    key_values, mask = memory.search(torch.randn(2, 512, 768), topk = 32)
    # D, I = gpu_index.search(xq, k) # actual search
    # D, I = cpu_index.search(xq, k) # actual search
timee = time.time()
timea = timee-times
print(timea)