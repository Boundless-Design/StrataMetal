# Apple Silicon (experimental)

The engines Strata ships are CUDA and HIP. Neither compiles for an Apple Silicon Mac, and the CPU expert
kernels are AVX2 and AVX-512, which this CPU does not have. `metal/` is the Apple Silicon engine. It does not
run the model yet. It names the GPU, checks published model sizes against the working set Metal reports, and
checks a BF16 matrix-vector product, the DeltaNet kernels, QSA decode attention, the Coder expert product,
the S-form dense projection, the GDN layer glue, and one GDN layer built from those kernels, against a CPU
result. The glue check also covers one embedding row, the expert-half add, and the QSA weighted norm.
The expert-half add is a kernel only. The embedding row is called from the token check. The QSA
weighted norm is called from the QSA attention block. The gated residual, the MoE router, the MoE
combine, and the shared expert are each checked on their own, and the token check calls them. The
sampler is checked on its own, and the token check calls it. The 2-bit product over a Q8_0
activation is checked on its own. The QSA gate is checked on its own, and the decode kernel
does not call that either. The indexer append is checked on its own, and the decode kernel
does not call that either. The indexer cell scores are checked on their own, and the decode
kernel does not call that either. The top-k selection is checked on its own, and the decode
kernel does not call that either. The paged KV append and gather are checked on their own,
and the decode kernel does not call that either. The partial NEOX rotation is checked on its
own. The indexer rotates one pooled row, and that kernel is a different call. One QSA attention
block is checked on its own, on synthetic weights. It calls the indexer, the cell scores, the top-k,
the paged append, the rotation, the gate, the weighted norm, and the decode kernel. It does not call
the gated residual, the router, the experts, or the sampler. One synthetic token calls the embedding
row, the gated residual, that attention block, the router, the routed experts, the shared expert,
the combine, the output projection, and the sampler. It is one block. The same check also draws
one greedy token at the model embedding, n_embd 2560, indexer dim 128, and 64 rotated values, with
rank 32, 8 experts, hidden 64, vocabulary 32, and top-k 2. It also draws one greedy token at the
model residual rank, hc_lr 320, on that same embedding. It also draws one greedy token there with
routed hidden 640, 8 experts, and shared hidden 64. It also draws one greedy token there with
512 experts, one there with shared hidden 640, and one at vocabulary 248320. It also draws
sixteen greedy cells at that vocabulary, top-k 2 against top-k 2048. Four synthetic
layers follow the pack schedule, three GDN and one QSA, and that is not the 48-layer model.
The pack index is read into those kernels, including one embedding row, one shared expert, one routed expert blob, one 2-bit query, and one synthetic greedy token through the stack. The same check writes one synthetic layer at n_embd 2560, 512 experts, and loads it back. When the Q2_0 extract is present, it also loads layer 0 of that file: the query plane is IQ4_XS, the gate is Q4_K, the output is Q5_K, those three products matched, one GDN step on them matched at relative error 1.460e-07, and the attention half around that step matched at read 9.033e-08, step 3.205e-05, and write 6.919e-06. The feed-forward half matched at read 8.694e-08, router 9.173e-07, expert 0, and write 4.573e-08. One layer-0 pass feeds the attention write into that half: attention write 6.919e-06, then read 5.457e-08, router 1.956e-07, expert 0, and write 3.809e-08. The shared-expert rows are a separate check: IQ4_XS gate relative error 0, Q3_K up 8.589e-08, Q5_0 down 5.944e-08, and the forward write 0. That check is not added into the layer-0 residual. The token embedding is Q3_K, relative error 7.273e-08 on 248320 rows, and the output weight is Q5_K, relative error 2.042e-07 on 248320 rows. Those rows are not added into the layer-0 residual. The output mixer matched at relative error 8.556e-08 on 2560 values. That check is not added into the layer-0 residual. One embedding row, token 7971, copied into the four residual streams, matched the head at mix 1.008e-07 and at logits 1.053e-07 on 248320 rows. That check is not added into the layer-0 residual. The same row through the layer-0 pass matched at attention write 1.132e-06, then read 4.949e-08, router 2.773e-08, expert 0, and write 1.019e-07. The head on that residual matched at mix 4.456e-08 and at logits 9.448e-08 on 248320 rows. Adding the shared expert, expert 0 weighted exactly 1 and the shared vector added plain, matched at mix 4.410e-08 and at logits 9.438e-08. The layer-0 router selection at k 10 matched at logits 2.773e-08 and weights 2.980e-08, with 0 id differences. Expert 0 is not in that selection. The ten selected experts matched at relative error 0, mix 4.325e-08, and logits 8.982e-08 on 248320 rows. Token 7971's state carried into token 1 matched at state 5.638e-08, mix 6.306e-08, and logits 1.263e-07. Layer 1's GDN step on that residual matched at block 1.310e-07. Layer 1's routed experts on that residual matched at mix 7.435e-08 and logits 1.176e-07. Layer 2's GDN step on that residual matched at block 0 and state 6.470e-08. Layer 2's routed experts on that residual matched at mix 4.534e-08 and logits 1.419e-07. The shared expert was not added. A synthetic Q4_0 block matched at relative error 0. Layer 2's shared expert on that residual matched at mix 4.745e-08 and logits 1.369e-07. The shared gate, up, and down matched at 0. Layer 3's query matched at integer 0 on 12288 rows, the key at 7.421e-08, the value at 2.297e-07, and the output at 2.095e-07. The QSA block matched at 0. Position 1 matched at 0. The rotated key matched at 1.353e-09. The attention write matched at 1.412e-09. The ten routed experts on that residual matched. The read matched at 3.250e-08, the inject at 1.103e-06, the router at 2.666e-08, and the expert products at 0. The combine matched at 0 and the write at 4.978e-10. The mix matched at 8.420e-08. The logits matched at 1.143e-07, integer 0, on 248320 rows. The experts were 396 64 405 40 454 171 310 78 424 235. The shared expert was not added in that routed check. The shared expert then ran on the same residual. Its gate is Q2_0, its up is Q2_0, and its down is Q4_0. The gate, the up, and the down matched at 0. The scale matched at 5.156e-07 and the shared write at 9.926e-07. The combine matched at 0 and the residual write at 3.695e-10. The mix matched at 3.940e-08. The logits matched at 1.205e-07, integer 0, on 248320 rows. Layer 4's GDN step on that residual matched. The query is IQ4_XS, the gate is Q3_K, and the output is IQ4_XS. The block matched at 0, the state at 5.946e-08, the convolution at 0, and the write at 0. 786432 of 786432 state values moved. 10240 of 30720 convolution values moved. The state started at zero. Query byte 2 differed on 5135 of 10240. Layer 4's ten routed experts on that residual matched. The shared expert was not added. The read matched at 8.642e-08, the inject at 6.520e-07, and the router at 2.236e-08. The expert products matched at 0. The combine matched at 0 and the write at 6.487e-09. The mix matched at 4.997e-06. The logits matched at 1.193e-07, integer 0, on 248320 rows. The experts were 472 283 309 92 81 410 342 496 41 423. Residual byte 2 differed on 248303 of 248320. Layer 4's shared expert then ran on the same residual. Its gate, its up, and its down are Q2_0. The gate, the up, and the down matched at 0. The scale matched at 6.248e-07 and the shared write at 1.460e-06. The combine matched at 0 and the residual write at 6.315e-09. The mix matched at 4.645e-08. The logits matched at 1.164e-07, integer 0, on 248320 rows. The experts were the same ten. Residual byte 3 differed on 248320 of 248320. Dropping the shared expert differed on 10240 of 10240. Layer 5's GDN step on that shared residual matched. The query is Q3_K, the gate is Q3_K, and the output is IQ4_XS. The block matched at 0, the state at 7.936e-08, the convolution at 2.403e-09, and the write at 1.295e-08. 786432 of 786432 state values moved. 10240 of 30720 convolution values moved. The state started at zero. Query byte 2288037 differed on 9631 of 10240. Layer 5's ten routed experts on that residual matched. The shared expert was not added. The read matched at 3.802e-08, the inject at 9.602e-07, and the router at 2.447e-08. The expert products matched at 0. The combine matched at 0 and the write at 0. The mix matched at 5.669e-08. The logits matched at 1.234e-07, integer 0, on 248320 rows. The experts were 167 454 478 98 42 216 236 291 263 459. Residual byte 10 differed on 248319 of 248320. A synthetic Q6_K block matched at relative error 1.360e-08, integer 0, on 8 rows. Layer 5's shared expert on that residual matched. The gate is Q2_0, the up is Q6_K, and the down is Q4_0. The gate, the up, and the down matched at 0. The scale matched at 3.060e-07 and the shared write at 4.764e-07. The combine matched at 0 and the residual write at 0. The mix matched at 5.262e-08. The logits matched at 1.195e-07, integer 0, on 248320 rows. The experts were 167 454 478 98 42 216 236 291 263 459. Residual byte 10 differed on 248319 of 248320. Dropping the shared expert differed on 10240 of 10240. Layer 6's GDN step on that shared residual matched. The query is Q3_K, the gate is Q3_K, and the output is Q3_K. The block matched at 3.472e-09, the state at 5.386e-08, the convolution at 2.109e-09, and the write at 0. 786432 of 786432 state values moved. 10240 of 30720 convolution values moved. The state started at zero. Query byte 2288037 differed on 5707 of 10240. Layer 6's ten routed experts on that residual matched. The shared expert was not added. Its up is Q4_K and was not loaded. The read matched at 6.108e-08, the inject at 1.550e-06, and the router at 2.467e-08. The expert products matched at 0. The combine matched at 0 and the write at 9.250e-09. The mix matched at 6.136e-08. The logits matched at 1.310e-07, integer 0, on 248320 rows. The experts were 223 201 33 127 77 66 379 152 159 283. Residual byte 6 differed on 248311 of 248320. Layer 6's shared expert on that residual matched. The gate is Q2_0, the up is Q4_K, and the down is Q2_0. The gate matched at 0, the up at 1.197e-07, and the down at 0. The scale matched at 4.663e-07 and the shared write at 9.102e-07. The combine matched at 0 and the residual write at 1.014e-08. The mix matched at 5.823e-08. The logits matched at 1.308e-07, integer 0, on 248320 rows. The experts were 223 201 33 127 77 66 379 152 159 283. Residual byte 7 differed on 248319 of 248320. Dropping the shared expert differed on 10240 of 10240. Layer 7's attention products matched on their own activations. The QSA block was not called. The query is Q2_0 and matched at 0 on 12288 rows. One code byte differed on 1 of 12288. A zero activation was 0 on all 12288. The key is Q4_K and matched at 2.325e-07, integer 0, on 512 rows. One code byte differed on 1 of 512. A zero activation was 0 on all 512. The value is Q3_K and matched at 7.426e-08, integer 0, on 512 rows. One code byte differed on 1 of 512. A zero activation was 0 on all 512. The output is Q4_K and matched at 2.186e-07, integer 0, on 2560 rows. One code byte differed on 1 of 2560. A zero activation was 0 on all 2560. Layer 7's QSA block on that shared residual matched. The query is Q2_0, the key is Q4_K, the value is Q3_K, and the output is Q4_K. The read matched at 7.924e-08 and the inject at 9.320e-07. The block matched at 0 and position 1 matched at 0. The rotated key matched at 3.458e-08. The selected ids were 0 and 1. A second draw differed on 0, and the guard word stayed. A zero residual differed on 2560 of 2560, 100.00% apart. Value byte 48 differed on 2559 of 2560. Layer 7's attention write on that residual matched. The read matched at 7.924e-08 and the inject at 9.320e-07. The write matched at 6.091e-09. A second draw differed on 0, and the guard word stayed. A zero residual differed on 10240 of 10240, 100.00% apart. Value byte 48 differed on 10159 of 10240. Layer 7's ten routed experts on that residual matched. The shared expert was not added. The gate, the up, and the down are Q2_0. The read matched at 3.570e-08, the inject at 1.012e-06, and the router at 2.360e-08. The expert products matched at 0. The combine matched at 0 and the write at 0. The mix matched at 5.239e-08. The logits matched at 1.373e-07, integer 0, on 248320 rows. The experts were 80 358 350 436 163 341 346 165 239 22. A second draw differed on 0, and the guard word stayed. A zero residual differed on 248320 of 248320, 100.00% apart. Residual byte 6 differed on 248319 of 248320 outputs. Layer 7's shared expert on that residual matched. The gate is Q2_0, the up is Q2_0, and the down is Q2_0. The read matched at 3.570e-08, the inject at 1.012e-06, and the router at 2.360e-08. The expert products matched at 0. The gate, the up, and the down matched at 0. The scale matched at 2.184e-07 and the shared write at 4.439e-07. The combine matched at 0 and the write at 0. The mix matched at 5.486e-08. The logits matched at 1.361e-07, integer 0, on 248320 rows. The experts were 80 358 350 436 163 341 346 165 239 22. Residual byte 6 differed on 248319 of 248320 rows. Dropping the shared expert differed on 10240 of 10240. Layer 8's GDN step on that shared residual matched. The query is Q3_K, the gate is Q3_K, and the output is IQ4_XS. The block matched at 0, the state at 7.377e-08, the convolution at 2.423e-09, and the write at 7.460e-09. 786432 of 786432 state values moved. 10240 of 30720 convolution values moved. The state started at zero. Query byte 4031108 differed on 9815 of 10240. Layer 8's ten routed experts on that residual matched. The shared expert was not added. Its down is Q5_0 and was not loaded. The gate, the up, and the down are Q2_0. The read matched at 3.453e-08, the inject at 1.117e-06, and the router at 2.455e-08. The expert products matched at 0. The combine matched at 0 and the write at 9.591e-09. The mix matched at 4.185e-08. The logits matched at 1.370e-07, integer 0, on 248320 rows. The experts were 456 308 134 387 131 181 114 263 175 183. Residual byte 2 differed on 248317 of 248320 logits. Layer 8's shared expert on that residual matched. The gate is Q2_0, the up is Q2_0, and the down is Q5_0. The read matched at 3.453e-08, the inject at 1.117e-06, and the router at 2.455e-08. The expert products matched at 0. The gate, the up, and the down matched at 0. The scale matched at 1.931e-07 and the shared write at 3.816e-07. The combine matched at 0 and the write at 9.746e-09. The mix matched at 3.255e-08. The logits matched at 1.346e-07, integer 0, on 248320 rows. The experts were 456 308 134 387 131 181 114 263 175 183. Residual byte 2 differed on 248320 of 248320. Dropping the shared expert differed on 9145 of 10240. Layer 9's GDN step on that shared residual matched. The query is Q2_0, the gate is Q2_0, and the output is Q3_K. The block matched at 1.895e-09, the state at 4.965e-08, the convolution at 0, and the write at 2.462e-09. 786432 of 786432 state values moved. 10240 of 30720 convolution values moved. The state started at zero. Query byte 2592188 differed on 9473 of 10240. Layer 9's ten routed experts on that residual matched. The shared expert was not added. Its gate is IQ4_XS and was not loaded. The gate, the up, and the down are Q2_0. The read matched at 3.408e-08, the inject at 9.187e-07, and the router at 2.836e-08. The expert products matched at 0. The combine matched at 0 and the write at 1.394e-09. The mix matched at 5.940e-08. The logits matched at 1.279e-07, integer 0, on 248320 rows. The experts were 68 427 277 376 142 506 96 322 430 375. Residual byte 2 differed on 248318 of 248320. Layer 9's shared expert on that residual matched. The gate is IQ4_XS, the up is Q2_0, and the down is Q2_0. The read matched at 3.408e-08, the inject at 9.187e-07, and the router at 2.836e-08. The expert products matched at 0. The gate, the up, and the down matched at 0. The scale matched at 1.573e-06 and the shared write at 2.680e-06. The combine matched at 0 and the write at 1.670e-09. The mix matched at 3.526e-08. The logits matched at 1.281e-07, integer 0, on 248320 rows. The experts were 68 427 277 376 142 506 96 322 430 375. Residual byte 2 differed on 248315 of 248320. Dropping the shared expert differed on 10238 of 10240. Layer 10's GDN step on that residual matched. The query is Q3_K, the gate is Q2_0, and the output is IQ4_XS. The block matched at 0, the state at 6.208e-08, the convolution at 1.719e-09, and the write at 1.397e-09. 786432 of 786432 state values moved. 10240 of 30720 convolution values moved. The state started at zero. Query byte 4031108 differed on 9477 of 10240. Layer 10's ten routed experts on that residual matched. The shared expert was not added. Its down is IQ4_NL and its up is Q3_K. Neither was loaded. The gate, the up, and the down are Q2_0. The read matched at 5.847e-08, the inject at 1.475e-06, and the router at 2.586e-08. The expert products matched at 0. The combine matched at 0 and the write at 5.794e-09. The mix matched at 5.822e-08. The logits matched at 1.212e-07, integer 0, on 248320 rows. The experts were 461 325 107 495 508 164 264 417 97 423. Residual byte 2 differed on 248309 of 248320. Layer 10's shared expert on that residual matched. The gate is Q2_0, the up is Q3_K, and the down is IQ4_NL. The read matched at 5.847e-08, the inject at 1.475e-06, and the router at 2.586e-08. The expert products matched at 0. The gate matched at 0, the up at 3.895e-09, and the down at 0. The scale matched at 1.172e-07 and the shared write at 1.295e-07. The combine matched at 0 and the write at 5.670e-09. The mix matched at 5.891e-08. The logits matched at 1.187e-07, integer 0, on 248320 rows. The experts were 461 325 107 495 508 164 264 417 97 423. Changing residual byte 3 differed on 248320 of 248320. Dropping the shared expert differed on 9842 of 10240. Layer 11's attention products matched on their own activations. The QSA block was not called. The query is Q3_K and matched at 7.729e-08, integer 7.313e-11, on 12288 rows. One code byte differed on 1 of 12288. A zero activation was 0 on all 12288. The key is IQ4_XS and matched at 0 on 512 rows. One code byte differed on 1 of 512. A zero activation was 0 on all 512. The value is IQ4_XS and matched at 0 on 512 rows. One code byte differed on 1 of 512. A zero activation was 0 on all 512. The output is Q6_K and matched at 1.141e-07, integer 0, on 2560 rows. One code byte differed on 1 of 2560. A zero activation was 0 on all 2560. Layer 11's QSA block on that shared residual matched. The query is Q3_K, the key is IQ4_XS, the value is IQ4_XS, and the output is Q6_K. The read matched at 2.072e-05 and the inject at 1.259e-06. The block matched at 0 and position 1 matched at 0. The rotated key matched at 3.513e-08. The selected ids were 0 and 1. A second draw differed on 0, and the guard word stayed. A zero residual differed on 2560 of 2560, 100.00% apart. Value byte 4088 differed on 2559 of 2560. Layer 11's attention write on that residual matched. The read matched at 2.072e-05 and the inject at 1.259e-06. The write matched at 5.608e-09. A second draw differed on 0, and the guard word stayed. A zero residual differed on 10240 of 10240, 100.00% apart. Value byte 4088 differed on 10080 of 10240. Layer 11's ten routed experts on that residual matched. The shared expert was not added. Its gate is IQ4_XS and its up is Q3_K. Neither was loaded. The gate, the up, and the down are Q2_0. The read matched at 3.405e-08, the inject at 7.294e-07, and the router at 2.709e-08. The expert products matched at 0. The combine matched at 0 and the write at 0. The mix matched at 4.102e-08. The logits matched at 1.231e-07, integer 0, on 248320 rows. The experts were 350 268 201 273 108 185 48 430 163 141. Changing residual byte 6 differed on 248316 of 248320. Layer 11's shared expert on that residual matched. The gate is IQ4_XS, the up is Q3_K, and the down is Q2_0. The read matched at 3.405e-08, the inject at 7.294e-07, and the router at 2.709e-08. The expert products matched at 0. The gate matched at 0, the up at 2.713e-09, and the down at 0. The scale matched at 6.811e-07 and the shared write at 1.594e-06. The combine matched at 0 and the write at 0. The mix matched at 6.031e-08. The logits matched at 1.250e-07, integer 0, on 248320 rows. The experts were 350 268 201 273 108 185 48 430 163 141. Changing residual byte 6 differed on 248319 of 248320. Dropping the shared expert differed on 10237 of 10240. Layer 12's GDN step on that residual matched. The query is Q3_K, the gate is Q2_0, and the output is IQ4_XS. The block matched at 0, the state at 6.290e-08, the convolution at 1.650e-09, and the write at 0. 786432 of 786432 state values moved. 10240 of 30720 convolution values moved. The state started at zero. Query byte 4031108 differed on 8893 of 10240. Layer 12's ten routed experts on that residual matched. The shared expert was not added. Its gate is Q3_K. Its up and its down are Q2_0. None was loaded. The gate, the up, and the down are Q2_0. The read matched at 4.816e-08, the inject at 4.739e-07, and the router at 2.345e-08. The expert products matched at 0. The combine matched at 0 and the write at 4.566e-09. The mix matched at 8.972e-06. The logits matched at 1.243e-07, integer 0, on 248320 rows. The experts were 28 139 421 277 201 433 52 63 288 134. Changing residual byte 2 differed on 248319 of 248320. Layer 12's shared expert on that residual matched. The gate is Q3_K, the up is Q2_0, and the down is Q2_0. The read matched at 4.816e-08, the inject at 4.739e-07, and the router at 2.345e-08. The expert products matched at 0. The gate matched at 3.792e-09, the up at 0, and the down at 0. The scale matched at 5.524e-07 and the shared write at 1.389e-06. The combine matched at 0 and the write at 4.768e-09. The mix matched at 4.196e-08. The logits matched at 1.191e-07, integer 0, on 248320 rows. The experts were 28 139 421 277 201 433 52 63 288 134. Changing residual byte 2 differed on 248316 of 248320. Dropping the shared expert differed on 10240 of 10240. Layer 13's GDN step on that residual matched. The query is Q3_K, the gate is Q3_K, and the output is Q4_K. The block matched at 1.295e-07, the state at 8.664e-08, the convolution at 4.285e-09, and the write at 6.173e-10. 786432 of 786432 state values moved. 10240 of 30720 convolution values moved. The state started at zero. Query byte 4031108 differed on 8062 of 10240. The ple tensors were not added. One expert's rows are Q2_0. That is not a loaded model. One synthetic draft step pairs a caller residual with the next token and draws one greedy id through the same block. That draft is not a loaded MTP runtime. `strata-metal-token --serve` answers on that same fixture. The check started the binary. The Python server was not started. There is no token rate.

## One memory pool

On a gaming PC the graphics card has its own memory. Strata keeps the attention and DeltaNet weights, the KV
cache and the experts that come up most often on that card, keeps all 24,576 experts in RAM, and runs the ones
the card does not hold on the CPU at the same time. The copy between the two goes over PCIe. That split is the
design in [HOW_IT_WORKS.md](HOW_IT_WORKS.md).

An Apple Silicon Mac has no second memory and no PCIe hop. The CPU and the GPU read the same RAM. A resident
weight is a shared Metal buffer, and the GPU computes it. There is nothing to gain by also computing experts on
the CPU: both would read the same pool. Weights that do not fit stay in the GGUF on the SSD and are read when a
token needs them. That is the same idea as the [low-RAM mode](MODELS.md#a-big-graphics-card-and-little-ram),
without a second copy in RAM. The 28.8 GB n-gram table already stays on the SSD.

The number that matters is Metal's `recommendedMaxWorkingSetSize`, the working set the GPU may wire. It is
smaller than the RAM sticker. `strata-metal-device` prints it and spends it. The discrete planner's fixed VRAM
pool (about 5.94 GB on a 12 GB card, beside a full expert arena in RAM) does not apply.

## What fits, as arithmetic

Expert sizes are the installer's `arena_gb` figures (decimal GB): 23.4 for the Coder, 34.0 for Q2_0, 35.5 for
IQ2_XS, 42.9 for IQ3_XXS, 50.3 for IQ3_S, 59.5 for UD-IQ4_XS, 77.0 for UD-Q4_K_XL. KV bytes per token and the
GDN recurrent state are computed from `include/strata/plan/plan.hpp` (INT8 KV, 12 QSA layers, 36 GDN layers).
The Coder's dense copy is the published ~3.4 GB, and the draft layer the published ~0.8 GB
([MULTI_GPU.md](MULTI_GPU.md)). Other sizes do not get that dense figure added: it was published for the Coder.
Prompt workspace is not in the sum.

A line that says "fits" is that sum against the working set. It is not a model loaded on a Mac. Sizes whose
experts alone exceed the working set need the SSD read described above, which is not built.

Run `strata-metal-device` for the verdict on the Mac in front of you. The working set is smaller than the
RAM sticker, and one Metal buffer is smaller than the working set, so an expert arena can fit the pool and
still have to be split across buffers.

Measured with `strata-metal-device --selftest` on an Apple M5 Max, 36 GB, Mac Studio, on 2026-10-14. The GPU
is Apple10, Metal 4, unified memory. The working set was 28.080 GiB and the longest buffer was 21.060 GiB.
The blit selftest passed. At 32,768 tokens the Coder sum was 28.145 GB and fit that working set with 2.005 GB
left; its 23.4 GB expert arena is larger than one buffer. At 262,144 tokens the same Coder sum was 31.140 GB
and did not fit. Q2_0 and every larger size were over the working set on experts alone. Prompt workspace is
still not in those sums, and no model was loaded.

## What is built

`strata-metal-device` talks to Metal: the GPU name, the family, the working set, and a 1 MiB shared-buffer blit
that has to come back unchanged. It does not compile a shader.

`strata-metal-bf16-gemv` loads one BF16 tensor from a GGUF into a shared buffer and multiplies it by a vector.
The weight row is contiguous bf16, shape `[n_in, n_out]`. The host reference is a double sum of the same bits.
The bound is 1e-5.

Measured with `strata-metal-bf16-gemv --selftest` on the same Apple M5 Max, on 2026-10-14. The shaders compiled
as Metal 4.1 with safe math and the precise math functions. The worst relative error was 3.88e-07, on
`ssm_alpha` `[2560, 48]`, the path with one thread per row. The simdgroup path and the 256-thread split were
tighter on every shape, including `indexer.k_proj` `[2560, 128]`, `indexer.q_proj` `[2560, 512]` and
`ple_value` `[2560, 2560]`. An fp16 activation, the wrong contract for these weights, sat 0.14% to 0.16% away
from the bf16 activation. The GGUF was one tensor, `blk.0.indexer.k_proj.weight`, type BF16. The bytes in the
mapped file matched the bytes written, and the product of those bytes matched the reference at 8.50e-08.

`strata-metal-gdn` checks the delta-net step, the short convolution, the L2 norm, the beta sigmoid and the
output norm, at the model's head size (state 128, 16 key heads, 48 value heads). Apple GPUs have no fp64, so
the two norms accumulate the sum of squares as a pair of floats. The step bound is 1e-5. The convolution and
the norms are 1e-6. Those are the same bounds as the CUDA test.

Measured with `strata-metal-gdn --selftest` on the same Mac, the same day. The output was 1.969e-07 from the
double reference, and the state, read back in `(S, h_v, S)` order with `j` fastest, was 4.686e-08. The
convolution output was 1.860e-08 and the slid state matched exactly. L2 was 1.573e-08, the output norm
3.151e-08, and the beta sigmoid 3.648e-09. A zero state with a zero gate and a zero beta stayed zero. The
wrong readings the test is there to catch (interleaved heads, decay after the update, a transposed convolution
weight, epsilon on the mean, SiLU instead of sigmoid) were each more than 5% away.

`strata-metal-qsa` checks decode attention over a paged KV cache. Head dim 256, 24 query heads, 2 KV heads,
chunks of 64. The cache is fp16, int8, q4_0, or int8 keys with q4_0 values. The bound is 1e-5 against a double
softmax of the same stored values. The scale is `1/sqrt(256)`.

Measured with `strata-metal-qsa --selftest` on the same Mac, the same day. The simdgroup was 32. The worst
relative error was 1.942e-07, on 200 fp16 cells, and page sizes 1, 4 and 64 agreed bitwise. Int8 was 1.708e-07,
q4_0 was 1.700e-07, and int8 keys with q4_0 values were 1.596e-07. A batch of two queries, with different
selections, was 1.698e-07, and the two outputs were 140.03% apart. One live cell matched the decoded value
exactly, including a page that was not the logical page, a masked sibling chunk, and each quantized layout. An
empty selection wrote zeros over a buffer filled with 7s. With a loud cell selected, head 0 dimension 0 was
9.8800 (rel 4.061e-09); with that cell left out it was 1.000000. The wrong readings the test is there to catch
(query head `h % 2`, dropping `1/sqrt(256)`, and ignoring a reversed page table on a 40-cell selection) were
71.31%, 749.44% and 141.36% apart. A page marked -1 matched attention that skips those cells, at 1.884e-07.

`strata-metal-iq` checks the Coder expert product. The pack is named IQ1_M. Its kept experts are stored like
IQ3_S: gate and up rows are IQ2_S, IQ3_XXS, IQ3_S or IQ4_XS, and down rows are IQ4_NL or Q2_0. The activation
is q8_1. One expert is a gate product, an up product, SiLU of the gate times the up row, a q8_1 quantize of
that hidden row, then the down product. The bound is 1e-5 against the integer partial the kernel loads,
summed in the same 32-lane order. IQ1_M itself (type 29) is not this product.

Measured with `strata-metal-iq --selftest` on the same Mac, the same day. The q8_1 blocks matched the host
block bitwise, including a scale clamped at 65504. Every matrix-vector product matched the reference exactly,
relative error 0: IQ2_S, IQ3_XXS, IQ3_S and IQ4_XS at 40 by 512 with 3 columns and at one block; IQ4_NL at
40 by 256 and at one block of 32; Q2_0 at 40 by 256 and at one block of 64. The four expert pairs, 256 by 64
with two tokens, also matched exactly: IQ2_S / IQ4_NL, IQ3_XXS / Q2_0, IQ3_S / IQ4_NL, and IQ4_XS / Q2_0.
IQ3_S / IQ4_NL at 1280 by 128 matched exactly, with some lanes adding two partials. Dropping the IQ3_S
integer scale was 3000.00% apart. A Q2_0 code that forgets the -1 was 57.33% apart. The two tokens were
141.28% apart. Swapping gate and up was 104.41% apart, and those swapped weights still matched the reference.
Zero tokens left a buffer of 7s unchanged.

`strata-metal-sgemv` checks the dense projection the GDN layer calls. The weight is the canonical S form: a
code, a float scale, and sometimes a float offset. That is not the ggml block layout the expert product dots.
attn_qkv is IQ4_XS (4-bit, group 32, bias -8, no offset), attn_gate is Q4_K (4-bit, group 32, with an offset),
and ssm_out is Q5_K (8-bit, group 32, with an offset). The Iq4Nl codebook is the same 16 values as IQ4_NL.
The activation is Q8_K, 292 bytes per 256 values, or Q8_0, 34 bytes per 32, when the width is not a multiple
of 256. The value is the code times the scale, plus the offset, and that sum times the activation. The bound
is 1e-5 against a double sum of the stored bytes. Q8_K runs on the GPU. Its scale is -127/max and it rounds
half to even. Q8_0 runs on the host: this GPU has no fp64, and the rule divides in double before rint. The
stored Q8_0 scale is fp16. A 2-bit code has no Q8_K contract.

Measured with `strata-metal-sgemv --selftest` on the same Mac, the same day. The worst relative error was
3.797e-07, on the Q5_K product summed by one thread, 64 by 512. Q8_K matched the host bytes, including a zero
block and exact .5 ties. Of 522240 ties, half-to-even differed from half-away on 260096 (49.8%). A block
starting at 127 stored scale -1, code -127, and block sum -127. nearest_int of 0.5 was 0, and of 2.5 was 2.
Using -128 instead of -127 was 0.918% of the magnitude, and 171824 bytes differed. Q8_0 matched, including a
block starting at 1: code 127 and scale f16(1/127). Dividing by that fp16 scale instead of the f32 scale
changed 965 bytes. The warp sums were 1.097e-07 (IQ4_XS), 9.390e-08 (Q4_K), 1.090e-07 (Q5_K), and 8.179e-08
(IQ4_NL). The one-thread sums were 3.446e-07, 3.412e-07, 3.797e-07, and 3.469e-07. Q8_0 activations on the
same weights were 7.385e-08, 8.651e-08, 9.299e-08, and 7.423e-08. A Q8_K activation and an fp16 activation
were 0.66%, 1.09%, 1.41%, and 0.66% apart. Decoding the Iq4Nl codes as affine was 685.32% apart. Width 640,
which is not a multiple of 256, was 1.069e-07 on Q8_0 and Q4_K. An fp16 2-bit row of width 32 was 1.863e-08.
Output widths 1, 7, 9, and 61 stayed within 8.181e-09, and 64 guard words past the last row stayed
0x7FC0DEAD. Q8_K refused a 2-bit code and a width of 640.

`strata-metal-elementwise` checks the glue between those projections. The gate is softplus(alpha + dt) times
ssm_a, and ssm_a is negative, so the state decays. softplus returns its input above 20. SiLU is the conv
output, before the split into q, k, and v. The scale is the extra 1/sqrt(S)
on q. The bf16 conversion is what ssm_alpha and ssm_beta read. This GPU has no fp64, and Metal has no log1p,
so SiLU is the precise f32 exp and softplus is log(1 + y) written to keep a small y. The same binary
checks one packed embedding row, the add of the two expert halves, and the QSA norm. The embedding
multiplies the code by its scale and then adds the offset, as two roundings. Widths are 2, 4, and 8,
with bias -1, -7, and -16. The expert add is not the residual. The QSA norm divides by the root of the
mean of squares and then multiplies by w. A null weight is legal. It is not the DeltaNet L2.

Measured with `strata-metal-elementwise --selftest` on the same Mac, the same day. The gate, 48 heads, was
2.989e-08 from the double reference. The bound is 1e-6. Sixteen heads were above 20 and five were above 88.
Without the branch, five heads were non-finite, and the two readings first differ at x = 88.8. exp(gate) was
below 1 on every head. Two tokens, with dt indexed by the head, were 2.419e-08. Zero tokens left a 7. SiLU
on 4096 values was 2.966e-08. The bound is 1e-7, and the precise f32 exp met it. Scale by 1/sqrt(128) matched
all 1024 products, and 16 words past the end stayed put. f32 to f16 matched the shared conversion, including
1e30 and -1e30 saturating to inf, 65504 staying finite, 65520 rounding to inf, and 2^-24 staying the smallest
subnormal. f32 to bf16 matched, including a NaN: 0x7FFFFFFF stayed 0x7FFF and 0x7F800001 stayed 0x7FC0.
The embedding matched the separate multiply and add on 108 cases: widths 2, 4, and 8, groups 16, 32,
and 64, lengths 320 and 2560, three rows, with and without an offset. There were 0 bit mismatches and
0 guard failures. An fma on the same inputs differed on 8141 values. add_inplace matched all 1024 sums,
and 16 words past the end stayed put. The QSA norm, 24 by 256, was 3.630e-08 from the f64
mean-of-squares reference. The bound is 1e-6. The worst element was 2.035e-07. Dividing by the root of
the sum, with no weight, was 89.36% apart. Leaving the weight out was 70.20% apart. A null weight
matched, 0 of 6144. Two rows, the k-norm shape, were 5.073e-08, and 16 words past the buffer stayed
put. A row of 100 was 7.435e-08, and 32 words past it stayed put. The QSA decode kernel does not call
this norm yet.

`strata-metal-gdn-layer` runs one GDN layer in the non-fused order: Q8_K quantize, attn_qkv, the conv,
SiLU on the whole conv output, L2 on q and k, 1/sqrt(S) on q only, ssm_alpha and ssm_beta, the beta
sigmoid, the softplus gate, the step, attn_gate, the output norm, and ssm_out. attn_qkv is IQ4_XS,
attn_gate is Q4_K, and ssm_out is Q5_K, so each activation is Q8_K. Conv weights are channel-major.
Tap i of channel c is `kW[c * d_conv + i]`. ssm_norm is S floats, the same vector on every head.
q and k are copied into their own buffers, because L2 reads from the start of the buffer it is given.
There is no new shader. The weights in the check are synthetic. No GGUF is loaded.

Measured with `strata-metal-gdn-layer --selftest` on the same Mac, the same day. The bound is 1e-5.
The reduced shape is n_embd 256, state 128, 2 key heads, 4 value heads, head ratio 2. The model head
ratio is 3, and the model embedding is 2560. d_conv is 4, so the conv width is 1024. The worst relative
error was 3.798e-07, on ssm_out. The qkv projection was 8.446e-08. The conv output, after SiLU, was
9.322e-08. q, after L2 and the scale, was 1.170e-07. k was 8.746e-08. v, which is not normalised, was
9.919e-08. The gate relative error was 0. The step output was 2.678e-07 and the state was 8.299e-08.
The output norm was 2.523e-07. The conv state was 7.506e-08. The same order with 16 key heads and 48
value heads, n_embd still 256 and the conv width 10240, had its largest error at 2.826e-07 on the
output norm. The step there was 2.429e-07 and ssm_out was 6.525e-08. Forgetting the q scale was
1031.37% apart on the step output. L2-normalising v was 19.63% apart. Leaving SiLU out was 40.39%
apart. SiLU on q and k only was 37.87% apart. SiLU on the projection, before the conv, was 40.41%
apart. Interleaved heads were 59.35% apart. Scaling k as well as q was 34.02% apart. Omitting the
softplus branch was non-finite. Two heads were above 20 and one was above 88. Sigmoid against SiLU
on z was 183.47% apart. A transposed conv weight, run on the GPU, was 73.36% apart. The conv state
appended the projection, and that column was 100.13% away from the SiLU output. S above 128 was
refused. A value width of 128, which is not a multiple of 256, was refused. An embedding of 200
was refused. The same order at the model embedding, n_embd 2560, 16 key heads, 48 value heads,
conv width still 10240, had its largest error at 2.856e-07 on the output norm. The qkv projection
was 9.406e-08. The conv output, after SiLU, was 1.029e-07. q, after L2 and the scale, was
1.165e-07. k was 1.139e-07. v, which is not normalised, was 1.052e-07. The gate was 3.115e-08.
The step output was 2.352e-07 and the state was 1.227e-07. ssm_out was 2.862e-08. The conv state
was 9.028e-08. The stored column was 100.47% away from the SiLU output. The weights are synthetic.
No GGUF was loaded. This is not a loaded model. There is no token rate.

`strata-metal-gr` checks the gated residual for one token. The residual is hc streams, each n_embd
wide. The norm is per stream, and w_norm is the stored (1 + w). The down projection is bf16, then
silu(dot / hc), with the division inside the silu. The up projection is a sigmoid, and mixed is the
mean over streams of the unrounded norm output times that sigmoid. inject is a third projection. A
null inject is the final mixer, and that buffer is left untouched. gr_write adds the same block
output onto every stream, weighted by 2*sigmoid(inject / hc), so a zero injection adds it once.
The default path rounds xn and lo to bf16. The f32 path keeps them in f32. The weights stay bf16
either way. Native MMVF and the fused multi-token read are not in this port.

Measured with `strata-metal-gr --selftest` on the same Mac, the same day. The bound on mixed and on
inject is a relative error of 1e-4. The reduced shape is n_embd 256, 4 streams, rank 32. A norm of
the whole stack was 86.11% apart. /hc outside the silu was 7.19% apart. Swapping SiLU and sigmoid
was 87.21% apart. A sum instead of the mean was 300.00% apart. The two activation contracts were
0.10% apart. The bf16 read was 5.840e-08 on mixed, worst element 8.725e-06, and 1.280e-07 on
inject. The f32 read was 2.751e-07 on mixed, worst element 5.746e-05, and 5.688e-07 on inject. A
transposed weight was 59.54% apart. A null injection left the buffer. A zero injection matched a
plain add, worst term error 5.791e-08. That bound is 1e-6. Leaving off the 2 was 50.00% apart on
the delta, and the kernel was 3.887e-07 from the centred gate. A non-zero injection was 6.144e-08.
The model shape, 2560 by 4 by 320, was 8.469e-08 on mixed for bf16 activations, worst element
1.020e-04, and 8.770e-07 on inject. f32 activations on that shape were 3.000e-07 on mixed, worst
element 3.101e-04, and 1.677e-06 on inject.

`strata-metal-router` checks the MoE router. The logits are already f32. The kernel takes a softmax
over every expert, keeps the top 10, breaks a tie toward the lower index, and divides those weights
by their sum. The sum is clamped at 2**-14. This GPU has no fp64, so the exponential is the precise
f32 exp and the sum is a pair of floats, added in expert order. The BF16 product that fills the
logits is a separate kernel, and this check does not call it. The native 512-by-10 router is not in
this port. More than 1024 experts is refused, because the probabilities sit in threadgroup memory.

Measured with `strata-metal-router --selftest` on the same Mac, the same day. The bound is 1e-5.
64 tokens, 512 experts, top 10. Every id matched. The worst weight error was 2.244e-07, on random
normal logits. All-equal logits were bitwise, and the ids were 0 through 9. A 12-way tie was
2.177e-07. One dominant expert was 5.960e-08. The top-10 probability sum is at least k/n = 0.01953,
which is 320x the clamp, so the clamp cannot fire on this shape. Breaking a tie toward the higher
index changed 100.00% of the all-equal ids. Leaving the renorm out was 98.05% apart on those
weights. Seven experts, top 3, was 1.208e-07. 1024 experts was 1.253e-07. k = 64 was 3.019e-07.
k = 65, 1025 experts, a negative count, and k above the expert count were refused. Zero tokens left
the id buffer.

`strata-metal-sampler` checks one token, or a row per draft, from f32 logits. The chain is penalties,
then top-k, then top-p, then min-p, then temperature, then one Philox draw. Temperature does not move
the shortlist. Greedy, and a temperature of 0, is the argmax and consumes no random number. Ties keep
the lower index. A non-positive logit is multiplied by the repeat penalty. A positive logit is divided
by it. The presence penalty is applied once. The window is the tail of that row's history. A penalty
bitmap covers 248320 logits, the model vocabulary. A wider row with penalties counts the window
directly. The cluster argmax, the split top-k, and CUDA graphs are not in this port.

Measured with `strata-metal-sampler --selftest` on the same Mac, the same day. The token ids are exact.
Greedy argmax, 512 logits and 4 tokens, differed on 0 of 4. Temperature before the shortlist kept 12
survivors where the chain kept 16, and the kernel stayed the argmax, 0 of 4. Seed 1 and seed 999999
agreed. The multiply rule picked token 1. Dividing every logit picked token 0. The presence penalty,
applied once, picked token 1, 0 of 2. Temperature 0 and greedy each returned the argmax, 0 of 4. A
penalised sample picked tokens 3 and 2. The clean rows picked 0. min_p of 0, 0.5, and 0.9 kept 12, 4,
and 2 survivors, summed over 2 rows. The clamped window picked token 0. The full history picked token
3. One penalties stage drew token 0 on 0.1040 of 8000 draws, against an expected share of 0.1050. The
bound on that share is 0.014. Every one of the 8000 ids matched. top_p before min_p drew token 2 on 58
of 256 draws, and 0 differed. penalty_last_n of 0 left a stale history unused, 0 of 4, sampled and
greedy. Per-row history matched for 1, 2, 4, and 8 tokens and for history lengths 1, 64, 1024, and
4096. 21 of 60 drafted rows differed under row 0's history. History ids outside the vocabulary matched.
top_k of 64, 0, 100, and -3 matched the same reference. The draw counter is the row index, including a
counter above 2^32. Under ties, 0 of 112 list positions differed at vocabularies 1000 and 248320, and
20 of those positions are the empty-round id 0. 0 of 192 sampled draws differed at vocabulary 512, and
119 of those picks tied another kept token. Vocabulary 248321 with penalties picked token 9. A negative
count was refused. Penalties without a history were refused. Zero tokens left the id.

`strata-metal-moe-combine` checks the last sum of a MoE block. The routed rows are weighted. The
shared output is added plain, and a null shared omits it. The weights are already renormalised.
This kernel does not divide them again. This GPU has no fp64, so each product is kept as a pair of
floats and the sum is that pair, added in expert order. The native combine is not in this port.

Measured with `strata-metal-moe-combine --selftest` on the same Mac, the same day. The bound is
1e-6. n_embd 256, k 10. The routed sum left unweighted was 90.68% apart. Weighting the shared
output by the first router weight was 93.32% apart. The kernel matched the double reference,
relative error 0. The model shape, 2560 by 10, also matched, relative error 0. A null shared was
98.49% apart. 16 words past the end stayed put. k = 65 and a negative count were refused. A zero
length left the buffer.

`strata-metal-s2-gemv` checks a 2-bit row, group 64, bias -1, dotted with a Q8_0 activation. The
fp16 activation product is a different kernel.

Measured with `strata-metal-s2-gemv --selftest` on the same Mac, the same day. The bound is 1e-4.
2560 by 64, 32 threads per row. Worst relative error 1.424e-05. Width 64 was 2.033e-07. Width 256
with 64 threads was 5.962e-05. The same weights with an fp16 activation were 0.6173% apart. 16 words
past the end stayed put. A width that is not a multiple of 64 was refused. A thread count that is
not a power of two was refused. A negative count was refused. Zero length left the buffer.

`strata-metal-shared-expert` checks one shared expert. The three weights are 2-bit, group 64, bias
-1, and the activation is Q8_0. SiLU is on the gate projection. The down projection reads the Q8_0
image of that product. The scalar gate is the existing BF16 product, one row, then a sigmoid of that
dot, multiplied onto every output. The hidden state in that dot is the original one. Native
projections are not in this port. A down weight that is not 2-bit is not this function.

Measured with `strata-metal-shared-expert --selftest` on the same Mac, the same day. The bound is
1e-2. n_embd 256, n_ff 64. SiLU on the up projection was 100.00% apart. The SwiGLU intermediate
peaked at 677.2, under 65504. Worst relative error 3.464e-04, and every output was finite. The mean
absolute reference was 4716.8802. The reference quantizes that intermediate with a double scale. The
kernel stores the scale as fp16. 16 words past the end stayed put. A width that is not a multiple of
64 was refused. A negative count was refused. Zero length left the buffer.

`strata-metal-qsa-gate` checks the last multiply of a QSA head. The query block is two head-widths.
The gate is the second half, and the function is a sigmoid. The fp16 result is that product rounded
by the shared conversion. The decode-attention kernel does not call it.

Measured with `strata-metal-qsa-gate --selftest` on the same Mac, the same day. The f32 bound is
1e-6. The fp16 bound is 1e-3. 24 heads, head dim 256. Reading the first half was 56.26% apart.
SiLU instead of a sigmoid was 85.63% apart. The f32 product was 3.912e-08 from the double
reference. The fp16 product was 1.763e-04, and 0 of 6144 bits differed from the shared conversion
of that f32 product. 4 heads by 8 was 3.925e-08. 16 words past the end stayed put. A negative
count was refused. Zero length left the buffer.

`strata-metal-indexer` appends one raw indexer key. A cell that finishes a block is averaged,
normed, and rotated at that block's first cell. The spare key is the norm of cell 0. The
decode-attention kernel does not call it.

Measured with `strata-metal-indexer --selftest` on the same Mac, the same day. The bound is
1e-6. idx_dim 128, block 4, 64 rotated dims, 10 cells, pos_base 100. Rotating at the block's
last cell was 6.67% apart. Norm then pool was 48.93% apart. The spare key taken from the last
raw key was 49.08% apart. Rotating the raw keys was 22.41% apart. Equating a cell's position
with its index was 14.06% apart. The spare key matched on 0 of 128 values. The pooled blocks
were 1.252e-07 from the double reference. The incomplete tail was 143.86% apart from that spare
key. A block of 2, 16 dims, 8 rotated, 5 cells, pos_base 7, matched the spare key on 0 of 16
and the blocks at 7.585e-08. The last complete block recorded position 104, and the next cell
left it at 100. 16 words past the end stayed put. A negative count, a block shorter than 2, an
odd rotation width, and a negative cell index were refused. The mean and the sum of squares run
on the CPU in IEEE double, because a shader has no fp64. The tail store and the rotation run on
the GPU.

`strata-metal-qsa-index` scores one pooled indexer row per block. Each head's dot is a pair of
floats. ReLU is per head, then the heads are summed, then the block bias is added, then the pair
collapses to one float. An incomplete tail block adds 1e9 after that collapse. The
decode-attention kernel does not call it.

Measured with `strata-metal-qsa-index --selftest` on the same Mac, the same day. The bound is
1e-6. 4 heads, dim 128, block 4, 10 cells. ReLU of the sum, instead of ReLU per head, was 51.32%
apart. Dropping the ReLU was 51.32% apart. Dropping the bias was 77.48% apart. The cell scores
matched the double reference, relative error 0. 2 heads, dim 8, block 2, 5 cells: those three
readings were 13.39%, 51.55%, and 49.05% apart, and the scores again matched. With a null bias
and zero inputs, n_kv 10 and n_kv 2051, the finished cells were exactly 0 and only the incomplete
tail was exactly 1e9. 16 words past the end stayed put. A negative count, a block count that is
not n_kv / r, and more than 32 indexer heads were refused.

`strata-metal-topk` keeps the largest cells. Ties keep the lower index. The ids come out
ascending. The width is the smaller of n_kv and idx_top_k + idx_block - 1, so 2048 and a block
of 4 select 2051 cells once n_kv is at least that. The decode-attention kernel does not call it.

Measured with `strata-metal-topk --selftest` on the same Mac, the same day. The ids are exact.
Seed 20260916, top_k 2048, block 4. n_kv of 1, 7, 2048, 2051, 2052, 5000, and 32768 matched an
independent sort. 2052 equal scores kept cells 0 through 2050. A 100-cell tie across the cut
matched. n_kv 2048 and 2051 selected every cell. n_kv 2052 dropped the one smallest score. A NaN
was not selected. -0 tied with +0. Zero length left the buffer. n_kv above 32768, a cap below
the width, and a negative count were refused. 16 words past the end stayed put.

`strata-metal-kv` appends one cell into a paged fp16 cache and gathers cells back out. The
layout is page, then kv head, then slot, then head dim. The page table is not the identity. A
negative physical page is not written. The conversion is the shared fp16 round. The
decode-attention kernel does not call it.

Measured with `strata-metal-kv --selftest` on the same Mac, the same day. 1100 cells, 2 kv
heads, head dim 256, reversed page table. Page size 1: 0 of 563200 wrong, and ignoring the page
table was 99.99% apart. Page size 4: 0 of 563200 wrong, ignoring the table was 100.00% apart,
and cell-major inside the page was 74.99% apart. Page size 512: 0 of 563200 wrong, ignoring the
table was 99.91% apart, and cell-major inside the page was 99.76% apart. The gather of cells
0, 1, 2, 3, 5, 8, 13, 21, 34, 39, 1099, and 700 was 0 wrong at each page size. A negative page
left the sentinel. 16 values past the end stayed put. A negative position, a head dim that is
not a multiple of 4, a page outside the table, a negative cell, and a negative physical page on
the gather were refused. A zero-length gather left the scratch.

`strata-metal-rope` rotates the first n_rot values of each row. Pair i is paired with i + n_rot/2.
The tail is copied. The cos and sin table is built on the host in float64. The indexer rotates one
pooled row with its own kernel.

Measured with `strata-metal-rope --selftest` on the same Mac, the same day. The bound on the
rotation is 1e-6, against the largest absolute input in the row. Head dim 256, 64 rotated values,
192 rows, theta 1e7. The table matched the float64 frequencies on 0 entries. The rotation matched
on 0 bits, relative error 0. The tail was unchanged. Pairing adjacent values instead was 43.27%
apart on the rotated prefix. A 1 at dim 0, position 7, moved dim 32 by 0.656987 and left dim 1 at
0. In place matched a separate buffer on 0 values. Position 0 left the row unchanged. Head dim 16,
8 rotated values, 3 rows: the table and the rotation again matched on 0 bits, and adjacent pairing
was 34.82% apart. 16 words past the end stayed put. Zero rows left the buffer. A negative row
count, an odd rotation width, a rotation wider than the row, a position past the table, and a
negative position were refused.

`strata-metal-qsa-layer` runs one attention block. The order is the indexer key, K and V, norm and
rotate K, append K and V, the query, the first half normed and rotated, the indexer query, the cell
scores, the top-k, paged attention, the gate, then the output projection. No new shader. K, V, Q, and
the output are synthetic 4-bit affine weights with an offset. The two indexer projections are bf16.
The activation is Q8_K. The artifact's attention weights are not all one type, and its top-k is 2048.

Measured with `strata-metal-qsa-layer --selftest` on the same Mac, the same day. The bound is 1e-4.
n_embd 256, 8 cells, 24 heads, 2 kv heads, head dim 256, 64 rotated values, indexer dim 64, 4 indexer
heads, block 4, top-k 2, page size 4, rope base 100. The norm weights past those 64 rotated values are
0, so the dot sits in the rotated pairs. Worst relative error 5.386e-07. The selected ids matched on
every cell, 0 differ. The pool differed on 1 key value and 5 value values, of 4096. Ignoring the page
table was 80.20% apart. Leaving K unrotated was 52.16% apart on the attention output. The gate's first
half was 15.73% apart. Mapping a query head with modulo instead of integer division was 28.77% apart.
16 words past the end stayed put. A negative position was refused. An embedding that is not a multiple
of 256 was refused. The same block at the model embedding, n_embd 2560 and indexer dim 128, kept
64 rotated values, 8 cells, 24 heads, 2 kv heads, head dim 256, 4 indexer heads, block 4, top-k 2,
page size 4, and rope base 100. The norm weights past those 64 rotated values are 0 on the head
and on the indexer row. Worst relative error 1.660e-07. The selected ids matched on every cell,
0 differ. The pool differed on 0 key values and 4 value values, of 4096. Ignoring the page table
was 82.45% apart. Leaving K unrotated was 12.37% apart on the attention output. The gate's first
half was 91.60% apart. Mapping a query head with modulo instead of integer division was 25.44%
apart. 16 words past the end stayed put. The weights are synthetic. The artifact's top-k is 2048,
and this check is not that. No GGUF was loaded. This is not a loaded model. There is no token rate.

`strata-metal-token` draws one greedy id. The embedding row replaces every residual stream. One block
is the attention gated residual, the QSA attention block, the write, the feed-forward gated residual,
the router, one routed expert per slot, the 2-bit shared expert, the combine, and the write. The
output mix is a gated residual with no injection, then a Q8_K projection. No new shader. The routed
experts are IQ4_XS gate and up with a Q2_0 down. The shared expert's down scales run from 8 to 12.9
so that term is visible in the sum. The attention fixture matches the QSA block above, with page size
2 and a rope table of 4 positions.

Measured with `strata-metal-token --selftest` on the same Mac, the same day. n_embd 256, 4 streams,
rank 32, 8 experts, top 2, vocabulary 32, greedy. The first token is 3. The logits differed from the same
kernel sequence on 0 of 32 values, relative error 0. The routed ids matched, 0 differ. The sampled id
was 15, the argmax. Writing the embedding into one stream was 36.13% apart on the logits. Skipping
the attention write was 42.96% apart. Routing every slot to expert 0 was 68.11% apart. Weighting the
shared expert by the first router weight was 8.67% apart on the combine output. 16 words past the
logits stayed put. The second token, the sampled id at the next cell, matched on 0 logits and sampled
15 again. A negative token was refused. An embedding that is not a multiple of 256 was refused.

The same run draws one greedy token at the model embedding. n_embd 2560, indexer dim 128, 64 rotated
values, 24 heads, 2 kv heads, head dim 256, 4 indexer heads, block 4, top-k 2, page size 2, 4 cells.
4 streams, rank 32, 8 experts, hidden 64, vocabulary 32. The routed experts stay IQ4_XS gate and up
with a Q2_0 down. The shared expert stays 2-bit, and its down scales run from 8 to 12.9. The norm
weights past those 64 rotated values are 0 on the head and on the indexer row. The first token is 3.
The logits differed from the same kernel sequence on 0 of 32 values, relative error 0. The routed ids
matched, 0 differ. The sampled id was 27, the argmax. Skipping the attention write was 10.71% apart
on the logits. 16 words past the logits stayed put. Rank, expert count, hidden size, and top-k stayed
on the small fixture. The weights are synthetic. No GGUF was loaded. This is not a loaded model.
There is no token rate.

The same run draws one greedy token at the model residual rank. n_embd 2560, hc_lr 320, 4 streams,
indexer dim 128, 64 rotated values, 24 heads, 2 kv heads, head dim 256, 4 indexer heads, block 4,
top-k 2, page size 2, 4 cells. 8 experts, hidden 64, vocabulary 32. The routed experts stay IQ4_XS
gate and up with a Q2_0 down. The shared expert stays 2-bit, and its down scales run from 8 to 12.9.
The norm weights past those 64 rotated values are 0 on the head and on the indexer row. The first
token is 3. The logits differed from the same kernel sequence on 0 of 32 values, relative error 0.
The routed ids matched, 0 differ. The sampled id was 30, the argmax. Skipping the attention write
was 38.60% apart on the logits. 16 words past the logits stayed put. Expert count, hidden size, and
top-k stayed on the small fixture. The weights are synthetic. No GGUF was loaded. This is not a
loaded model. There is no token rate.

The same run draws one greedy token at the model routed-expert hidden size. n_embd 2560, hc_lr 320,
4 streams, indexer dim 128, 64 rotated values, 24 heads, 2 kv heads, head dim 256, 4 indexer heads,
block 4, top-k 2, page size 2, 4 cells. 8 experts, routed hidden 640, vocabulary 32. The routed
experts stay IQ4_XS gate and up with a Q2_0 down. The shared expert stays 2-bit at hidden 64, and
its down scales run from 8 to 12.9. The norm weights past those 64 rotated values are 0 on the head
and on the indexer row. The first token is 3. The logits differed from the same kernel sequence on
0 of 32 values, relative error 0. The routed ids matched, 0 differ. The sampled id was 15, the
argmax. Skipping the attention write was 248.70% apart on the logits. 16 words past the logits
stayed put. Expert count, shared-expert hidden size, and top-k stayed on the small fixture. The
weights are synthetic. No GGUF was loaded. This is not a loaded model. There is no token rate.

The same run draws one greedy token at the model expert count. n_embd 2560, hc_lr 320, 4 streams,
indexer dim 128, 64 rotated values, 24 heads, 2 kv heads, head dim 256, 4 indexer heads, block 4,
top-k 2, page size 2, 4 cells. 512 experts, routed hidden 640, vocabulary 32. The routed experts
stay IQ4_XS gate and up with a Q2_0 down. The shared expert stays 2-bit at hidden 64, and its down
scales run from 8 to 12.9. The norm weights past those 64 rotated values are 0 on the head and on
the indexer row. The first token is 3. The logits differed from the same kernel sequence on 0 of 32
values, relative error 0. The routed ids matched, 0 differ. The sampled id was 28, the argmax.
Skipping the attention write was 107.96% apart on the logits. 16 words past the logits stayed put.
Shared-expert hidden size, vocabulary, and top-k stayed on the small fixture. The weights are
synthetic. No GGUF was loaded. This is not a loaded model. There is no token rate.

The same run draws one greedy token at the model shared-expert hidden size. n_embd 2560, hc_lr 320,
4 streams, indexer dim 128, 64 rotated values, 24 heads, 2 kv heads, head dim 256, 4 indexer heads,
block 4, top-k 2, page size 2, 4 cells. 512 experts, routed hidden 640, shared hidden 640,
vocabulary 32. The routed experts stay IQ4_XS gate and up with a Q2_0 down. The shared expert stays
2-bit. Its down scales run from 1.0 to 5.9. Scales from 8 left the attention write 4.07% apart, so
this check uses the smaller base. The norm weights past those 64 rotated values are 0 on the head
and on the indexer row. The first token is 3. The logits differed from the same kernel sequence on
0 of 32 values, relative error 0. The routed ids matched, 0 differ. The sampled id was 30, the
argmax. Skipping the attention write was 25.92% apart on the logits. 16 words past the logits
stayed put. Vocabulary and top-k stayed on the small fixture. The weights are synthetic. No GGUF
was loaded. This is not a loaded model. There is no token rate.

The same run draws one greedy token at the model vocabulary. n_embd 2560, hc_lr 320, 4 streams,
indexer dim 128, 64 rotated values, 24 heads, 2 kv heads, head dim 256, 4 indexer heads, block 4,
top-k 2, page size 2, 4 cells. 512 experts, routed hidden 640, shared hidden 640, vocabulary
248320. The routed experts stay IQ4_XS gate and up with a Q2_0 down. The shared expert stays
2-bit, and its down scales run from 1.0 to 5.9. The embedding and the output projection are 4-bit
with an offset, and the output activation is Q8_K. The norm weights past those 64 rotated values
are 0 on the head and on the indexer row. The first token is 3. The logits differed from the same
kernel sequence on 0 of 248320 values, relative error 0. The routed ids matched, 0 differ. The
sampled id was 52472, the argmax. Skipping the attention write was 9.56% apart on the logits.
16 words past the logits stayed put. Top-k stayed at 2. The weights are synthetic. No GGUF was
loaded. This is not a loaded model. There is no token rate.

The same run draws sixteen greedy cells at that vocabulary. n_embd 2560, hc_lr 320, 4 streams,
indexer dim 128, 64 rotated values, 24 heads, 2 kv heads, head dim 256, 4 indexer heads, block 4,
top-k 2, page size 2, 16 cells. The last cell has n_kv 16. The selection width is min(n_kv, top-k
+ block - 1), so top-k 2 keeps 5 cells and top-k 2048 keeps 16. 512 experts, routed hidden 640,
shared hidden 640, vocabulary 248320. The routed experts stay IQ4_XS gate and up with a Q2_0 down.
The shared expert stays 2-bit, and its down scales run from 0.05 to 0.54. The key plane is zero.
The value plane is 4-bit with bias -8, nibbles 7 and 9, group scale 400, and offset 0. The output
projection and the head use bias -8. Feeding each sample back repeated one id, and that wider
selection was 0.01% apart. Distinct inputs on the uncentered value rows were 0.01% apart as well.
The inputs are 3, 15003, 30003, 45003, 60003, 75003, 90003, 105003, 120003, 135003, 150003,
165003, 180003, 195003, 210003, and 225003. The logits differed from the same kernel sequence on
0 of 248320 values, relative error 0. The routed ids matched, 0 differ. The sampled ids were
77714, 25900, 2551, 228616, 19310, 165164, 28183, 19310, 2551, 2551, 113960, 2551, 25900, 119547,
207094, and 119547, each the argmax. Top-k 2 kept cells 4, 12, 13, 14, and 15. Replaying those
inputs at top-k 2048 was 165.49% apart on the last logits. 16 words past the logits stayed put.
The weights are synthetic. No GGUF was loaded. This is not a loaded model. There is no token rate.

The same check draws one draft step. The caller's residual is left as it was. The next token is
gathered, normalized, and projected. The whole residual is normalized as one row, and each stream
of that row is projected by the same hidden weight. The layer residual is the sum, with the
embedding projection added on every stream. Both projections are 4-bit S-form with an offset, and
the activation is Q8_0. The hidden scales start at 0.4 so dropping that projection is visible.
The block after the sum is the same one the greedy token uses, including that layer's own K and V.
The shared expert is the 2-bit function. The output projection is Q8_K. Eight experts, top 2.
One cell.

Measured on the same run. The first cell pairs token 3 with one residual and samples id 26, the
argmax. The logits matched the same kernel sequence on 0 of 32 values. 16 words past the logits
stayed put. The caller's residual was unchanged. A second cell, token 5 and a different residual,
kept the first cell's K and V and sampled id 15, matching on 0 of 32. A second draw of the first
cell from a fresh state matched on 0 of 32. Token 7 differed on 32 of 32 logits. A different
residual differed on 32 of 32 logits. Adding the embedding projection to one stream was 95.49%
apart on the logits. Skipping the hidden projection was 541.46% apart. A token outside the
vocabulary was refused. A short residual was refused. A missing embedding norm was refused. A
2-bit hidden projection was refused: code_bits has no Q8_0 contract. One synthetic greedy draft.
There is no token rate. This is not a loaded model.

The same binary answers `--serve`. The selftest started it as `strata-metal-token --serve --max-context 8192`.
Extra arguments are ignored. It prints `INFO vocab=32 n_embd=256` and then `READY 4`. The context is
the fixture's 4 cells. The line does not say `stop`. Each `GEN` starts from an empty sequence.
`GEN 1 3` sampled id 15. A second `GEN 1 3` sampled 15 again. `GEN 2 3` sampled 15, then 15.
`GEN 1 3,5` sampled 15, and the first prompt id only filled K and V. `temperature=0` sampled 15.
Each reply is `T <id>` and then `DONE` with the generated count, the prompt length, the prompt
time, the decode time, and the word `length`. An id of 99 was refused, and the next `GEN 1 3`
still sampled 15. Five new tokens on one prompt id was refused: the request does not fit in the
4 cells. A line that is not `GEN` was refused. `temperature=0.7` was refused. `top_p=0.9` was
refused. `VRAM`, `STOP`, `BGEN`, `BSTOP`, and `GENI` were refused, and the process stayed up.
`QUIT` exited 0. The check started that binary. The Python server was not started. One synthetic
greedy token. There is no token rate. This is not a loaded model.

`strata-metal-stack` draws one greedy id through four layers. A layer is full attention when its index
modulo 4 is 3, which is the pack's rule. Layers 0, 1, and 2 are GDN. Layer 3 is the QSA attention
block. Every layer then does the same feed-forward half as the one-block token. No new shader. The
GDN shape is the reduced one: state 128, 2 key heads, 4 value heads, convolution 4. The QSA shape
matches the attention block above, with page size 2 and a rope table of 4 positions. The shared
expert's down scales run from 8 to 12.9.

Measured with `strata-metal-stack --selftest` on the same Mac, the same day. n_embd 256, 4 streams,
rank 32, 8 experts, top 2, vocabulary 32, greedy. The first token is 3. The logits differed from the
same kernel sequence on 0 of 32 values, relative error 0. The last layer's routed ids matched, 0
differ. The full-attention ids matched, 0 differ. The sampled id was 28, the argmax. The GDN state
and its convolution matched, 0 differ. A GDN layer wrote all 65536 state values. The full-attention
layer left its unused GDN state unchanged, 0 changed. Skipping layer 1 was 96.63% apart on the
logits. Running the full-attention layer as GDN was 7.25% apart on the logits. Zeroing the GDN state
was 238.31% apart on the logits. Routing the last layer's slots to expert 0 was 162.34% apart on the
logits. 16 words past the logits stayed put. The second token matched on 0 logits and sampled 18.
Tokens 3, 5, and 7 filled three cells. Clearing the KV pool before token 9 at the next cell was
30.47% apart on the attention output. A negative token was refused. A negative cell was refused. No
layers, a zero interval, and a GDN width that does not match were refused.

`strata-metal-pack` reads the same 19-field index the engine loads. Each plane is its own shared
buffer. No new shader. Measured with `strata-metal-pack --selftest` on the same Mac, the same day.
The high 16 bits of a dirty bf16 were exact: 0x3f800001 stayed 0x3f80. A value whose rounding
conversion is 0x3f82 stayed 0x3f81. fp16 1.0 widened to 1. A subnormal scale widened exactly, and
it did not flush to zero. Verbatim codes and offsets stayed in their own buffers. A raw 16-bit
bf16 was copied. A raw 16-bit f16 was copied. An f32 of 65520 promoted to f16 became 0x7C00, not
the high half 0x477f. A loaded bf16 matrix matched a direct upload, 0 differ. A loaded 4-bit
S-form, with fp16 scales widened and an offset copied, matched a direct upload through the Q8_K
product, 0 differ. An 18-field row was refused. A plane sum of 8/8 B against an index of 10/10 B
was refused. A promoted tensor of 2 elements, 4 B in and 4 B out, was refused. A missing bin was
refused. A short read was refused. A row that holds the tensor only in its GGUF form was refused.
A complete GDN layer and a complete full-attention layer kept 2 sentinel bytes. The GDN layer had
no QSA tensor. The full-attention layer had no GDN tensor. A zero interval was refused. A layer
outside 0..3 was refused. Missing `blk.0.attn_qkv.weight` was refused. Missing
`blk.3.attn_q.weight` was refused. A ne0 of 9 where the kernels require 8 was refused. The
shared-expert gate stored as f32 was 32 B, and the kernels read that tensor as 16 B. The bound
layers use the reduced GDN shape, state 128 and 4 value heads, and one QSA shape the decode kernel
accepts, 24 heads and head dim 256. A 4-bit GDN plane bound, bias -8, with an offset on the gate.
A 4-bit QSA plane bound, and the rope table was left empty. Full attention was not bound as GDN.
A GDN layer was not bound as QSA. The shared expert, the routed expert span, and the embedding
stayed unbound by the layer bind. A 2-bit plane on a Q8_K projection was refused. An S2, S4, and
S8 embedding row matched a direct gather of the same loaded planes, 0 differ. The row length was
32, the vocabulary was 3, and the token was 1. Token 0 differed from token 1. The 4-bit row had
bias -8 and an offset of 0.25. The 2-bit row had bias -1 and no offset. The 8-bit row had bias
-16 and no offset. The fp16 scales widened to 1. The word after the row stayed 7. The row was
read from embd.bin. A missing embedding was refused. A bf16 embedding was refused, and the output
stayed 7. An IQ4NL embedding was refused, and the output stayed 7. A row length other than
n_embd was refused. Token 3 on a table of 3 rows was refused, and token -1 was refused, and the
output stayed 7. A group of 24 on a row of 32 was refused. Planes that covered one row of a
two-row table were refused, and the output stayed 7. Gathering that row does not draw a token.
A shared expert with n_embd 256 and n_ff 640 matched the three products on the loaded planes, 0
differ. The gate was 2-bit, group 64, bias -1, with no offset. The up projection was 4-bit Q8_K,
group 32, bias -8, with an offset. The down projection was 4-bit IQ4NL, group 32, with an offset,
and its activation was Q8_0. Reading that down weight as an affine codebook differed on 256 of 256
outputs. The fp16 scales widened to 1. The down offset was 0.25. The scalar gate's high 16 bits
were 0x3f80. The word after the output stayed 7. A missing up weight was refused. On a row of 32,
a scalar gate stored as f32 was 128 B, and the kernel reads it as 64 B of bf16. A gate whose ne0
was 8, where that expert requires 32, was refused. A 2-bit gate with bias -8 was refused. A Q8_K
down weight whose n_in was 640 was refused. A group of 24 was refused. An f32 gate weight was
refused. The output stayed 7 on each refusal. The 2-bit shared_expert function is not this call.
Running that expert does not draw a token.
A routed expert with n_embd 256 and n_ff 32 matched a copy of the same blob through the expert kernel,
0 differ, on expert 0 and on expert 1. The blob is 13312 bytes. The gate and up rows are IQ4_XS.
The down rows are IQ4_NL. The file named 2 experts. The geometry's expert count stayed 512, and
expert 2 was refused as outside 0..1. The two experts differed on 256 of 256 outputs. The output
was not left at 7. The word after the 256 outputs stayed 7. A row named blk.0.ffn_gate_exps.weight,
the same size as the blob, stayed as it was loaded. Running the expert kernel on that row differed
on 256 of 256 outputs. That row is a 3-D role span and was not passed to the expert kernel. A
32-byte row of that name was refused: it is not the concatenated blob. A blob of 32 bytes, where
the formats make 13312, was refused. A gate type of 29 was refused. An IQ4_NL gate was refused,
because that type is a down quant. A pack with no experts.bin was refused, and a GGUF shard was
not opened. The output stayed 7 on each refusal. The layer bind still leaves the routed expert
span unbound. Running that expert does not draw a token.
A 2-bit query on layer 3 matched the 2-bit product on the loaded planes, 0 differ. n_embd was
256, with 24 heads and head dim 256, so the projection wrote 12288 values. The weight was group
64, bias -1, with no offset, and the codebook was affine. The loaded activation kind was 0. The
fp16 scales widened to 1. Flipping the first code byte differed on 1 of 12288 outputs. The output
was not left at 7. The word after those 12288 outputs stayed 7. A query on layer 0 was refused:
that layer is not full attention. A missing query on the full-attention layer was refused. A zero
interval was refused. A 4-bit query was refused. An offset was refused. Bias -8 was refused.
Group 32 was refused. An IQ4NL codebook was refused. A ne1 of 512 was refused; the query requires
12288. An n_in of 32 was refused. An empty shape was refused. A short activation was refused, and
a short output was refused. The output stayed 7 on each refusal. This call is the 2-bit product
on the query projection. The attention block is not called. The 2-bit shared_expert function is
not this call. The Q8_K bind still refuses a 2-bit plane. The layer bind does not return this
query. Running it does not draw a token.
A pack fed the stack from the same kind of index. Four layers, vocabulary 32, n_embd 256, 4
streams, rank 32, 2 experts, top 2. Layers 0, 1, and 2 are GDN, state 128, 2 key heads, 4 value
heads, convolution 4. Layer 3 is full attention, 24 heads, head dim 256. The query on that layer
is 4-bit Q8_K. A 2-bit query on this path was refused: code_bits 2 has no Q8_K contract. The
2-bit query stays the separate product. The shared expert is the 2-bit shared_expert function,
group 64, bias -1, with no offset, and the codebook is affine. That is the stack's shared expert.
output.weight is 4-bit Q8_K, because the output projection is. The rope table is not in the pack.
The caller passed it. One greedy token from a zero state sampled id 1, the argmax. A second draw
from a fresh zero state matched on 0 of 32 logits. Token 3 differed on 32 of 32 logits. 16 words
past the logits stayed 7. The embedding codes, the layer-0 qkv codes, the layer-3 query codes,
the shared-expert gate codes, and the output codes were the loaded planes. Each routed expert is
its own blob, 22016 bytes. The gate and up rows are IQ4_XS. The down rows are Q2_0. Two experts
sit on each layer. Expert 0 matched the file bytes. A 32-byte row named blk.0.ffn_gate_exps.weight
was loaded from extra.bin and was not an expert. A geometry that does not match the pack was
refused. A missing rope table was refused. A missing embedding was refused. An IQ4NL embedding
was refused. A missing output mix was refused. A missing output.weight was refused. A 2-bit
output.weight was refused. A shared expert that is not 2-bit was refused. A missing
native_experts.txt was refused, and a missing experts.bin was refused. A GGUF shard was not
opened. The layer bind still leaves the embedding, the shared expert, the routed experts, the
2-bit query, and the output projection unbound. One synthetic greedy token. There is no token
rate. This is not a loaded model.

The same run writes one layer at the model width and loads it back. n_embd 2560, and n_layers is 1.
The other defaults stay the model's: 4 streams, rank 320, 512 experts, hidden 640, interval 4, state
128, 16 key heads, 48 value heads, convolution 4, conv width 10240, value width 6144. Layer 0 is GDN.
The check wrote the weights. The gate and up rows are IQ4_XS and the down rows are Q2_0. One blob is
2,201,600 bytes. The file is 1,127,219,200 bytes. The name check saw 23 tensors. Layer 1 is outside
0..0. The layer bound. One GDN step from a zero state repeated, 0 differ, and the guard word stayed 1.
A zero activation differed on 2560 of 2560 outputs, 100.00% apart. Experts 0 and 511 matched the file
bytes, 0 differ, and those two outputs were 44.88% apart. The 3-D role span differed on 2560 of 2560
outputs, 120.22% apart. Expert 512 was refused. No GGUF was opened. This is not a loaded model. There
is no token rate.

When the extract directory is present, the same run loads layer 0 of the Q2_0 file. On this Mac that
directory holds 22 tensors in dense.bin, 66,030,976 bytes, the token embedding, the output weight, and the output mixer in embd.bin, 723,343,360 bytes, and one expert in experts.bin, 1,382,400
bytes. The index names 27 tensors. The loader did not open a GGUF. n_embd is 2560. The other 47 layers are not in the directory.
The name check passed. `blk.0.attn_qkv.weight` is 13,926,400 bytes, stored as IQ4_XS, and the binder
refused it: code_bits 0 has no Q8_K contract. The same bytes through the IQ4_XS product matched the
integer reference, relative error 0, on 10240 rows. Flipping one code byte differed on 1 of 10240
outputs. `blk.0.attn_gate.weight` is Q4_K, 8,847,360 bytes, 6144 rows of 2560. The product matched
the dequantized values, relative error 1.747e-07. The integer dot matched, relative error 0.
Flipping one code byte differed on 1 of 6144 outputs. A zero activation was 0 on all 6144 outputs.
`blk.0.ssm_out.weight` is Q5_K, 10,813,440 bytes, 2560 rows of 6144. The product matched the
dequantized values, relative error 1.786e-07. The integer dot matched, relative error 0. Flipping
one code byte differed on 1 of 2560 outputs. A zero activation was 0 on all 2560 outputs. A
synthetic Q4_K block matched at relative error 1.139e-08, and a synthetic Q5_K block matched at
2.116e-08. Each flip moved 1 of 8 outputs. One GDN step calls those three products. The activation
is q8_1. From a zero state the output matched the host chain, relative error 1.460e-07, on 2560
values. A second draw differed on 0, and the guard word stayed. A zero activation differed on 2560
of 2560 outputs, 100.00% apart. One code bit on query channel 588, the query row with the largest
last convolution tap, differed on 2560 of 2560 outputs. The Q8_K binder still refuses the three
planes, so the S-form layer does not run them. The attention half is the gated residual
around that step. It calls the residual read, that step, and the residual write on
`blk.0.hc_attn_down.weight`, `blk.0.hc_attn_up.weight`, `blk.0.hc_attn_inject.weight`, and
`blk.0.hc_attn_norm.weight`. The norm weight is used as stored. The read matched at relative
error 9.033e-08 on 2560 values. The step, from that mixed vector, matched at relative error
3.205e-05 on 2560 values. The write matched at relative error 6.919e-06 on 10240 values. A
second draw differed on 0, and the guard word stayed. A zero residual differed on 10240 of
10240 values, 100.00% apart. One code bit on query channel 588 differed on 10120 of 10240
values. The feed-forward half reads that kind of residual on its own. It calls the residual read, the router, expert 0, and the residual write on `blk.0.hc_ffn_down.weight`, `blk.0.hc_ffn_up.weight`, `blk.0.hc_ffn_inject.weight`, and `blk.0.hc_ffn_norm.weight`. The norm weight is used as stored. The block written back is expert 0. The router logits stay off that block. The read matched at relative error 8.694e-08. The router matched at relative error 9.173e-07. The expert matched at relative error 0. The write matched at relative error 4.573e-08. A second draw differed on 0, and the guard word stayed. A zero residual differed on 10240 of 10240 values, 100.00% apart. One code bit at byte 921602 differed on 4 of 10240 values. The feed-forward half does not call the shared-expert rows. The expert wrapper still refuses the Q2_0 gate. One layer-0 pass writes the attention half into the residual this feed-forward half reads. The attention write matched at relative error 6.919e-06. On that residual the read matched at 5.457e-08, the router at 1.956e-07, the expert at 0, and the write at 3.809e-08. A second draw differed on 0, and the guard word stayed. A zero residual differed on 10240 of 10240 values, 100.00% apart. One code bit on query channel 588 differed on 10234 of 10240 values. Feeding the attention host's own residual into the expert differs by 7.607e-03 on the expert and 3.125e-04 on the write. On its own activation, the router,
`blk.0.ffn_gate_inp.weight`, is bf16, 2560 by 512. The product matched at relative error 6.683e-08.
A zero activation was 0 on all 512 outputs and differed on 512 of 512 from the patterned activation. Expert 0 is Q2_0 on the gate, the up, and the down,
460,800 bytes each. The three products matched the integer reference, relative error 0. Flipping one
code byte differed on 1 of 640 gate outputs. The expert wrapper refused the gate: the gate type is a
down quant. The shared-expert rows are their own check, and they are not added into the layer-0 residual. The GGUF header names the quants. `blk.0.ffn_gate_shexp.weight` is IQ4_XS, 870,400 bytes, 640 rows of 2560. The product matched the integer reference, relative error 0. Flipping one code byte differed on 1 of 640 outputs. A zero activation was 0 on all 640 outputs. `blk.0.ffn_up_shexp.weight` is Q3_K, 704,000 bytes, 640 rows of 2560. The product matched the dequantized values, relative error 8.589e-08. The integer dot matched, relative error 0. Flipping one code byte differed on 1 of 640 outputs. A zero activation was 0 on all 640 outputs. `blk.0.ffn_down_shexp.weight` is Q5_0, 1,126,400 bytes, 2560 rows of 640. The product matched the dequantized values, relative error 5.944e-08. The integer dot matched, relative error 0. Flipping one code byte differed on 1 of 2560 outputs. A zero activation was 0 on all 2560 outputs. A synthetic Q3_K block matched at relative error 5.650e-08, and a synthetic Q5_0 block matched at relative error 0. Each flip moved 1 of 8 outputs. A synthetic Q4_0 block matched at relative error 0, and the integer dot matched at relative error 0, on 8 rows. Flipping one code byte differed on 1 of 8 outputs. A zero activation was 0 on all 8 outputs. The forward multiplies the IQ4_XS gate and the Q3_K up, takes SiLU of the gate times the up, stores that hidden row as q8_1, multiplies the Q5_0 down, and scales by the sigmoid of the bf16 row `blk.0.ffn_gate_inp_shexp.weight`, 5,120 bytes. The gate matched at relative error 0. The up matched at relative error 0. The down, on the q8 bytes that product wrote, matched at relative error 0. The scalar matched at relative error 8.488e-07. The scaled write matched at relative error 0. A second host chain, from the host gate and the host up, matched at relative error 0. A second draw differed on 0, and the guard word stayed. A zero activation was 0 on 2560 of 2560 outputs, 100.00% apart. One code bit at down byte 6 differed on 1 of 2560 outputs. The S-form runner refused these rows: code_bits 0 is not an S2/S4/S8 form. The expert wrapper refused a Q3_K gate, because that type is not a Coder expert type. It still refuses Q6_K, Q4_0, Q5_0, Q4_K, Q5_K, and a Q2_0 gate. `token_embd.weight` is Q3_K, 273,152,000 bytes, 248320 rows of 2560. The product matched the dequantized values, relative error 7.273e-08. The integer dot matched, relative error 9.004e-11. Flipping one code byte differed on 1 of 248320 outputs. A zero activation was 0 on all 248320 outputs. `output.weight` is Q5_K, 437,043,200 bytes, 248320 rows of 2560. The product matched the dequantized values, relative error 2.042e-07. The integer dot matched, relative error 0. Flipping one code byte differed on 1 of 248320 outputs. A zero activation was 0 on all 248320 outputs. The embedding gather refused these rows: token_embd.weight is not an S2/S4/S8 tensor. These rows are not added into the layer-0 residual. The output mixer is its own check, and it is not added into the layer-0 residual. `output_hc_down.weight` is bf16, 6,553,600 bytes, 320 rows of 10240. `output_hc_norm.weight` is f32, 40,960 bytes, 10240 values, used as stored. `output_hc_up.weight` is bf16, 6,553,600 bytes, 10240 rows of 320. The mixed vector matched at relative error 8.556e-08 on 2560 values. A null inject left the inject buffer untouched. A second draw differed on 0, and the guard word stayed. A zero residual was 0 on 2560 of 2560 values, 100.00% apart. One down byte, 6623, differed on 2516 of 2560 values. One embedding row is then copied into the four residual streams and read by that mixer. The row is token 7971, the embedding row with the largest sum of squares. The mix matched at relative error 1.008e-07 on 2560 values. A null inject left the inject buffer untouched. The Q5_K product of that mixed vector matched the dequantized values, relative error 1.053e-07, and the integer dot matched, relative error 0, on 248320 rows. A second draw differed on 0, and the guard word stayed. A zero embedding was 0 on 248320 of 248320 logits, 100.00% apart. One code byte, 8768132, differed on 248318 of 248320 logits. The embedding gather is not called. The Q8_K head is not called. This check is not added into the layer-0 residual. The same embedding row is copied into the four residual streams and run through the layer-0 pass. The patterned pass still starts from its own residual. The attention write matched at relative error 1.132e-06. On that GPU write the read matched at 4.949e-08, the router at 2.773e-08, the expert at 0, and the write at 1.019e-07. Feeding the attention host's own residual into the expert differs by 3.094e-03 on the expert and 7.748e-04 on the write. A second draw differed on 0, and the guard word stayed. A zero residual differed on 10240 of 10240 values, 100.00% apart. One embedding byte, 8768132, differed on 10240 of 10240 values. The head is not called on that pass. That residual then goes through the output mixer and the Q5_K output weight. The layer write matched at relative error 1.019e-07. The mix of that GPU residual matched at relative error 4.456e-08. A null inject left the inject buffer untouched. The logits matched the dequantized values at relative error 9.448e-08, and the integer dot matched at relative error 0, on 248320 rows. Feeding the attention host's own residual through the expert and the mixer differs by 7.402e-04 on the mix and 2.347e-03 on the logits. A second draw differed on 0, and the guard word stayed. A zero residual differed on 248320 of 248320 logits, 100.00% apart. One embedding byte, 8768132, differed on 248318 of 248320 logits. The shared expert is added on a later check of that residual. Expert 0 is weighted exactly 1 and the shared vector is added plain. The mix matched at 4.410e-08 and the logits at 9.438e-08, integer 0, on 248320 rows. The router selection at k 10 matched at logits 2.773e-08 and weights 2.980e-08, with 0 id differences. Expert 0 is not in that selection. The router weight is not applied. The router logits stay off that block. The file has 512 experts. This check loaded expert 0. The ten selected experts were raw-read from the GGUF and combined with those route weights. The shared vector is added plain. Expert 0's three planes match the extract, 460,800 bytes each. The expert products matched at relative error 0. The shared gate matched at 0, the up at 2.081e-09, the down at 0, and the scale at 2.360e-07. The shared write matched at 7.931e-08. The combine matched at 0. The residual write matched at 0. The mix matched at 4.325e-08. The logits matched at 8.982e-08, integer 0, on 248320 rows. The host-of-host chain differs by 7.227e-04 on the mix and 2.437e-03 on the logits. A second draw differed on 0, and the guard word stayed. A zero residual differed on 248320 of 248320 logits, 100.00% apart. Embedding byte 8768132 differed on 248319 of 248320 logits. Dropping the shared term differed on 10240 of 10240 values. The ids were 232, 479, 66, 342, 308, 16, 27, 164, 347, and 99. Token 7971's state was then carried into token 1, a fixed row. The state matched at 5.638e-08 and the convolution at 0. 786432 state values moved, and 10240 convolution values moved. Token 1's GDN block matched at 1.296e-07. The expert products matched at 0. The shared gate matched at 0, the up at 1.651e-09, the down at 0, and the scale at 7.151e-08. The shared write matched at 9.748e-08. The combine matched at 0. The residual write matched at 4.489e-09. The mix matched at 6.306e-08. The logits matched at 1.263e-07, integer 0, on 248320 rows. The host-of-host chain differs by 1.170e-03 on the mix and 5.182e-03 on the logits. A second draw differed on 0, and the guard word stayed. Clearing that state differed on 248320 of 248320 logits, 141.14% apart. Embedding byte 1132 differed on 248320 of 248320 logits. Dropping the shared term differed on 10240 of 10240 values. Token 1 selected experts 329, 157, 1, 199, 5, 269, 69, 476, 406, and 240. That residual then went through layer 1's GDN step. The query is IQ4_XS, the gate is IQ4_XS, and the output is Q4_K. The block matched at 1.310e-07, the state at 5.489e-08, the convolution at 0, and the write at 5.622e-09. 783872 state values moved, and 10226 convolution values moved. A second draw differed on 0, and the guard word stayed. A zero residual differed on 10240 of 10240, 100.00% apart. Query byte 2 differed on 10021 of 10240. The six blk.1.ple_* tensors were not added. Layer 1's ten routed experts and its Q2_0 shared expert then ran on that attention residual. The read matched at 7.192e-08, the inject at 3.745e-07, and the router at 2.851e-08. The expert products matched at 0. The shared gate, the up, and the down matched at 0. The scale matched at 3.395e-07, the shared write at 3.766e-07, the combine at 0, and the residual write at 5.545e-10. The mix matched at 7.435e-08. The logits matched at 1.176e-07, integer 0, on 248320 rows. The experts were 228 464 470 165 166 249 276 101 19 41. A second draw differed on 0, and the guard word stayed. A zero residual differed on 248320 of 248320, 100.00% apart. Residual byte 2 differed on 248317 of 248320. Dropping the shared term differed on 10240 of 10240. A host chain through the embedding was not run. The six blk.1.ple_* tensors were still not added. Layer 2's GDN step then ran on that feed-forward residual. The query, the gate, and the output are IQ4_XS. The block matched at 0, the state at 6.470e-08, the convolution at 0, and the write at 1.327e-09. 786048 state values moved, and 10237 convolution values moved. A second draw differed on 0, and the guard word stayed. A zero residual differed on 10240 of 10240, 100.00% apart. Query byte 2 differed on 8152 of 10240. Layer 1's ple tensors were still not added. Layer 2's ten routed experts then ran on that attention residual. The shared expert was not added. Its down plane is Q4_0. A synthetic Q4_0 block matched at relative error 0, integer 0, on 8 rows. One code byte differed on 1 of 8, and a zero activation was 0 on all 8. The read matched at 7.115e-08, the inject at 2.729e-07, and the router at 2.857e-08. The expert products matched at 0, the combine at 0, and the residual write at 3.168e-10. The mix matched at 4.534e-08. The logits matched at 1.419e-07, integer 0, on 248320 rows. The experts were 75 502 469 12 13 417 22 331 288 259. A second draw differed on 0, and the guard word stayed. A zero residual differed on 248320 of 248320, 100.00% apart. Residual byte 6 differed on 248320 of 248320. A host chain through the embedding was not run. Layer 2's shared expert then ran on that same attention residual. The gate is IQ4_XS, the up is Q2_0, and the down is Q4_0. The read matched at 7.115e-08, the inject at 2.729e-07, and the router at 2.857e-08. The expert products matched at 0. The shared gate, the up, and the down matched at 0. The scale matched at 6.573e-07, the shared write at 8.199e-07, the combine at 0, and the residual write at 2.997e-10. The mix matched at 4.745e-08. The logits matched at 1.369e-07, integer 0, on 248320 rows. The experts were 75 502 469 12 13 417 22 331 288 259. A second draw differed on 0, and the guard word stayed. A zero residual differed on 248320 of 248320, 100.00% apart. Residual byte 6 differed on 248319 of 248320. Dropping the shared term differed on 10240 of 10240. Layer 1's ple tensors were still not added. Layer 3's attention products then ran on their own activations. The QSA block was not called. The query is Q2_0, 8,847,360 bytes, 12288 rows of 2560, and the integer dot matched at 0. One code byte differed on 1 of 12288. A zero activation was 0 on all 12288. The key is Q3_K, 563,200 bytes, 512 rows of 2560. The product matched the dequantized values at 7.421e-08, and the integer dot at 3.075e-10. One code byte differed on 1 of 512. A zero activation was 0 on all 512. The value is Q5_K, 901,120 bytes, 512 rows of 2560. The product matched the dequantized values at 2.297e-07, and the integer dot at 0. One code byte differed on 1 of 512. A zero activation was 0 on all 512. The output is Q4_K, 8,847,360 bytes, 2560 rows of 6144. The product matched the dequantized values at 2.095e-07, and the integer dot at 0. One code byte differed on 1 of 2560. A zero activation was 0 on all 2560. The QSA block then ran on layer 2's shared residual, at position 0, from an empty cache. The read matched at 6.595e-08 and the inject at 1.008e-06. The block matched at 0 on 2560 outputs. The same activation at position 1 matched at 0. The rotated key matched at 1.353e-09. The selected ids were 0 and 1. Two cells are under the top-k, so the selection is the identity. The rope dimension count is 64 and the frequency base is 10000000. The section array was not applied. A second draw differed on 0, and the guard word stayed. A zero residual differed on 2560 of 2560, 100.00% apart. Value byte 3600 differed on 2560 of 2560. The attention write then put that position-0 block back onto the residual. The read matched at 6.595e-08 and the inject at 1.008e-06. The write matched at 1.412e-09. A second draw differed on 0, and the guard word stayed. A zero residual differed on 10240 of 10240, 100.00% apart. Value byte 3600 differed on 7890 of 10240. The ten routed experts then ran on that attention residual. The shared expert was not added. The read matched at 3.250e-08, the inject at 1.103e-06, and the router at 2.666e-08. The expert products matched at 0. The combine matched at 0 and the residual write at 4.978e-10. The mix matched at 8.420e-08. The logits matched at 1.143e-07, integer 0, on 248320 rows. The experts were 396 64 405 40 454 171 310 78 424 235. A second draw differed on 0, and the guard word stayed. A zero residual differed on 248320 of 248320, 100.00% apart. Residual byte 6 differed on 248318 of 248320. A host chain was not run. The shared expert then ran on the same attention residual. The gate is Q2_0, the up is Q2_0, and the down is Q4_0. The gate, the up, and the down matched at 0. The scale matched at 5.156e-07 and the shared write at 9.926e-07. The combine matched at 0 and the residual write at 3.695e-10. The mix matched at 3.940e-08. The logits matched at 1.205e-07, integer 0, on 248320 rows. The experts were the same ten. A second draw differed on 0, and the guard word stayed. A zero residual differed on 248320 of 248320, 100.00% apart. Residual byte 6 differed on 248316 of 248320. Dropping the shared expert differed on 10240 of 10240. A host chain was not run. Layer 4's GDN step then ran on that shared residual. The query is IQ4_XS, the gate is Q3_K, and the output is IQ4_XS. The state started at zero. The block matched at 0, the state at 5.946e-08, the convolution at 0, and the write at 0. 786432 of 786432 state values moved. 10240 of 30720 convolution values moved. A second draw differed on 0, and the guard word stayed. A zero residual differed on 10240 of 10240, 100.00% apart. Query byte 2 differed on 5135 of 10240. Layer 4's ten routed experts then ran on that attention residual. The shared expert was not added. The read matched at 8.642e-08, the inject at 6.520e-07, and the router at 2.236e-08. The expert products matched at 0. The combine matched at 0 and the residual write at 6.487e-09. The mix matched at 4.997e-06. The logits matched at 1.193e-07, integer 0, on 248320 rows. The experts were 472 283 309 92 81 410 342 496 41 423. A second draw differed on 0, and the guard word stayed. A zero residual differed on 248320 of 248320, 100.00% apart. Residual byte 2 differed on 248303 of 248320. A host chain was not run. Layer 4's shared expert then ran on the same attention residual. The gate, the up, and the down are Q2_0. The gate, the up, and the down matched at 0. The scale matched at 6.248e-07 and the shared write at 1.460e-06. The combine matched at 0 and the residual write at 6.315e-09. The mix matched at 4.645e-08. The logits matched at 1.164e-07, integer 0, on 248320 rows. The experts were the same ten. A second draw differed on 0, and the guard word stayed. A zero residual differed on 248320 of 248320, 100.00% apart. Residual byte 3 differed on 248320 of 248320. Dropping the shared expert differed on 10240 of 10240. A host chain was not run. Layer 1's ple tensors were not applied. Layer 5's GDN step then ran on that shared attention residual. The query is Q3_K, the gate is Q3_K, and the output is IQ4_XS. The state started at zero. The block matched at 0, the state at 7.936e-08, the convolution at 2.403e-09, and the write at 1.295e-08. 786432 of 786432 state values moved. 10240 of 30720 convolution values moved. A second draw differed on 0, and the guard word stayed. A zero residual differed on 10240 of 10240, 100.00% apart. Query byte 2288037 differed on 9631 of 10240. Layer 5's ten routed experts then ran on that attention residual. The shared expert was not added. The read matched at 3.802e-08, the inject at 9.602e-07, and the router at 2.447e-08. The expert products matched at 0. The combine matched at 0 and the write at 0. The mix matched at 5.669e-08. The logits matched at 1.234e-07, integer 0, on 248320 rows. The experts were 167 454 478 98 42 216 236 291 263 459. A second draw differed on 0, and the guard word stayed. A zero residual differed on 248320 of 248320, 100.00% apart. Residual byte 10 differed on 248319 of 248320. A host chain was not run. A synthetic Q6_K block matched at relative error 1.360e-08, and the integer dot matched at relative error 0, on 8 rows. One code byte differed on 1 of 8, and a zero activation was 0 on all 8. iq_expert still refuses a Q6_K up. Layer 5's shared expert then ran on that attention residual. The gate is Q2_0, the up is Q6_K, and the down is Q4_0. The read matched at 3.802e-08, the inject at 9.602e-07, and the router at 2.447e-08. The expert products matched at 0. The gate, the up, and the down matched at 0. The scale matched at 3.060e-07 and the shared write at 4.764e-07. The combine matched at 0 and the write at 0. The mix matched at 5.262e-08. The logits matched at 1.195e-07, integer 0, on 248320 rows. The experts were 167 454 478 98 42 216 236 291 263 459. A second draw differed on 0, and the guard word stayed. A zero residual differed on 248320 of 248320, 100.00% apart. Residual byte 10 differed on 248319 of 248320. Dropping the shared expert differed on 10240 of 10240. A host chain was not run. Layer 6's GDN step then ran on that shared attention residual. The query is Q3_K, the gate is Q3_K, and the output is Q3_K. The state started at zero. The block matched at 3.472e-09, the state at 5.386e-08, the convolution at 2.109e-09, and the write at 0. 786432 of 786432 state values moved. 10240 of 30720 convolution values moved. A second draw differed on 0, and the guard word stayed. A zero residual differed on 10240 of 10240, 100.00% apart. Query byte 2288037 differed on 5707 of 10240. Layer 6's ten routed experts then ran on that attention residual. The shared expert was not added. Its up is Q4_K and was not loaded. The read matched at 6.108e-08, the inject at 1.550e-06, and the router at 2.467e-08. The expert products matched at 0. The combine matched at 0 and the write at 9.250e-09. The mix matched at 6.136e-08. The logits matched at 1.310e-07, integer 0, on 248320 rows. The experts were 223 201 33 127 77 66 379 152 159 283. A second draw differed on 0, and the guard word stayed. A zero residual differed on 248320 of 248320, 100.00% apart. Residual byte 6 differed on 248311 of 248320. A host chain was not run. Layer 6's shared expert then ran on that attention residual. The gate is Q2_0, the up is Q4_K, and the down is Q2_0. The read matched at 6.108e-08, the inject at 1.550e-06, and the router at 2.467e-08. The expert products matched at 0. The gate matched at 0, the up at 1.197e-07, and the down at 0. The scale matched at 4.663e-07 and the shared write at 9.102e-07. The combine matched at 0 and the write at 1.014e-08. The mix matched at 5.823e-08. The logits matched at 1.308e-07, integer 0, on 248320 rows. The experts were 223 201 33 127 77 66 379 152 159 283. A second draw differed on 0, and the guard word stayed. A zero residual differed on 248320 of 248320, 100.00% apart. Residual byte 7 differed on 248319 of 248320. Dropping the shared expert differed on 10240 of 10240. A host chain was not run. Layer 7's four attention products then ran on their own activations. The QSA block was not called. The query is Q2_0, 8,847,360 bytes, 12288 rows of 2560, and matched at 0. One code byte differed on 1 of 12288. A zero activation was 0 on all 12288. The key is Q4_K, 737,280 bytes, 512 rows of 2560, and matched at 2.325e-07, integer 0. One code byte differed on 1 of 512. A zero activation was 0 on all 512. The value is Q3_K, 563,200 bytes, 512 rows of 2560, and matched at 7.426e-08, integer 0. One code byte differed on 1 of 512. A zero activation was 0 on all 512. The output is Q4_K, 8,847,360 bytes, 2560 rows of 6144, and matched at 2.186e-07, integer 0. One code byte differed on 1 of 2560. A zero activation was 0 on all 2560. Layer 7's QSA block then ran on that shared residual, at position 0, from an empty cache. The query is Q2_0, the key is Q4_K, the value is Q3_K, and the output is Q4_K. The read matched at 7.924e-08 and the inject at 9.320e-07. The block matched at 0 on 2560 outputs. Position 1 matched at 0. The rotated key matched at 3.458e-08. The selected ids were 0 and 1. Two cells are under the top-k, so the selection is the identity. The rope dimension count is 64 and the frequency base is 10000000. The section array was not applied. A second draw differed on 0, and the guard word stayed. A zero residual differed on 2560 of 2560, 100.00% apart. Value byte 48 differed on 2559 of 2560. The residual write was not run. Layer 1's PLE was not applied. Layer 7's attention write then put that position-0 block back onto the residual. The query is Q2_0, the key is Q4_K, the value is Q3_K, and the output is Q4_K. The read matched at 7.924e-08 and the inject at 9.320e-07. The write matched at 6.091e-09. A second draw differed on 0, and the guard word stayed. A zero residual differed on 10240 of 10240, 100.00% apart. Value byte 48 differed on 10159 of 10240. The cache started empty. Layer 1's PLE was not applied. Layer 7's ten routed experts then ran on that attention residual. The shared expert was not added. The gate, the up, and the down are Q2_0. The read matched at 3.570e-08, the inject at 1.012e-06, and the router at 2.360e-08. The expert products matched at 0. The combine matched at 0 and the write at 0. The mix matched at 5.239e-08. The logits matched at 1.373e-07, integer 0, on 248320 rows. The experts were 80 358 350 436 163 341 346 165 239 22. A second draw differed on 0, and the guard word stayed. A zero residual differed on 248320 of 248320, 100.00% apart. Residual byte 6 differed on 248319 of 248320 outputs. A host chain was not run. Layer 7's shared expert then ran on that attention residual. The gate is Q2_0, the up is Q2_0, and the down is Q2_0. The read matched at 3.570e-08, the inject at 1.012e-06, and the router at 2.360e-08. The expert products matched at 0. The gate, the up, and the down matched at 0. The scale matched at 2.184e-07 and the shared write at 4.439e-07. The combine matched at 0 and the write at 0. The mix matched at 5.486e-08. The logits matched at 1.361e-07, integer 0, on 248320 rows. The experts were 80 358 350 436 163 341 346 165 239 22. Residual byte 6 differed on 248319 of 248320 rows. Dropping the shared expert differed on 10240 of 10240. A host chain was not run. Layer 8's GDN step then ran on that shared attention residual. The query is Q3_K, the gate is Q3_K, and the output is IQ4_XS. The state started at zero. The block matched at 0, the state at 7.377e-08, the convolution at 2.423e-09, and the write at 7.460e-09. 786432 of 786432 state values moved. 10240 of 30720 convolution values moved. A second draw differed on 0, and the guard word stayed. A zero residual differed on 10240 of 10240, 100.00% apart. Query byte 4031108 differed on 9815 of 10240. Layer 1's ple tensors were not applied. Layer 8's ten routed experts then ran on that attention residual. The shared expert was not added. Its down is Q5_0 and was not loaded. The gate, the up, and the down are Q2_0. The read matched at 3.453e-08, the inject at 1.117e-06, and the router at 2.455e-08. The expert products matched at 0. The combine matched at 0 and the write at 9.591e-09. The mix matched at 4.185e-08. The logits matched at 1.370e-07, integer 0, on 248320 rows. The experts were 456 308 134 387 131 181 114 263 175 183. Residual byte 2 differed on 248317 of 248320 logits. A host chain was not run. Layer 8's shared expert then ran on that attention residual. The gate is Q2_0, the up is Q2_0, and the down is Q5_0. The read matched at 3.453e-08, the inject at 1.117e-06, and the router at 2.455e-08. The expert products matched at 0. The gate, the up, and the down matched at 0. The scale matched at 1.931e-07 and the shared write at 3.816e-07. The combine matched at 0 and the write at 9.746e-09. The mix matched at 3.255e-08. The logits matched at 1.346e-07, integer 0, on 248320 rows. The experts were 456 308 134 387 131 181 114 263 175 183. Residual byte 2 differed on 248320 of 248320. Dropping the shared expert differed on 9145 of 10240. A host chain was not run. Layer 9's GDN step then ran on that shared attention residual. The query is Q2_0, the gate is Q2_0, and the output is Q3_K. The state started at zero. The block matched at 1.895e-09, the state at 4.965e-08, the convolution at 0, and the write at 2.462e-09. 786432 of 786432 state values moved. 10240 of 30720 convolution values moved. A second draw differed on 0, and the guard word stayed. A zero residual differed on 10240 of 10240, 100.00% apart. Query byte 2592188 differed on 9473 of 10240. Layer 1's ple tensors were not applied. Layer 9's ten routed experts then ran on that attention residual. The shared expert was not added. Its gate is IQ4_XS and was not loaded. The gate, the up, and the down are Q2_0. The read matched at 3.408e-08, the inject at 9.187e-07, and the router at 2.836e-08. The expert products matched at 0. The combine matched at 0 and the write at 1.394e-09. The mix matched at 5.940e-08. The logits matched at 1.279e-07, integer 0, on 248320 rows. The experts were 68 427 277 376 142 506 96 322 430 375. Residual byte 2 differed on 248318 of 248320. A host chain was not run. Layer 9's shared expert then ran on that attention residual. The gate is IQ4_XS, the up is Q2_0, and the down is Q2_0. The read matched at 3.408e-08, the inject at 9.187e-07, and the router at 2.836e-08. The expert products matched at 0. The gate, the up, and the down matched at 0. The scale matched at 1.573e-06 and the shared write at 2.680e-06. The combine matched at 0 and the write at 1.670e-09. The mix matched at 3.526e-08. The logits matched at 1.281e-07, integer 0, on 248320 rows. The experts were 68 427 277 376 142 506 96 322 430 375. Residual byte 2 differed on 248315 of 248320. Dropping the shared expert differed on 10238 of 10240. A host chain was not run. Layer 10's GDN step then ran on that shared residual. The query is Q3_K, the gate is Q2_0, and the output is IQ4_XS. The state started at zero. The block matched at 0, the state at 6.208e-08, the convolution at 1.719e-09, and the write at 1.397e-09. 786432 of 786432 state values moved. 10240 of 30720 convolution values moved. A second draw differed on 0, and the guard word stayed. A zero residual differed on 10240 of 10240, 100.00% apart. Query byte 4031108 differed on 9477 of 10240. Layer 1's ple tensors were not applied. Layer 10's ten routed experts then ran on that attention residual. The shared expert was not added. Its down is IQ4_NL and its up is Q3_K. Neither was loaded. The gate, the up, and the down are Q2_0. The read matched at 5.847e-08, the inject at 1.475e-06, and the router at 2.586e-08. The expert products matched at 0. The combine matched at 0 and the write at 5.794e-09. The mix matched at 5.822e-08. The logits matched at 1.212e-07, integer 0, on 248320 rows. The experts were 461 325 107 495 508 164 264 417 97 423. Residual byte 2 differed on 248309 of 248320. A host chain was not run. Layer 10's shared expert then ran on that attention residual. The gate is Q2_0, the up is Q3_K, and the down is IQ4_NL. The read matched at 5.847e-08, the inject at 1.475e-06, and the router at 2.586e-08. The expert products matched at 0. The gate matched at 0, the up at 3.895e-09, and the down at 0. The scale matched at 1.172e-07 and the shared write at 1.295e-07. The combine matched at 0 and the write at 5.670e-09. The mix matched at 5.891e-08. The logits matched at 1.187e-07, integer 0, on 248320 rows. The experts were 461 325 107 495 508 164 264 417 97 423. Changing residual byte 3 differed on 248320 of 248320. Dropping the shared expert differed on 9842 of 10240. A host chain was not run. Layer 11's four attention products then ran on their own activations. The QSA block was not called. The query is Q3_K, 13,516,800 bytes, 12288 rows of 2560, and matched at 7.729e-08, integer 7.313e-11. One code byte differed on 1 of 12288. A zero activation was 0 on all 12288. The key is IQ4_XS, 696,320 bytes, 512 rows of 2560, and matched at 0. One code byte differed on 1 of 512. A zero activation was 0 on all 512. The value is IQ4_XS, 696,320 bytes, 512 rows of 2560, and matched at 0. One code byte differed on 1 of 512. A zero activation was 0 on all 512. The output is Q6_K, 12,902,400 bytes, 2560 rows of 6144, and matched at 1.141e-07, integer 0. One code byte differed on 1 of 2560. A zero activation was 0 on all 2560. Layer 11's QSA block then ran on that shared residual, at position 0, from an empty cache. The query is Q3_K, the key is IQ4_XS, the value is IQ4_XS, and the output is Q6_K. The read matched at 2.072e-05 and the inject at 1.259e-06. The block matched at 0 on 2560 outputs. Position 1 matched at 0. The rotated key matched at 3.513e-08. The selected ids were 0 and 1. Two cells are under the top-k, so the selection is the identity. The rope dimension count is 64 and the frequency base is 10000000. The section array was not applied. A second draw differed on 0, and the guard word stayed. A zero residual differed on 2560 of 2560, 100.00% apart. Value byte 4088 differed on 2559 of 2560. The residual write was not run. Layer 1's PLE was not applied. Layer 11's attention write then put that position-0 block back onto the residual. The query is Q3_K, the key is IQ4_XS, the value is IQ4_XS, and the output is Q6_K. The read matched at 2.072e-05 and the inject at 1.259e-06. The write matched at 5.608e-09. A second draw differed on 0, and the guard word stayed. A zero residual differed on 10240 of 10240, 100.00% apart. Value byte 4088 differed on 10080 of 10240. The cache started empty. Layer 1's PLE was not applied. Layer 11's ten routed experts then ran on that attention residual. The shared expert was not added. Its gate is IQ4_XS and its up is Q3_K. Neither was loaded. The gate, the up, and the down are Q2_0. The read matched at 3.405e-08, the inject at 7.294e-07, and the router at 2.709e-08. The expert products matched at 0. The combine matched at 0 and the write at 0. The mix matched at 4.102e-08. The logits matched at 1.231e-07, integer 0, on 248320 rows. The experts were 350 268 201 273 108 185 48 430 163 141. Changing residual byte 6 differed on 248316 of 248320. A host chain was not run. Layer 11's shared expert then ran on that attention residual. The gate is IQ4_XS, the up is Q3_K, and the down is Q2_0. The read matched at 3.405e-08, the inject at 7.294e-07, and the router at 2.709e-08. The expert products matched at 0. The gate matched at 0, the up at 2.713e-09, and the down at 0. The scale matched at 6.811e-07 and the shared write at 1.594e-06. The combine matched at 0 and the write at 0. The mix matched at 6.031e-08. The logits matched at 1.250e-07, integer 0, on 248320 rows. The experts were 350 268 201 273 108 185 48 430 163 141. Changing residual byte 6 differed on 248319 of 248320. Dropping the shared expert differed on 10237 of 10240. A host chain was not run. Layer 12's GDN step then ran on that shared residual. The query is Q3_K, the gate is Q2_0, and the output is IQ4_XS. The state started at zero. The block matched at 0, the state at 6.290e-08, the convolution at 1.650e-09, and the write at 0. 786432 of 786432 state values moved. 10240 of 30720 convolution values moved. A second draw differed on 0, and the guard word stayed. A zero residual differed on 10240 of 10240, 100.00% apart. Query byte 4031108 differed on 8893 of 10240. Layer 1's PLE was not applied. Layer 12's ten routed experts then ran on that attention residual. The shared expert was not added. Its gate is Q3_K. Its up and its down are Q2_0. None was loaded. The gate, the up, and the down are Q2_0. The read matched at 4.816e-08, the inject at 4.739e-07, and the router at 2.345e-08. The expert products matched at 0. The combine matched at 0 and the write at 4.566e-09. The mix matched at 8.972e-06. The logits matched at 1.243e-07, integer 0, on 248320 rows. The experts were 28 139 421 277 201 433 52 63 288 134. Changing residual byte 2 differed on 248319 of 248320. A host chain was not run. Layer 12's shared expert then ran on that attention residual. The gate is Q3_K, the up is Q2_0, and the down is Q2_0. The read matched at 4.816e-08, the inject at 4.739e-07, and the router at 2.345e-08. The expert products matched at 0. The gate matched at 3.792e-09, the up at 0, and the down at 0. The scale matched at 5.524e-07 and the shared write at 1.389e-06. The combine matched at 0 and the write at 4.768e-09. The mix matched at 4.196e-08. The logits matched at 1.191e-07, integer 0, on 248320 rows. The experts were 28 139 421 277 201 433 52 63 288 134. Changing residual byte 2 differed on 248316 of 248320. Dropping the shared expert differed on 10240 of 10240. A host chain was not run. Layer 13's GDN step then ran on that shared residual. The query is Q3_K, the gate is Q3_K, and the output is Q4_K. The state started at zero. The block matched at 1.295e-07, the state at 8.664e-08, the convolution at 4.285e-09, and the write at 6.173e-10. 786432 of 786432 state values moved. 10240 of 30720 convolution values moved. A second draw differed on 0, and the guard word stayed. A zero residual differed on 10240 of 10240, 100.00% apart. Query byte 4031108 differed on 8062 of 10240. Layer 1's PLE was not applied. No token was drawn. There is no
token rate. This is not a loaded model.

The GDN check is one layer, on synthetic weights. The stack check is four layers, on synthetic
weights. The pack check reads an index into those kernels and draws one synthetic greedy token
from the loaded planes. It also writes one synthetic layer at n_embd 2560 and, when the
extract is present, loads layer 0 of the Q2_0 file. The query plane there is IQ4_XS. Its product matched, relative error 0, on 10240 rows, and
the binder still refuses it. The gate product matched at relative error 1.747e-07, 6144 rows.
The output product matched at relative error 1.786e-07, 2560 rows. One GDN step on those three
matched at relative error 1.460e-07, on 2560 outputs. A second draw differed on 0. A zero
activation differed on 2560 of 2560, 100.00% apart. One code bit on query channel 588 differed
on 2560 of 2560. The attention half around that step matched at read 9.033e-08, step 3.205e-05,
and write 6.919e-06. A second draw differed on 0. A zero residual differed on 10240 of 10240,
100.00% apart. One code bit on query channel 588 differed on 10120 of 10240. The feed-forward half matched at read 8.694e-08, router 9.173e-07, expert 0, and write 4.573e-08. A second draw differed on 0. A zero residual differed on 10240 of 10240, 100.00% apart. One code bit at byte 921602 differed on 4 of 10240. One layer-0 pass matched at attention write 6.919e-06, then read 5.457e-08, router 1.956e-07, expert 0, and write 3.809e-08 on that residual. A second draw differed on 0. A zero residual differed on 10240 of 10240, 100.00% apart. One code bit on query channel 588 differed on 10234 of 10240. The attention host's residual through the same expert differs by 7.607e-03, and the write by 3.125e-04. The shared expert is its own check and is not added into that residual. The GGUF header names the quants: an IQ4_XS gate, a Q3_K up, a Q5_0 down, and a bf16 scalar. The gate rows matched at relative error 0 on 640 rows. The up rows matched at 8.589e-08. The down rows matched at 5.944e-08. The forward matched at gate 0, up 0, down 0, scale 8.488e-07, and write 0. A second host chain matched at 0. A second draw differed on 0. A zero activation differed on 2560 of 2560, 100.00% apart. One code bit at down byte 6 differed on 1 of 2560. The token embedding matched at relative error 7.273e-08 on 248320 rows, and the integer dot at 9.004e-11. The output weight matched at relative error 2.042e-07 on 248320 rows, and the integer dot at 0. A changed code byte differed on 1 of 248320 outputs for each. A zero activation was 0 on all 248320 outputs. The embedding gather refused the row. Those rows are not added into the layer-0 residual. The output mixer matched at relative error 8.556e-08 on 2560 values. A second draw differed on 0. A zero residual was 0 on 2560 of 2560, 100.00% apart. One down byte, 6623, differed on 2516 of 2560. That check is not added into the layer-0 residual. One embedding row, token 7971, matched the head at mix 1.008e-07 and at logits 1.053e-07 on 248320 rows, integer 0. A second draw differed on 0. A zero embedding was 0 on 248320 of 248320, 100.00% apart. One embedding byte, 8768132, differed on 248318 of 248320. That check is not added into the layer-0 residual. The same embedding row through the layer-0 pass matched at attention write 1.132e-06, then read 4.949e-08, router 2.773e-08, expert 0, and write 1.019e-07. The attention host's residual through the same expert differs by 3.094e-03, and the write by 7.748e-04. A second draw differed on 0. A zero residual differed on 10240 of 10240, 100.00% apart. One embedding byte, 8768132, differed on 10240 of 10240. That residual through the output mixer and the Q5_K output weight matched at write 1.019e-07, mix 4.456e-08, and logits 9.448e-08 on 248320 rows, integer 0. The attention host's residual through that mixer differs by 7.402e-04, and the logits by 2.347e-03. A second draw differed on 0. A zero residual differed on 248320 of 248320, 100.00% apart. One embedding byte, 8768132, differed on 248318 of 248320. The shared expert is added on that residual at weight 1, and the shared vector is added plain. The mix matched at 4.410e-08 and the logits at 9.438e-08. The router selection at k 10 matched at logits 2.773e-08 and weights 2.980e-08, with 0 id differences. Expert 0 is not in that selection. The ten selected experts were raw-read from the GGUF and combined with those weights, and the shared vector is added plain. The expert products matched at 0. The mix matched at 4.325e-08 and the logits at 8.982e-08, integer 0, on 248320 rows. The host-of-host chain differs by 7.227e-04 on the mix and 2.437e-03 on the logits. A second draw differed on 0. A zero residual differed on 248320 of 248320, 100.00% apart. Embedding byte 8768132 differed on 248319 of 248320. Dropping the shared term differed on 10240 of 10240. Token 7971's state carried into token 1, a fixed row. The state matched at 5.638e-08 and the convolution at 0. The block matched at 1.296e-07. The mix matched at 6.306e-08 and the logits at 1.263e-07, integer 0, on 248320 rows. Clearing the state differed on 248320 of 248320, 141.14% apart. Embedding byte 1132 differed on 248320 of 248320. Layer 1's GDN step on that residual, with the six ple tensors left out, matched at block 1.310e-07 and write 5.622e-09. Layer 1's routed experts on that residual matched at mix 7.435e-08 and logits 1.176e-07. Layer 2's GDN step on that residual matched at block 0, state 6.470e-08, and write 1.327e-09. Layer 2's routed experts on that residual matched at mix 4.534e-08 and logits 1.419e-07. The shared expert was not added. A synthetic Q4_0 block matched at relative error 0. Layer 2's shared expert on that residual matched at mix 4.745e-08 and logits 1.369e-07. The shared gate, up, and down matched at 0. Layer 3's query matched at integer 0 on 12288 rows, the key at 7.421e-08, the value at 2.297e-07, and the output at 2.095e-07. The QSA block matched at 0. Position 1 matched at 0. The rotated key matched at 1.353e-09. The attention write matched at 1.412e-09. The ten routed experts on that residual matched. The read matched at 3.250e-08, the inject at 1.103e-06, the router at 2.666e-08, and the expert products at 0. The combine matched at 0 and the write at 4.978e-10. The mix matched at 8.420e-08. The logits matched at 1.143e-07, integer 0, on 248320 rows. The experts were 396 64 405 40 454 171 310 78 424 235. The shared expert was not added in that routed check. The shared expert then ran on the same residual. Its gate is Q2_0, its up is Q2_0, and its down is Q4_0. The gate, the up, and the down matched at 0. The scale matched at 5.156e-07 and the shared write at 9.926e-07. The combine matched at 0 and the residual write at 3.695e-10. The mix matched at 3.940e-08. The logits matched at 1.205e-07, integer 0, on 248320 rows. Layer 4's GDN step on that residual matched. The query is IQ4_XS, the gate is Q3_K, and the output is IQ4_XS. The block matched at 0, the state at 5.946e-08, the convolution at 0, and the write at 0. 786432 of 786432 state values moved. 10240 of 30720 convolution values moved. The state started at zero. Query byte 2 differed on 5135 of 10240. Layer 4's ten routed experts on that residual matched. The shared expert was not added. The read matched at 8.642e-08, the inject at 6.520e-07, and the router at 2.236e-08. The expert products matched at 0. The combine matched at 0 and the write at 6.487e-09. The mix matched at 4.997e-06. The logits matched at 1.193e-07, integer 0, on 248320 rows. The experts were 472 283 309 92 81 410 342 496 41 423. Residual byte 2 differed on 248303 of 248320. Layer 4's shared expert then ran on the same residual. Its gate, its up, and its down are Q2_0. The gate, the up, and the down matched at 0. The scale matched at 6.248e-07 and the shared write at 1.460e-06. The combine matched at 0 and the residual write at 6.315e-09. The mix matched at 4.645e-08. The logits matched at 1.164e-07, integer 0, on 248320 rows. The experts were the same ten. Residual byte 3 differed on 248320 of 248320. Dropping the shared expert differed on 10240 of 10240. Layer 5's GDN step on that shared residual matched. The query is Q3_K, the gate is Q3_K, and the output is IQ4_XS. The block matched at 0, the state at 7.936e-08, the convolution at 2.403e-09, and the write at 1.295e-08. 786432 of 786432 state values moved. 10240 of 30720 convolution values moved. The state started at zero. Query byte 2288037 differed on 9631 of 10240. Layer 5's ten routed experts on that residual matched. The shared expert was not added. The read matched at 3.802e-08, the inject at 9.602e-07, and the router at 2.447e-08. The expert products matched at 0. The combine matched at 0 and the write at 0. The mix matched at 5.669e-08. The logits matched at 1.234e-07, integer 0, on 248320 rows. The experts were 167 454 478 98 42 216 236 291 263 459. Residual byte 10 differed on 248319 of 248320. A synthetic Q6_K block matched at relative error 1.360e-08, integer 0, on 8 rows. Layer 5's shared expert on that residual matched. The gate is Q2_0, the up is Q6_K, and the down is Q4_0. The gate, the up, and the down matched at 0. The scale matched at 3.060e-07 and the shared write at 4.764e-07. The combine matched at 0 and the residual write at 0. The mix matched at 5.262e-08. The logits matched at 1.195e-07, integer 0, on 248320 rows. The experts were 167 454 478 98 42 216 236 291 263 459. Residual byte 10 differed on 248319 of 248320. Dropping the shared expert differed on 10240 of 10240. Layer 6's GDN step on that shared residual matched. The query is Q3_K, the gate is Q3_K, and the output is Q3_K. The block matched at 3.472e-09, the state at 5.386e-08, the convolution at 2.109e-09, and the write at 0. 786432 of 786432 state values moved. 10240 of 30720 convolution values moved. The state started at zero. Query byte 2288037 differed on 5707 of 10240. Layer 6's ten routed experts on that residual matched. The shared expert was not added. Its up is Q4_K and was not loaded. The read matched at 6.108e-08, the inject at 1.550e-06, and the router at 2.467e-08. The expert products matched at 0. The combine matched at 0 and the write at 9.250e-09. The mix matched at 6.136e-08. The logits matched at 1.310e-07, integer 0, on 248320 rows. The experts were 223 201 33 127 77 66 379 152 159 283. Residual byte 6 differed on 248311 of 248320. Layer 6's shared expert on that residual matched. The gate is Q2_0, the up is Q4_K, and the down is Q2_0. The gate matched at 0, the up at 1.197e-07, and the down at 0. The scale matched at 4.663e-07 and the shared write at 9.102e-07. The combine matched at 0 and the residual write at 1.014e-08. The mix matched at 5.823e-08. The logits matched at 1.308e-07, integer 0, on 248320 rows. The experts were 223 201 33 127 77 66 379 152 159 283. Residual byte 7 differed on 248319 of 248320. Dropping the shared expert differed on 10240 of 10240. Layer 7's attention products matched on their own activations. The QSA block was not called. The query is Q2_0 and matched at 0 on 12288 rows. One code byte differed on 1 of 12288. A zero activation was 0 on all 12288. The key is Q4_K and matched at 2.325e-07, integer 0, on 512 rows. One code byte differed on 1 of 512. A zero activation was 0 on all 512. The value is Q3_K and matched at 7.426e-08, integer 0, on 512 rows. One code byte differed on 1 of 512. A zero activation was 0 on all 512. The output is Q4_K and matched at 2.186e-07, integer 0, on 2560 rows. One code byte differed on 1 of 2560. A zero activation was 0 on all 2560. Layer 7's QSA block on that shared residual matched. The query is Q2_0, the key is Q4_K, the value is Q3_K, and the output is Q4_K. The read matched at 7.924e-08 and the inject at 9.320e-07. The block matched at 0 and position 1 matched at 0. The rotated key matched at 3.458e-08. The selected ids were 0 and 1. A second draw differed on 0, and the guard word stayed. A zero residual differed on 2560 of 2560, 100.00% apart. Value byte 48 differed on 2559 of 2560. Layer 7's attention write on that residual matched. The read matched at 7.924e-08 and the inject at 9.320e-07. The write matched at 6.091e-09. A second draw differed on 0, and the guard word stayed. A zero residual differed on 10240 of 10240, 100.00% apart. Value byte 48 differed on 10159 of 10240. Layer 7's ten routed experts on that residual matched. The shared expert was not added. The gate, the up, and the down are Q2_0. The read matched at 3.570e-08, the inject at 1.012e-06, and the router at 2.360e-08. The expert products matched at 0. The combine matched at 0 and the write at 0. The mix matched at 5.239e-08. The logits matched at 1.373e-07, integer 0, on 248320 rows. The experts were 80 358 350 436 163 341 346 165 239 22. A second draw differed on 0, and the guard word stayed. A zero residual differed on 248320 of 248320, 100.00% apart. Residual byte 6 differed on 248319 of 248320 outputs. Layer 7's shared expert on that residual matched. The gate is Q2_0, the up is Q2_0, and the down is Q2_0. The read matched at 3.570e-08, the inject at 1.012e-06, and the router at 2.360e-08. The expert products matched at 0. The gate, the up, and the down matched at 0. The scale matched at 2.184e-07 and the shared write at 4.439e-07. The combine matched at 0 and the write at 0. The mix matched at 5.486e-08. The logits matched at 1.361e-07, integer 0, on 248320 rows. The experts were 80 358 350 436 163 341 346 165 239 22. Residual byte 6 differed on 248319 of 248320 rows. Dropping the shared expert differed on 10240 of 10240. Layer 8's GDN step on that shared residual matched. The query is Q3_K, the gate is Q3_K, and the output is IQ4_XS. The block matched at 0, the state at 7.377e-08, the convolution at 2.423e-09, and the write at 7.460e-09. 786432 of 786432 state values moved. 10240 of 30720 convolution values moved. The state started at zero. Query byte 4031108 differed on 9815 of 10240. Layer 8's ten routed experts on that residual matched. The shared expert was not added. Its down is Q5_0 and was not loaded. The gate, the up, and the down are Q2_0. The read matched at 3.453e-08, the inject at 1.117e-06, and the router at 2.455e-08. The expert products matched at 0. The combine matched at 0 and the write at 9.591e-09. The mix matched at 4.185e-08. The logits matched at 1.370e-07, integer 0, on 248320 rows. The experts were 456 308 134 387 131 181 114 263 175 183. Residual byte 2 differed on 248317 of 248320 logits. Layer 8's shared expert on that residual matched. The gate is Q2_0, the up is Q2_0, and the down is Q5_0. The read matched at 3.453e-08, the inject at 1.117e-06, and the router at 2.455e-08. The expert products matched at 0. The gate, the up, and the down matched at 0. The scale matched at 1.931e-07 and the shared write at 3.816e-07. The combine matched at 0 and the write at 9.746e-09. The mix matched at 3.255e-08. The logits matched at 1.346e-07, integer 0, on 248320 rows. The experts were 456 308 134 387 131 181 114 263 175 183. Residual byte 2 differed on 248320 of 248320. Dropping the shared expert differed on 9145 of 10240. Layer 9's GDN step on that shared residual matched. The query is Q2_0, the gate is Q2_0, and the output is Q3_K. The block matched at 1.895e-09, the state at 4.965e-08, the convolution at 0, and the write at 2.462e-09. 786432 of 786432 state values moved. 10240 of 30720 convolution values moved. The state started at zero. Query byte 2592188 differed on 9473 of 10240. Layer 9's ten routed experts on that residual matched. The shared expert was not added. Its gate is IQ4_XS and was not loaded. The gate, the up, and the down are Q2_0. The read matched at 3.408e-08, the inject at 9.187e-07, and the router at 2.836e-08. The expert products matched at 0. The combine matched at 0 and the write at 1.394e-09. The mix matched at 5.940e-08. The logits matched at 1.279e-07, integer 0, on 248320 rows. The experts were 68 427 277 376 142 506 96 322 430 375. Residual byte 2 differed on 248318 of 248320. Layer 9's shared expert on that residual matched. The gate is IQ4_XS, the up is Q2_0, and the down is Q2_0. The read matched at 3.408e-08, the inject at 9.187e-07, and the router at 2.836e-08. The expert products matched at 0. The gate, the up, and the down matched at 0. The scale matched at 1.573e-06 and the shared write at 2.680e-06. The combine matched at 0 and the write at 1.670e-09. The mix matched at 3.526e-08. The logits matched at 1.281e-07, integer 0, on 248320 rows. The experts were 68 427 277 376 142 506 96 322 430 375. Residual byte 2 differed on 248315 of 248320. Dropping the shared expert differed on 10238 of 10240. Layer 10's GDN step on that residual matched. The query is Q3_K, the gate is Q2_0, and the output is IQ4_XS. The block matched at 0, the state at 6.208e-08, the convolution at 1.719e-09, and the write at 1.397e-09. 786432 of 786432 state values moved. 10240 of 30720 convolution values moved. The state started at zero. Query byte 4031108 differed on 9477 of 10240. Layer 10's ten routed experts on that residual matched. The shared expert was not added. Its down is IQ4_NL and its up is Q3_K. Neither was loaded. The gate, the up, and the down are Q2_0. The read matched at 5.847e-08, the inject at 1.475e-06, and the router at 2.586e-08. The expert products matched at 0. The combine matched at 0 and the write at 5.794e-09. The mix matched at 5.822e-08. The logits matched at 1.212e-07, integer 0, on 248320 rows. The experts were 461 325 107 495 508 164 264 417 97 423. Residual byte 2 differed on 248309 of 248320. Layer 10's shared expert on that residual matched. The gate is Q2_0, the up is Q3_K, and the down is IQ4_NL. The read matched at 5.847e-08, the inject at 1.475e-06, and the router at 2.586e-08. The expert products matched at 0. The gate matched at 0, the up at 3.895e-09, and the down at 0. The scale matched at 1.172e-07 and the shared write at 1.295e-07. The combine matched at 0 and the write at 5.670e-09. The mix matched at 5.891e-08. The logits matched at 1.187e-07, integer 0, on 248320 rows. The experts were 461 325 107 495 508 164 264 417 97 423. Changing residual byte 3 differed on 248320 of 248320. Dropping the shared expert differed on 9842 of 10240. Layer 11's attention products matched on their own activations. The QSA block was not called. The query is Q3_K and matched at 7.729e-08, integer 7.313e-11, on 12288 rows. One code byte differed on 1 of 12288. A zero activation was 0 on all 12288. The key is IQ4_XS and matched at 0 on 512 rows. One code byte differed on 1 of 512. A zero activation was 0 on all 512. The value is IQ4_XS and matched at 0 on 512 rows. One code byte differed on 1 of 512. A zero activation was 0 on all 512. The output is Q6_K and matched at 1.141e-07, integer 0, on 2560 rows. One code byte differed on 1 of 2560. A zero activation was 0 on all 2560. Layer 11's QSA block on that shared residual matched. The query is Q3_K, the key is IQ4_XS, the value is IQ4_XS, and the output is Q6_K. The read matched at 2.072e-05 and the inject at 1.259e-06. The block matched at 0 and position 1 matched at 0. The rotated key matched at 3.513e-08. The selected ids were 0 and 1. A second draw differed on 0, and the guard word stayed. A zero residual differed on 2560 of 2560, 100.00% apart. Value byte 4088 differed on 2559 of 2560. Layer 11's attention write on that residual matched. The read matched at 2.072e-05 and the inject at 1.259e-06. The write matched at 5.608e-09. A second draw differed on 0, and the guard word stayed. A zero residual differed on 10240 of 10240, 100.00% apart. Value byte 4088 differed on 10080 of 10240. Layer 11's ten routed experts on that residual matched. The shared expert was not added. Its gate is IQ4_XS and its up is Q3_K. Neither was loaded. The gate, the up, and the down are Q2_0. The read matched at 3.405e-08, the inject at 7.294e-07, and the router at 2.709e-08. The expert products matched at 0. The combine matched at 0 and the write at 0. The mix matched at 4.102e-08. The logits matched at 1.231e-07, integer 0, on 248320 rows. The experts were 350 268 201 273 108 185 48 430 163 141. Changing residual byte 6 differed on 248316 of 248320. Layer 11's shared expert on that residual matched. The gate is IQ4_XS, the up is Q3_K, and the down is Q2_0. The read matched at 3.405e-08, the inject at 7.294e-07, and the router at 2.709e-08. The expert products matched at 0. The gate matched at 0, the up at 2.713e-09, and the down at 0. The scale matched at 6.811e-07 and the shared write at 1.594e-06. The combine matched at 0 and the write at 0. The mix matched at 6.031e-08. The logits matched at 1.250e-07, integer 0, on 248320 rows. The experts were 350 268 201 273 108 185 48 430 163 141. Changing residual byte 6 differed on 248319 of 248320. Dropping the shared expert differed on 10237 of 10240. Layer 12's GDN step on that residual matched. The query is Q3_K, the gate is Q2_0, and the output is IQ4_XS. The block matched at 0, the state at 6.290e-08, the convolution at 1.650e-09, and the write at 0. 786432 of 786432 state values moved. 10240 of 30720 convolution values moved. The state started at zero. Query byte 4031108 differed on 8893 of 10240. Layer 12's ten routed experts on that residual matched. The shared expert was not added. Its gate is Q3_K. Its up and its down are Q2_0. None was loaded. The gate, the up, and the down are Q2_0. The read matched at 4.816e-08, the inject at 4.739e-07, and the router at 2.345e-08. The expert products matched at 0. The combine matched at 0 and the write at 4.566e-09. The mix matched at 8.972e-06. The logits matched at 1.243e-07, integer 0, on 248320 rows. The experts were 28 139 421 277 201 433 52 63 288 134. Changing residual byte 2 differed on 248319 of 248320. Layer 12's shared expert on that residual matched. The gate is Q3_K, the up is Q2_0, and the down is Q2_0. The read matched at 4.816e-08, the inject at 4.739e-07, and the router at 2.345e-08. The expert products matched at 0. The gate matched at 3.792e-09, the up at 0, and the down at 0. The scale matched at 5.524e-07 and the shared write at 1.389e-06. The combine matched at 0 and the write at 4.768e-09. The mix matched at 4.196e-08. The logits matched at 1.191e-07, integer 0, on 248320 rows. The experts were 28 139 421 277 201 433 52 63 288 134. Changing residual byte 2 differed on 248316 of 248320. Dropping the shared expert differed on 10240 of 10240. Layer 13's GDN step on that residual matched. The query is Q3_K, the gate is Q3_K, and the output is Q4_K. The block matched at 1.295e-07, the state at 8.664e-08, the convolution at 4.285e-09, and the write at 6.173e-10. 786432 of 786432 state values moved. 10240 of 30720 convolution values moved. The state started at zero. Query byte 4031108 differed on 8062 of 10240. The ple tensors were still left out. The patterned layer-0 pass still starts from its own residual. The block in that patterned pass is still expert 0. Expert 0's three
Q2_0 products match, and a Q2_0 gate is refused. The token
check also draws one synthetic greedy draft from a caller residual and the next token. The same check draws one greedy token at n_embd 2560, indexer dim 128,
64 rotated values, and top-k 2. Rank stayed 32, with 8 experts, hidden 64, and vocabulary 32. The same
check draws one greedy token at hc_lr 320 on that embedding. The sampled id was 30. Expert count,
hidden size, and top-k stayed at 8, 64, and 2. The same check draws one greedy token with routed
hidden 640 on that rank. The sampled id was 15. Shared hidden stayed 64. The same check draws
one greedy token there with 512 experts. The sampled id was 28. Shared hidden and top-k stayed at
64 and 2. The same check draws one greedy token with shared hidden 640. The sampled id was 30.
Vocabulary and top-k stayed at 32 and 2. The same check draws one greedy token at vocabulary
248320. The sampled id was 52472. Top-k stayed 2. The same check draws sixteen greedy cells
there. The last cell has n_kv 16. Top-k 2 keeps 5 cells and top-k 2048 keeps 16. Those logits
were 165.49% apart. The same binary answers
the serve lines on that fixture. The check started the binary. The Python server was not started.
The model does not run.
There is no token rate.
The QSA decode kernel is its own check. The indexer, the indexer cell scores, the top-k, the paged
KV append, the partial NEOX rotation, and the QSA gate pass as their own checks, and the decode
kernel does not call them. `strata-metal-qsa-layer` calls those, plus the weighted norm and the
decode kernel. `strata-metal-token` calls the embedding row, the gated residual, that attention
block, the router, the routed experts, the shared expert, the combine, the output projection, and
the sampler. It is one block. It also draws one draft step from a caller residual and the next
token. Those two projections use a Q8_0 activation. The block after the sum is the same one.
The caller's residual is not written. The same binary answers `--serve` on that fixture. The check
started it. The Python server was not started. `strata-metal-stack` calls that token path's feed-forward half on
every layer, the GDN layer on the first three, and the QSA block on the last. It is four layers.
`strata-metal-pack` calls the bf16 product, the Q8_K projection, one embedding row, one shared
expert, one routed expert, and one 2-bit query on bytes it loaded. It also feeds one stack token
from those planes. It also writes one synthetic layer at n_embd 2560 and, when the extract is
present, loads layer 0 of the Q2_0 file, including the attention half around the GDN step, the feed-forward half, one layer-0 pass, one embedding through that pass, one layer residual through the head, one shared expert on that residual, one router selection on that residual, one routed ten on that residual, one carried state into a second token, one layer-1 GDN step on that residual, one layer-1 routed expert pass on that residual, one layer-2 GDN step on that residual, one layer-2 routed expert pass on that residual, one layer-2 shared expert on that residual, one layer-3 query, one layer-3 key, one layer-3 value, one layer-3 output, one layer-3 QSA block, one layer-3 attention write, one layer-3 routed expert pass, one layer-3 shared expert, one layer-4 GDN step, one layer-4 routed expert pass, one layer-4 shared expert, one layer-5 GDN step, one layer-5 routed expert pass, one layer-5 shared expert, one layer-6 GDN step, one layer-6 routed expert pass, one layer-6 shared expert, one layer-7 query, one layer-7 key, one layer-7 value, one layer-7 output, one layer-7 QSA block, one layer-7 attention write, one layer-7 routed expert pass, one layer-7 shared expert, one layer-8 GDN step, one layer-8 routed expert pass, one layer-8 shared expert, one layer-9 GDN step, one layer-9 routed expert pass, one layer-9 shared expert, one layer-10 GDN step, one layer-10 routed expert pass, one layer-10 shared expert, one layer-11 query, one layer-11 key, one layer-11 value, one layer-11 output, one layer-11 QSA block, one layer-11 attention write, one layer-11 routed expert pass, one layer-11 shared expert, one layer-12 GDN step, one layer-12 routed expert pass, one layer-12 shared expert, one layer-13 GDN step, one Q6_K product, one Q4_0 product, the shared-expert rows, the Q3_K token embedding, the Q5_K output weight, the output mixer, and one head. The stack query is the Q8_K plane. The stack's shared
expert is the 2-bit function. output.weight on that path is Q8_K. The rope table is not in the
pack. One synthetic greedy token. There is no token rate. The expert-half add is still a kernel only. The 2-bit
shared expert function is still its own check, and the stack path calls it.
Still to write, in this order, each one checked before the next depends on it:

1. Run layer 13's ten routed experts on the attention residual. The gate, the up, and the down are Q2_0. The shared gate is Q4_K, the shared up is Q3_K, and the shared down is Q2_0. None of those shared rows is loaded. Do not load layers 14 through 47. Do not sample an argmax. Do not port the PLE table. Expert streaming is not in this tree. There is no token rate.

About 25,000 lines of CUDA implement the rest of those kernels today. The Intel engine is a separate tree
(`sycl/`) for the same reason this one is: that compiler cannot take the `.cu` files either. The CUDA and HIP
builds are unchanged. No token-per-second figure belongs here until one is measured on a named Mac.

## Xcode

CMake is the file list, the same as the CUDA and HIP builds. `metal/xcode.sh` generates
`build-metal-xcode/strata_metal.xcodeproj` from that list, with a scheme for each tool. Open the project to
capture a GPU frame or to run Instruments' Metal System Trace. The schemes turn Metal frame capture and shader
validation on.

The M5 features used here are the Metal 4.1 language and simdgroups. Both generators call the same Metal
compiler. Leave Derived Data on Xcode's default location, on the internal disk. The sources may sit on a
network volume.

On this Mac the project to open is also on the internal disk. `./metal/xcode.sh --local` copies the Metal
tree to `~/Developer/Strata-metal` and generates `~/Developer/Strata-metal/xcode/strata_metal.xcodeproj`.
The Handoff group in that project is `metal/XCODE_HANDOFF.md`. The parity schemes run `--selftest` in
Release. The canonical sources stay in this repository.

The BF16 product and the S-form warp product are the same reduction as the CUDA warp kernels, which is what
the 1e-5 checks measure. The matrix units on this GPU are a later kernel. A capture is how you see whether
that kernel issues them.

## Build

Xcode's command-line tools are required, including the Metal toolchain. From the Strata folder:

```sh
cmake -S metal -B build-metal
cmake --build build-metal
./build-metal/strata-metal-device --selftest
./build-metal/strata-metal-bf16-gemv --selftest
./build-metal/strata-metal-gdn --selftest
./build-metal/strata-metal-qsa --selftest
./build-metal/strata-metal-iq --selftest
./build-metal/strata-metal-sgemv --selftest
./build-metal/strata-metal-elementwise --selftest
./build-metal/strata-metal-gdn-layer --selftest
./build-metal/strata-metal-gr --selftest
./build-metal/strata-metal-router --selftest
./build-metal/strata-metal-sampler --selftest
./build-metal/strata-metal-moe-combine --selftest
./build-metal/strata-metal-s2-gemv --selftest
./build-metal/strata-metal-shared-expert --selftest
./build-metal/strata-metal-qsa-gate --selftest
./build-metal/strata-metal-indexer --selftest
./build-metal/strata-metal-qsa-index --selftest
./build-metal/strata-metal-topk --selftest
./build-metal/strata-metal-kv --selftest
./build-metal/strata-metal-rope --selftest
./build-metal/strata-metal-qsa-layer --selftest
./build-metal/strata-metal-token --selftest
./build-metal/strata-metal-stack --selftest
./build-metal/strata-metal-pack --selftest
```

The same targets, opened for capture:

```sh
./metal/xcode.sh
open build-metal-xcode/strata_metal.xcodeproj
```

The top-level option `-DSTRATA_ENABLE_METAL=ON` builds the same tree and nothing else. Leave
`STRATA_ENABLE_CUDA`, `STRATA_ENABLE_HIP` and `STRATA_ENABLE_SYCL` off.

`python setup.py` on macOS stops and points at this page. It does not download the Linux engine. The setup
tests call `setup.main()` and are unchanged.

## Installer

When the engine can run a model, setup on a Mac should do what it does on Windows and Linux: look at the memory,
pick the Coder when that is what fits, download that GGUF, build this tree, and start `serve/`. Until that loop
can run a loaded model, none of that runs.
