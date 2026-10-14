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
the combine, the output projection, and the sampler. It is one block. Four synthetic
layers follow the pack schedule, three GDN and one QSA, and that is not the 48-layer model.
The pack index is read into those kernels, including one embedding row, one shared expert, one routed expert blob, and one 2-bit query. It does not draw a token.

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
refused. A value width of 128, which is not a multiple of 256, was refused.

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
of 256 was refused.

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

The GDN check is one layer, on synthetic weights. The stack check is four layers, on synthetic
weights. The pack check reads an index into those kernels. The model does not run. There is no
token rate.
The QSA decode kernel is its own check. The indexer, the indexer cell scores, the top-k, the paged
KV append, the partial NEOX rotation, and the QSA gate pass as their own checks, and the decode
kernel does not call them. `strata-metal-qsa-layer` calls those, plus the weighted norm and the
decode kernel. `strata-metal-token` calls the embedding row, the gated residual, that attention
block, the router, the routed experts, the shared expert, the combine, the output projection, and
the sampler. It is one block. `strata-metal-stack` calls that token path's feed-forward half on
every layer, the GDN layer on the first three, and the QSA block on the last. It is four layers.
`strata-metal-pack` calls the bf16 product, the Q8_K projection, one embedding row, one shared
expert, one routed expert, and one 2-bit query on bytes it loaded. It does not draw a token. The
expert-half add is still a kernel only. The 2-bit shared expert function is still its own check.
Still to write, in this order, each one checked before the next depends on it:

1. A pack can feed the stack. Then the MTP draft layer, then the Python server in `serve/`, which
   can stay as it is once it has an engine binary to start.

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
