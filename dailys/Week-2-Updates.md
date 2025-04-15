## Strategic Overview
---
### Demand Hypothesis
- *Current*: Problem of under-utilization of GPUs still remains the same, even more evident now that this video from [OpenAI](https://youtu.be/6nJZopACRuQ?si=Cvva3az1UFPQIZbb) came out.
- *Computational Challenges*: Scaling up GPUs significantly complicated training, making rare issues critical [05:18].
- *Data Efficiency*: The transformer architecture is efficient at absorbing data, but has limits on insight depth [08:42]. The team discussed improving data efficiency and algorithmic innovations [08:36]. Current algorithms are far from human-level efficiency in language learning [28:40].
- *System Challenges*: Training required major system changes like state management and multi-cluster training [09:38]. Co-design between ML and systems is vital for performance [35:07]. The current system is evolving towards an idealized vision [37:02].
- *Future Scaling*: Fault tolerance and system co-design are crucial for the next 10x pre-training jump [11:03]. Future training might involve 10 million+ GPUs [30:51]. Memory bandwidth and balanced system design are potential bottlenecks [34:07].
- *Hardware*: Transport-level networking improvements are important for reliable bandwidth [27:55].
- *Compression & Intelligence*: Pre-training is viewed as compression linked to intelligence; models learning quickly become great compressors [38:53].
---
## Weekly Execution Updates

### Shipped/Worked This Week
- Studied WGMMA layout for FP32 and FP8 precision on Hopper GPUs.
- Analyzed Register Permutation techniques for FP8 operations.
- Investigated Block Quantization and Incoherent Processing in FlashAttention-3.
- Performed SASS analysis of the 2-stage pipeline in FlashAttention-3.
- Implemented Softmax and Decoding functions within FlashMLA.
- Deepened understanding of Unfused vs. Fused Multiply-Accumulate (FMA) operations.
- Reviewed LLVM Intermediate Representation (IR) optimizations.

### Key Learnings
- Fundamental Insight: In order to use the FP8 WGMMA operations on Hopper GPUs, we'd have to use register permutations, which basically means that setting the data in the format of the format of registers in the Hardware. Another important thing to note that is we can make use of these fast register operations by using tiling (making small chunks of big computational blocks). Also, it's not always possible that certain instructions will run exactly as intended on the GPU (because of the virtual ISA, aka PTX). 

### External Validation
- Reached out to several people but only a few of them replied, esp Soumith, validating the idea of slow training in Pytorch. He said to make a detailed processreach out to some compiler ppl at pytorch to get more feedback.

### Blockers & Help Needed
- There's a conflict b/w feedback given by Soumith (from Pytorch)  saying, "Sounds exciting. Can you post more details on what you plan to do to [https://dev-discuss.pytorch.org/](https://dev-discuss.pytorch.org/). I think Jason Ansel or Elias Elison would be able to give you some feedback if you'd like."

  & Vikram (from NVIDIA) saying,

  "Instead write agents that can take PyTorch code as input and writes fused cuDNN + cutlass + CUDA stack code.  Basically remove the overhead of PyTorch and python entirely making it native hardware language. This is source to source compilation method and will be much more efficient and effective. From perf expect 20-50% bump, docker container size dropped from 50GB to 10GB and most importantly highly tuned code for given hardware generation! 

  One drawback - code becomes a bit harder to read. So you need to ensure that there I sufficient comments added!"

### Next Week's Focus
- Feedback beats Planning. Hence will be experimenting more with the current literature/solution available. AMD fast kernel challenge coming up.

