## Strategic Overview

### Demand Hypothesis
- Current: The biggest problem right now is the under-utilization of the available GPU throughput (they are sitting idle, even during the training)
- Future: This problem will likely evolve, if not solved, such that we'd need 100k GPU just to train a foundational model, that too being only 20-30% utilized at best.

### Status Quo Analysis
- Current Solutions: Right now, frameworks like TensorRT, xformers and Unsloth exist but they are only making sure that chunking is efficient so that required GPU would be lower in size, not maximizing the throughput and FLOPS.
- Expected Evolution: In the near future, I expect some sort of scheduler/compiler that would arrange the blocks/processes that would keep the GPU in constant 100% usage at all time

### 10x Improvement Vision
- Key Differentiator: The improvement that we are targeting is going to be both consumer/enterprise GPUs, ranging from hobbyist to professionals.
- User Impact: Combined with something like Unsloth (which is crazy memory efficient), we can probably use models bigger than 1B on just a free Kaggle notebook.

### Innovation Proposal
- Core Innovation: The key innovation is going to be a technique that will make sure to keep the GPU occupied at all times.
- Implementation Path: My current hypothesis is using MLIR and a particular type of scheduler to do this.

### Value Capture Strategy
- Revenue Streams: Possible revenue streams are going to be solutions/integrations to enterprise hyper-scalers in a paid version.
- Product Portfolio: Consultancy and Integrations in their current training pipelines

### Compounding Advantages
- Moat Strategy: Proprietary hardware-aware scheduling algorithms combined with deep technical expertise in GPU architecture optimizations. The approach builds cumulative advantage through specialized kernel implementations, GPU-specific optimizations, and a growing database of hardware performance profiles that enable us to create increasingly efficient schedulers across diverse GPU architectures.
- Network Effects: Once torch-fuser is used with the already used ecosystems like Pytorch and Jax, it is going to be very tough to replace.

## Weekly Execution Update

### Shipped This Week
- Deep technical analysis of Flash-Attention 3's architecture and optimizations:
  - Producer-Consumer asynchrony through warp-specialization and pingpong scheduling
  - 2-stage and 3-stage GEMM-softmax pipelining techniques for better GPU utilization
  - Detailed understanding of GPU memory hierarchy and thread organization
  - Analysis of low-precision number formats and their layout constraints
  - Mathematical derivation of attention backward pass and softmax Jacobian
  - Insights into modern GPU architecture's performance disparities

### Key Learnings
- Fundamental Insight: GPU architecture optimization requires understanding the complex architecture between memory hierarchies, asynchronous execution, and specialized hardware units. Effective pipelining (like the 2-stage GEMM-softmax approach in FlashAttention-3) can overcome sequential dependencies by leveraging hardware-level asynchrony.
- Market Insight: Modern GPU architecture is evolving toward dedicated accelerators for specific operations, with significant performance disparities between different operations (e.g., 989 TFLOPS for FP16 matrix multiply vs. 3.9 TFLOPS for special functions on H100). System design must account for these imbalances by optimizing workload scheduling.

### External Validation
- Conversations: With Dwarak and Sachin
- Feedback: From Sachin: He is only able to make <30% use of his H100s during RL training process, which he is also regretting. From Dwarak: Being a performance engineer, he said right now there is no common way that make 100% sure of GPU and someone has to sit and improve the perf, looking at the profiling results.
- Related Discoveries: 
  - [FlashAttention-3 (2024)](https://arxiv.org/abs/2407.08608): Achieves 1.5-2.0x speedup over previous version using asynchronous execution, warp-specialization, and low-precision techniques. Reaches 75% GPU utilization on H100 (up from 35%), demonstrating the potential of scheduler optimizations and hardware-level asynchrony.
  - [Towards high-performance AI compiler with upstream MLIR (2024)](https://arxiv.org/abs/2404.15204): Proposes a compilation flow using MLIR that achieves >90% of the performance of manually-optimized code from a high-level abstraction, validating our MLIR-based approach.
  - [Droplet Search and TVM's Ansor (2024)](https://arxiv.org/abs/2406.20037): Recently approved scheduler optimization that combines exploration and exploitation phases to find optimal kernel implementations using coordinate descent algorithms.
  - [ML-Compiler-Bridge (2023)](https://arxiv.org/abs/2311.10800): Approach for enhancing compiler optimizations with ML models, particularly relevant for our scheduler that will need to balance multiple optimization objectives.
  - Open-source projects: TensorRT, xformers, and Unsloth all address memory efficiency but not maximizing computational throughput, confirming our hypothesis about the gap in the market.

### Blockers & Help Needed
- Current challenges are to be able to design the framework and test with popular models on hyper-accelerators like H100/B200.
- Need more insights from people whose expertise are MLIR, schedulers and performance/optimizations.

### Next Week's Focus
- Work on the blockers and more Flash-Attention kind of algorithms