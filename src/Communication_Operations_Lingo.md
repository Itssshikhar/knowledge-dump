## Rank vs Node

- Node is basically a computer itself. Many of these nodes are grouped and connected to a network, then it's called a **cluster**.
- Rank is a **unique ID** for a process that is given when launching any parallel application by specifying how many parallel processes you want to run.
- Each process is given a rank, **starting from 0**, and using these a user can send and receive messages.
- No order is followed b/w a rank and a node. Can be arbitrary.

## Communication cost b/w rank and node

- While sharding does increase the communication overhead, most of the overhead depends upon the actual location of the nodes within the cluster.
	- **Intra-node comms**: Mutiple ranks can execute on a single node without having to deal with comms as in the node is generally more fast and efficient thanks to interconnects *NVLINK*.
	- **Inter-node comms**: When ranks b/w different nodes has to communicate, data has to traverse the network, which obv makes for higher latency and lower bandwidth when compared to *intra-node comms*.
- So, the impact of sharding on the comms is diff across diff settings.
- Where a rank actually is becomes important.
- A good strategy is to **min(Frequency and volume)** b/w inter-node comms by keeping data on same node where it has to be processed.

## Common collective ops

### Broadcast 

- A broadcast operation sends the same piece of data from one designated *"root" rank* to *all other ranks* in the group.
![](Pasted%20image%2020250729155228.png)
- **Initial State:**
    - *Rank 0*: has no data
    - *Rank 1:* has no data
    - *Rank 2:* has data `42`
    - *Rank 3:* has no data
- **Final State:**
    - *Rank 0:* has data `42`
    - *Rank 1:* has data `42`
    - *Rank 2:* has data `42`
    - *Rank 3:* has data `42`
- Rank 2 sends the value `42` to all other ranks (Rank 0, Rank 1, and Rank 3). In practice, this might happen through **a direct send** to each or through a more optimized **network pattern like a tree**, where Rank 2 sends to Rank 0 and Rank 1, and they in turn forward it to others.

### Scatter

- A scatter operation takes an array of data on a single root rank and distributes _different chunks_ of that array to the other ranks in the group.
![](Pasted%20image%2020250729155720.png)
- **Initial State:**
    - *Rank 0:* has no data
    - *Rank 1:* has no data
    - *Rank 2:* has data `[10, 20, 30, 40]`
    - *Rank 3:* has no data
- **Final State:**
    - *Rank 0:* has data `10`
    - *Rank 1:* has data `20`
    - *Rank 2:* has data `30`
    - *Rank 3:* has data `40`
- Rank 2 sends a unique chunk to each corresponding rank. It sends `10` to Rank 0, `20` to Rank 1, and `40` to Rank 3. It keeps the third chunk, `30`, for itself. Using Rank 0 as an example, `Y = 0` because Rank 0. `count = 1` because each rank gets 1 integer. `i = 0` because we want the first integer. So the value that rank 0 gets out of `data` is `data[0*1+0]` = `data[0]` = 10.

### Gather

- A gather operation is the inverse of a scatter. It collects individual data values from all ranks in the group and assembles them into an array on a single designated root rank.
![](Pasted%20image%2020250729160341.png)
- **Initial State:** 
    - *Rank 0:* has result `A`
    - *Rank 1:* has result `B`
    - *Rank 2:* has result `C`
    - *Rank 3:* has result `D`
- **Final State:**
    - *Rank 0:* still has its data `A`
    - *Rank 1:* still has its data `B`
    - *Rank 2:* has data `[A, B, C, D]`
    - *Rank 3:* still has its data `D`
- Each rank sends its individual data value to Rank 2. Rank 0 sends `A`, Rank 1 sends `B`, and Rank 3 sends `D`. Rank 2 receives the data from all ranks and assembles it into an array in a specific order (usually based on rank ID). 

### Reduce

- A reduce operation is similar to a gather, but as it collects data from all ranks, it combines them into a _single final value_ using a specified operation (like sum, max, min, or logical AND). This final result is stored on only one root rank.
![](Pasted%20image%2020250729161529.png)
- **Initial State**:
	- *rank 0* `in0`: `[10, 20, 5]`
	- *rank 1* `in1`: `[2, 4, 6]`
	- *rank 2* `in2`: `[1, 1, 1]`
	- *rank 3* `in3`: `[5, 10, 15]`
- **Final State:**
	- *rank 0, 1, and 3:* Their input buffers remain unchanged. They have no `out` buffer.
	- *rank 2 (root):* Now has the `out` buffer: `[18, 35, 27]`.
- After the `Reduce` operation, the `out` buffer on `rank 2` will be calculated as follows:
	- *out[0]* = `sum(inX[0])` = `in0[0] + in1[0] + in2[0] + in3[0]` = `10 + 2 + 1 + 5` = **`18`**
	- *out[1]* = `sum(inX[1])` = `in0[1] + in1[1] + in2[1] + in3[1]` = `20 + 4 + 1 + 10` = **`35`**
	- *out[2]* = `sum(inX[2])` = `in0[2] + in1[2] + in2[2] + in3[2]` = `5 + 6 + 1 + 15` = **`27`**

### All-Reduce

- An all-reduce operation is a combination of a reduce and a broadcast. It performs a reduction (combining values from all ranks into a single result), and then it broadcasts that final result back to _all_ the ranks.
![](Pasted%20image%2020250729231749.png)
- **Initial State:**
	- *rank 0* `in0`: `[10, 20, 5]`
	- *rank 1* `in1`: `[2, 4, 6]`
	- *rank 2* `in2`: `[1, 1, 1]`
	- *rank 3* `in3`: `[5, 10, 15]`
- **Intermediate State:**
	- *sum(inX[0])* = `10 + 2 + 1 + 5` = **`18`**
	- *sum(inX[1])* = `20 + 4 + 1 + 10` = **`35`**
	- *sum(inX[2])* = `5 + 6 + 1 + 15` = **`27`**
	- The final, reduced buffer is `[18, 35, 27]`.
	- Now, the `Broadcast` phase ensures this result is distributed.
- **Final State**:
	- After the `All-Reduce` operation, **every single rank** will have an identical `out` buffer:
	- *rank 0* `out`: `[18, 35, 27]`
	- *rank 1* `out`: `[18, 35, 27]`
	- *rank 2* `out`: `[18, 35, 27]`
	- *rank 3* `out`: `[18, 35, 27]`

### Reduce-Scatter

- A reduce-scatter operation first combines data from all ranks using a specified operation (like a reduce) and then scatters the resulting combined data chunks back to the ranks. So, *each rank* ends up with *a piece of the final result*.
![](Pasted%20image%2020250729233415.png)
- **Initial State:**
	- *rank 0* `in0`: `[10, 20, 5, 1]`
	- *rank 1* `in1`: `[2, 4, 6, 2]`
	- *rank 2* `in2`: `[1, 1, 1, 3]`
	- *rank 3* `in3`: `[5, 10, 15, 4]`
- **Intermediate State:**
	- 1. For **`rank 0` (`Y=0`):** It needs to calculate its output buffer, `out0`. Since `count=1`, `i` can only be 0.
	    - *out0[0]* = `sum(inX[0*1+0])` = `sum(inX[0])`
	    - *out0[0]* = `in0[0] + in1[0] + in2[0] + in3[0]` = `10 + 2 + 1 + 5` = **`18`**
- **Final State:**
	- *rank 0:* `out0` = `[18]`
	- *rank 1:* `out1` = `[35]`
	- *rank 2:* `out2` = `[27]`
	- *rank 3:* `out3` = `[10]`

### All-Gather

- An all-gather operation is where every rank collects all the individual data values from all other ranks. Unlike a standard gather where only one root rank receives the data, in an all-gather, *every rank* ends up with a complete set of the *data from all ranks*.
![](Pasted%20image%2020250729233838.png)
- **Initial State:**
	- *rank 0* `in0`: `[10, 11]`
	- *rank 1* `in1`: `[20, 21]`
	- *rank 2* `in2`: `[30, 31]`
	- *rank 3* `in3`: `[40, 41]`
- **Intermediate State:**
	- Placing data from **rank 0 (`Y=0`):**
	    - *out[02+0]* = `in0[0]` => `out[0]` = `10`
	    - *out[02+1]* = `in0[1]` => `out[1]` = `11`
- **Final State:**
	- After the operation, **every single rank** will have the identical, fully assembled `out` buffer:
		- *rank 0* **`out`**: `[10, 11, 20, 21, 30, 31, 40, 41]`
		- *rank 1* **`out`**: `[10, 11, 20, 21, 30, 31, 40, 41]`
		- *rank 2* **`out`**: `[10, 11, 20, 21, 30, 31, 40, 41]`
		- *rank 3* **`out`**: `[10, 11, 20, 21, 30, 31, 40, 41]`