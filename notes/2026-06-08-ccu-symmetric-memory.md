# CCU 支持对称内存的价值判断

日期：2026-06-08

## 结论

CCU 支持对称内存有明确意义，但应作为面向特定通信模式的 fast path，而不是替代所有集合通信路径。

最值得优先覆盖的场景是：

1. MoE Dispatch/Combine 对应的 AllToAll / AllToAllV / AllToAllVC；
2. 单机或超节点内 full-mesh、peer-to-peer 可达的中小消息通信；
3. 重复使用同一批通信 buffer/window 的低时延路径；
4. 需要把通信控制下沉到 device/CCU 侧、减少 host/AICPU 展开开销的路径。

对大包带宽受限的 AllReduce/AllGather，收益不应默认假设存在。已有高度优化的 ring/tree/SDMA/RDMA pipeline 可能已经接近链路上限，对称内存主要减少的是控制开销、搬运层级和地址计算开销，而不是物理链路带宽。

## 公开资料依据

- NVIDIA NCCL Device API 明确依赖 symmetric memory / window registration；NCCL 2.27 的对称内存低时延 kernel 在小消息 AllReduce 上给出过最高 7.6x 的延迟下降数据。
- Ascend SHMEM MoE Dispatch/Combine 适配 issue 中，Shmem 对称通信域地址管理替换 HCCL/CAM 后，内部验证显示算子下发时间约下降 50%，单轮端到端时延下降 10%～15%。这与 HCCL/CCU 要解决的低时延控制面开销高度相关。
- 也有反例：SGLang 对 PyTorch/NVSHMEM symmetric memory all-to-all 的 issue 中，修正 benchmark 后 NVSHMEM 在测试环境下没有超过 NCCL，大包更慢。因此必须基于场景选择，而不是一刀切。

## HCOMM 代码依据

### API 层

HCOMM 已暴露两类相关 API：

1. 零拷贝 memory range：`HcclCommSetMemoryRange`、`HcclCommUnsetMemoryRange`、`HcclCommActivateCommMemory`、`HcclCommDeactivateCommMemory`。
2. 对称内存 window：`HcclCommSymWinRegister`、`HcclCommSymWinDeregister`、`HcclCommSymWinGet`。

`HcclCommSymWinRegister` 当前只接受 `HCCL_WIN_COLL_SYMMETRIC`，`HCCL_WIN_DEFAULT` 分支明确返回不支持。这说明当前实现意图不是普通用户内存注册，而是专门的 collective symmetric window。

### 对称内存管理器

`symmetric_memory.{h,cc}` 的核心逻辑是：

```text
heapBase + stride * rank + alignedHeapOffset
```

每个 rank 预留同样大小的 VA heap；本地 PA 导出为 shareable handle，所有 rank exchange handle，然后把每个 rank 的物理内存映射到本进程 VA heap 的对应 slot。这样任意 rank 都可以用同一套公式计算 peer buffer 地址。

### AllToAll fast path

`CollRunAlltoAllFullMeshSymmetricMemory` 设置 `desc_.isZeroCopy = true`，并注册 `TEMPLATE_ALL_2_ALL_FULL_MESH_SYMMETRIC_MEMORY`。模板中通过 `GetRemoteMem(UserMemType::INPUT_MEM)` 获取远端用户输入地址，然后直接 `HcclD2DMemcpyAsync` 到本地用户输出地址。

这说明当前对称内存最直接服务的是 AllToAll/AllToAllVC，而不是普通大包 collective。

### CCU 适配切入点

`pkg_inc/hcomm/ccu/ccu_kernel.h` 中 CCU 已经有 `CreateRemoteAddr`、`GetRemoteAddr`、`ReadNb`、`WriteNb`、`ReadReduceNb`、`WriteReduceNb` 等远端内存操作抽象。对称内存接入 CCU 的本质，是把远端地址生成和 window 映射标准化：

```text
remote_addr(peer, user_offset) = heapBase + stride * peer + alignedHeapOffset + user_offset
```

这样 CCU 可以直接面向 user buffer/window 发起 read/write/reduce，而不必依赖 CCL staging buffer 或 host 侧复杂地址表。

## 设计建议

1. 先做 AllToAll / AllToAllVC / MoE Dispatch-Combine 的 CCU symmetric-window fast path。
2. window 注册必须 amortize：初始化或 graph capture 前注册，运行时只传 `winHandle + offset + split/counter`。
3. 大包路径保留现有 HCCL 算法选择，按 message size/topology/是否复用 window 做选择。
4. CCU kernel 参数中不要传完整 VA 表，优先传 `heapBase/stride/window offset/peer rank` 或经过压缩的 window descriptor。
5. 对多通信域、多 group 要明确 VA slot/stride 的隔离，否则相同用户 VA、不同 group/window 可能产生地址管理冲突。

## 一句话判断

CCU 支持对称内存值得做，但收益主要来自“减少控制面与 staging 开销 + 使 CCU 能直接访问 peer user window + 便于 MoE all-to-all 类通信下沉/融合”，不是来自物理带宽变大。优先目标应是低时延、中小消息、重复 window、MoE dispatch/combine，而不是所有 collective 全量替换。
