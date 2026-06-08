# CCU 支持对称内存的价值判断

日期：2026-06-08

## 结论

CCU 支持对称内存有意义，但在“CCU kernel 首轮同步已经交换 peer VA，并且后续可以直接访问 peer user buffer”的前提下，收益边界需要重新收敛。

此时对称内存不再主要提供“去 staging copy”的收益，因为当前 CCU 路径本身已经不依赖 CCL staging buffer。它可能带来的性能收益主要剩下：

1. 减少每次 kernel 首轮同步时的 VA exchange；
2. 减少 peer VA 表、地址表、descriptor 的加载和保存；
3. 简化 CCU 指令序列里的远端地址生成；
4. 降低 SQE 参数数量和 CCU XN/GSA 等资源压力；
5. 有利于 graph/replay、persistent window、MoE dispatch/combine 这类重复窗口路径。

如果现有 CCU VA exchange 只在通信域初始化或 template 初始化时做一次，后续所有 op 都复用 peer VA，那么对称内存的运行时性能优势会很小，甚至可能不值得引入复杂的 VA 预留、物理页映射和生命周期管理。

## 与已有 CCU VA exchange 路径的关系

已有 CCU 路径：

```text
kernel 首轮同步：每个 rank 把本端 VA 写给 peer
运行阶段：每个 rank 根据 peer VA 直接 Read/Write 对端 user buffer
```

对称内存路径：

```text
注册阶段：所有 rank 建立一致的 symmetric VA/window 布局
运行阶段：peer addr = base + stride * peer + offset
```

两者都可以做到直接访问 peer user buffer。区别不在于是否 zero-copy，而在于地址获取方式：

- 现有 CCU 路径是“动态交换 VA”；
- 对称内存路径是“注册期建立确定性地址公式”。

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

在已有 VA exchange 机制下，这不是从 staging 到 zero-copy 的改变，而是从“运行期动态交换地址”变成“注册期建立确定性地址公式”。

## 性能判断

### 有收益的情况

1. VA exchange 每次 op 或每次 kernel 都发生，且首轮同步在小消息中占比高。
2. peer 数较多，例如 8P/16P fullmesh，每个 peer 都需要保存和加载 VA。
3. AllToAllV/AllToAllVC/MoE dispatch-combine 中每轮都要根据 peer + offset 形成大量远端地址。
4. SQE 参数数量、CCU Load 指令数量、XN/GSA 资源成为瓶颈。
5. 需要 graph replay / persistent kernel / persistent communication window，运行时不希望再做动态地址交换。

### 收益很小的情况

1. peer VA 在通信域初始化时已经交换，并且跨 op 长期复用。
2. 地址表常驻 CCU buffer/register，不在关键路径重复加载。
3. 通信主要受链路带宽或 DMA engine 吞吐限制。
4. 大包 AllReduce/AllGather 已经通过 pipeline 打满带宽。
5. window 注册成本无法 amortize，例如一次性临时 buffer。

## 设计建议

1. 先测量当前 CCU 路径首轮 VA exchange 的耗时和占比。如果只占极小比例，不要为了对称内存重构数据面。
2. 对 MoE dispatch/combine、AllToAllVC 这类 repeated window 小中消息路径，可以做 symmetric-window fast path。
3. CCU kernel 参数中优先传 `base/stride/windowOffset` 或压缩 win descriptor，避免传完整 peer VA 表。
4. 保留现有 CCU VA exchange 路径作为 fallback，避免对称内存 VA 预留失败、跨通信域冲突或一次性 buffer 场景拖慢。
5. 多通信域、多 group 下必须明确 window namespace、stride 和 lifetime，避免不同 group 复用相同用户 VA 时出现资源冲突。

## 一句话判断

在 CCU 已经通过首轮同步交换 peer VA、并能直接访问 peer user buffer 的前提下，对称内存仍然有工程价值，但性能优势主要来自“减少动态 VA exchange 和地址描述开销”，不是来自“去 staging copy”。如果 VA exchange 已经被初始化期摊掉，CCU 对称内存的运行时收益会非常有限；优先级应低于优化 task 展开、SQE 填充、同步粒度和 CCU/URMA overlap。
