# CCU 支持对称内存的价值判断

日期：2026-06-08

## 修正后的核心结论

对称内存不是某一种地址计算方式，而是一种**提前注册的跨 rank 共享内存窗口语义**：所有 rank 以 collective 方式注册一段 buffer/window，注册阶段完成远端可访问地址、权限、token/key、生命周期等信息的建立和交换，运行阶段用 `winHandle + peerRank + offset` 访问对端 window。

ZBVA / iova=0 不是和对称内存并列的另一条路径，而是实现对称内存的一种手段。没有提前注册，就没有稳定的 window、token/key、iova=0 映射关系，也就谈不上直接使用 ZBVA。

在 CCU 场景下，价值判断应该改成：

```text
非对称动态路径：
    每次 CCU kernel 启动时交换 peer VA / remote descriptor。

对称内存路径：
    注册阶段建立 stable window descriptor，运行时只传 winHandle/peer/offset/len。

ZBVA-backed 对称内存路径：
    注册阶段把每个 peer window 的 remote address base 建成 0；
    运行时 remote addr = offset；
    但 token/key/channel/context 仍来自注册期建立的 window descriptor。
```

所以，如果当前非对称 CCU kernel 每次启动都要交换内存信息，那么对称内存仍然有性能价值；如果对称内存进一步用 ZBVA 实现，则运行时地址生成可以进一步压缩到 offset，收益更干净。

## 与已有 CCU VA exchange 路径的关系

已有非对称 CCU 路径：

```text
kernel 首轮同步：每个 rank 把本端 VA 或 remote descriptor 写给 peer
运行阶段：每个 rank 根据 peer VA / descriptor 直接 Read/Write 对端 user buffer
```

注册式对称内存路径：

```text
注册阶段：所有 rank collective register window
          建立 peer window 的远端访问描述符
          交换/固化 addr、size、token/key、权限、lifetime

运行阶段：CCU kernel 使用 winHandle + peerRank + offset + len
```

若底层用普通 peer VA / symmetric heap 实现，则运行时地址可能是：

```text
remote_addr(peer, offset) = base(peer) + offset
或
remote_addr(peer, offset) = heapBase + stride * peer + windowOffset + offset
```

若底层用 ZBVA/iova=0 实现，则运行时地址可以是：

```text
remote_addr(peer, offset) = offset
```

但这个 offset 能成立的前提，是注册阶段已经把 peer window 映射成 zero-based remote address space，并且 token/key/context 已经作为 window descriptor 固化下来。

## 内存语义与网络语义的差异

### 内存语义

如果底层支持内存语义，远端访问凭据可以近似抽象成“peer VA 可访问”。非对称路径每次 kernel 交换 peer VA 后就能直接访问；对称内存路径把这类交换提前到注册阶段。

ZBVA-backed 对称内存进一步把 peer VA/base 抹掉，使运行时只需要 offset。

### 网络语义 / URMA 语义

如果 CCU 调度的是 URMA/RDMA-like 网络语义，仅有 VA 或 offset 不够。远端访问至少需要：

```text
remote addr/offset + length + remote token/key + channel/endpoint/entity/QP-like context
```

HCOMM next 代码中 `CcuTransport::CclBufferInfo` 明确包含 `addr`、`size`、`tokenId`、`tokenValue`。`CcuRepRemMem::Translate` 会从 channel 获取 remote buffer 的 addr/size/tokenId/tokenValue，然后把 addr 加载到 GSA，把 token 信息加载到 XN。后续 `ReadNb/WriteNb` 指令使用 remote addr 和 remote token 执行传输。

因此，ZBVA 只解决 remote address base 问题，不解决 remote access authority 问题。对称内存注册必须同时把 token/key/context 纳入 window descriptor，才能真正减少运行时交换。

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

这是一种“VA heap/stride”实现方式；如果底层支持 ZBVA/iova=0，则可以把 peer window 的 remote base 统一成 0，使运行时地址从 `base + offset` 退化成 `offset`。但这仍然属于对称内存注册的实现细节，不是独立于对称内存之外的机制。

### AllToAll fast path

`CollRunAlltoAllFullMeshSymmetricMemory` 设置 `desc_.isZeroCopy = true`，并注册 `TEMPLATE_ALL_2_ALL_FULL_MESH_SYMMETRIC_MEMORY`。模板中通过 `GetRemoteMem(UserMemType::INPUT_MEM)` 获取远端用户输入地址，然后直接 `HcclD2DMemcpyAsync` 到本地用户输出地址。

这说明当前对称内存最直接服务的是 AllToAll/AllToAllVC，而不是普通大包 collective。

### CCU 适配切入点

`pkg_inc/hcomm/ccu/ccu_kernel.h` 中 CCU 已经有 `CreateRemoteAddr`、`GetRemoteAddr`、`ReadNb`、`WriteNb`、`ReadReduceNb`、`WriteReduceNb` 等远端内存操作抽象。对称内存接入 CCU 的本质，是把远端访问描述符从运行时动态交换改成注册期固化：

```text
非 ZBVA 对称内存：
    remote_addr = base(peer) + offset
    token/key   = winDesc.token(peer)

ZBVA-backed 对称内存：
    remote_addr = offset
    token/key   = winDesc.token(peer)
```

### CCU URMA channel 当前限制

`CcuUrmaChannel` 当前注释明确写着“当前仅支持交换 hccl buffer”，并且“当前 CCU 不支持自定义内存交换，仅包含 hccl buffer”。这意味着用户 buffer 直通如果要接 URMA/network semantics，需要扩展 channel/memHandle/window descriptor，而不只是把 VA 传给对端。

## 性能判断

### 有收益的情况

1. 非对称路径中，VA 或 remote descriptor 每次 op/kernel 都发生交换，且首轮同步在小消息中占比高。
2. peer 数较多，例如 8P/16P fullmesh，每个 peer 都需要保存和加载 VA/token descriptor。
3. AllToAllV/AllToAllVC/MoE dispatch-combine 中每轮都要形成大量远端访问描述符。
4. SQE 参数数量、CCU Load 指令数量、XN/GSA 资源成为瓶颈。
5. 需要 graph replay / persistent kernel / persistent communication window，运行时不希望再做动态地址/descriptor 交换。
6. 网络语义下，token/key 可以注册期交换并作为 window descriptor 稳定复用。
7. 如果底层支持 ZBVA/iova=0，运行时 remote addr 可直接用 offset，进一步减少地址装载和计算。

### 收益很小的情况

1. peer VA/token descriptor 在通信域初始化时已经交换，并且跨 op 长期复用。
2. 地址表/token 表常驻 CCU buffer/register，不在关键路径重复加载。
3. 通信主要受链路带宽或 DMA engine 吞吐限制。
4. 大包 AllReduce/AllGather 已经通过 pipeline 打满带宽。
5. window 注册成本无法 amortize，例如一次性临时 buffer。
6. 对称内存只做了地址对称，没有把 token/key/context 纳入注册期稳定 descriptor；这种情况下网络语义收益不完整。

## 设计建议

1. 明确把“对称内存”定义成注册式 window 语义，不要把 ZBVA 和对称内存并列。
2. 对 CCU 网络语义，window descriptor 不能只包含 addr/offset，必须包含 token/key/context 索引。
3. 如果支持 ZBVA/iova=0，CCU kernel 参数优先只传 `winHandle + peerRank + offset + len`，由 win descriptor 提供 peer token/context。
4. 如果不支持 ZBVA，则 win descriptor 需要额外包含 `base/stride/rankOffset`。
5. 保留现有非对称 VA exchange 路径作为 fallback，用于一次性 buffer、注册失败、跨通信域冲突或旧硬件。
6. 多通信域、多 group 下必须明确 window namespace、stride/token lifetime，避免不同 group 复用相同用户 VA 时出现资源冲突。

## 一句话判断

对称内存是提前注册的跨 rank window 语义，ZBVA/iova=0 是实现对称内存的一种方式。若 CCU 非对称路径每次 kernel 启动都要交换内存信息，对称内存有明确性能价值；若再用 ZBVA 实现，运行时 remote addr 可以退化成 offset。但在 URMA/RDMA-like 网络语义下，offset 不能单独完成远端访问，token/key/channel/context 必须随对称 window 在注册期固化，否则收益不完整。
