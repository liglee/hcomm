# AICPU 直接填写 RDMA SQE 完成集合通信的可行性分析

## 背景

当前 HCCL/HCOMM 的通信算子下发链路可以抽象为：Host 下发通信算子 kernel，AICPU 在 device 侧执行通信算法/指令展开，生成本地 SDMA、notify、RDMA doorbell 等 STARS SQE，提交到 STARS/RTSQ，由 STARS 负责按 stream 语义调度执行，并进一步触发片内搬运或 RDMA 网卡动作。

代码证据：

- `ins_to_sqe_rule_v82.cc` 将 `InsRead/InsWrite/InsWriteReduce/BatchWrite` 等通信指令解释成 `transport.Read/Write/WriteReduce/BatchTransfer` 调用。
- `aicpu_hccl_sqcqv1.cc::AddOneRdmaDbSendSqeV1` 构造的是 `RT_STARS_SQE_TYPE_WRITE_VALUE`，sub type 为 `RT_STARS_WRITE_VALUE_SUB_TYPE_RDMA_DB_SEND`，本质是 STARS SQE 触发 RDMA doorbell，而不是裸写 NIC WQE。
- `sqe_mgr.cc` 中 `SqeMgr::Commit` 负责将本地 SQE buffer 拷贝到 RTSQ base，并通过配置 SQ tail 提交给 STARS。

## 问题定义

“直接使用 AICPU 给 RDMA 填写 SQE”有两种含义：

1. **现有语义内的直接化**：AICPU 继续生成 STARS SQE，但减少中间 task/transport 封装，直接生成 RDMA doorbell 相关 SQE。这与当前 `RDMA_DB_SEND_SQE` 思路一致，只是进一步压缩软件层。
2. **绕过 STARS 的裸 RDMA WQE 路径**：AICPU 直接写 RDMA NIC 的 send queue/WQE，并直接写 doorbell，STARS 不再参与通信任务调度。这是更激进的方案。

## 性能优先视角下的修正判断

如果目标是极致性能，不能简单地认为“保留 STARS 更稳妥”。NVIDIA 的方向已经证明：对 MoE、小包 alltoall、kernel 内触发 one-sided 通信这类场景，device/kernel initiated networking 是重要方向。NCCL 2.28 引入 Device API 与 GIN；GIN 的设计目标就是让 CUDA kernel 内调用远端内存操作，并通过 GPUDirect Async Kernel-Initiated backend 利用 DOCA GPUNetIO 做 GPU-to-NIC 直连通信，同时保留 Proxy backend 作为兼容路径。

因此真正的问题不是“直驱有没有性能优势”，而是：

- 发起方是 AI Core / AIV kernel，还是 AICPU？
- 直接填的是 NIC WQE，还是填 STARS RDMA doorbell SQE？
- completion、ordering、资源回收、错误处理是否仍有硬件/firmware/driver 支撑？
- 是否只针对规则小包通信 fast path，而不是覆盖所有 collective 主路径？

## 可行性判断

从硬件/软件原理看，AICPU 如果具备以下能力，理论上可以直接填 RDMA WQE：

- 能访问 RDMA QP/SQ/CQ/doorbell 对应的 MMIO 或 device memory；
- 能拿到 QP number、lkey/rkey、remote addr、packet length、opcode、inline/SGE 等完整 WQE 字段；
- 能保证 WQE 写入、cache flush、memory barrier、doorbell write 的顺序；
- 能处理 CQE、错误码、QP 异常、重试、超时、资源回收；
- 能与 AI Core/SDMA/notify 的 stream 依赖关系正确同步。

但在 HCCL 当前架构下，这会绕过大量现有抽象：STARS 的 stream 顺序、notify/event、profiling、timeout、resource manager、transport/link 抽象、异常恢复与跨芯片差异封装。

## 潜在优势

1. **降低小包控制面开销**

   对大量小粒度 RDMA write/read/notify，若每个通信动作都要生成 STARS task，再由 STARS 触发 RDMA doorbell，直接填 WQE 理论上可以减少一层调度与 SQE 转换开销。

2. **降低尾延迟抖动**

   对 MoE dispatch/combine、MC2 小包同步等场景，控制路径越短，AICPU 侧可以更直接地做 batch WQE 生成和 doorbell 合并，减少 task 排队抖动。

3. **更容易实现通信协议专用优化**

   HCCL 可以针对 alltoall、batch write、one-sided write/reduce 等模式生成更贴近 NIC 的 WQE 模板，例如多目的 rank 的 WQE 模板化、只改 length/addr/imm data、doorbell batch。

4. **减少 STARS SQ 深度压力**

   如果大量网络操作不进入 STARS RTSQ，STARS SQ 深度、tail/head 查询、SQ 满等待等压力会下降。

5. **支持真正细粒度计算通信融合**

   如果未来不是 AICPU，而是 AI Core kernel 或 CCU 在数据准备好后立即触发 RDMA，就可以减少“算子完成后再由控制面启动通信”的间隙，更接近 NVIDIA GIN/NVSHMEM/DeepEP 方向。

## 主要劣势和风险

1. **破坏统一 stream 语义**

   STARS 统一调度 SDMA、notify、event、RDMA doorbell 等任务。绕过 STARS 后，RDMA WQE 与 AI Core/SDMA/local reduce/notify 之间的先后关系需要重新建立，否则会出现数据尚未就绪就发包、远端 reduce 尚未完成就 post notify 等问题。

2. **completion 与错误处理复杂度显著上升**

   直接写 WQE 后必须自己处理 CQE、CQ overflow、QP error、retry exceeded、RNR、timeout、链路异常、重传、flush error 等状态。现有 HCCL 的 DFX、profiling、timeout、task mirror 等机制也要重做映射。

3. **资源安全和隔离问题**

   RDMA QP、doorbell、MR token、remote key 等资源通常由驱动/运行时严格管理。让 AICPU 直接写 NIC 队列意味着 AICPU 要持有底层资源地址和权限，容易扩大故障半径，也不利于多租户隔离和版本兼容。

4. **AICPU 可能成为新的瓶颈**

   AICPU 适合控制逻辑，但不是高吞吐 WQE 生成引擎。对高带宽集合通信，如果每个 chunk/rank 都由 AICPU 逐项填 WQE，AICPU 单线程/少线程、cache、访存、MMIO doorbell 开销可能吞掉收益。

5. **芯片/网卡强绑定，维护成本高**

   直接填 NIC WQE 会暴露网卡私有格式。不同芯片、不同 RDMA 引擎、不同固件版本、不同 WQE layout 都会影响 HCCL 数据面，削弱 HCOMM 当前通过 transport 抽象屏蔽底层差异的价值。

6. **调度优化空间反而变窄**

   STARS 可以统一看到本地 copy、reduce、notify、event、doorbell 的依赖关系。完全绕过后，网络调度和本地调度割裂，后续做计算通信融合、QoS、优先级、profiling、重放等都更难。

## 推荐方案：性能优先的 fast path

不建议把所有 HCCL 集合通信都改成裸 RDMA WQE，但建议明确建设一个高性能 fast path：

1. **短期：AICPU + WQE template + batch doorbell**

   通信域建链阶段预生成每个 peer/link 的 WQE 模板；执行时 AICPU 只 patch local offset、remote offset、length、imm/notify 等字段；多个 peer/chunk 合并 doorbell。

2. **中期：保留 STARS ordering，但跳过高层 task 解释开销**

   对 `BatchWrite/BatchRead/AlltoAll/Dispatch/Combine` 这类规则通信，直接从 op desc 生成 compact command list，不再逐条走复杂 transport/instruction 解释。

3. **长期：AI Core/CCU/NIC 发起通信**

   真正对齐 NVIDIA 的不是“AICPU 直驱”，而是“计算 kernel 或专用通信单元直接触发网络”。AICPU 可以负责建链和生成模板，数据面触发应尽量靠近数据生产者或网卡。

## 结论

从性能角度看，直驱大概率是正确方向，尤其适合小包、高频、规则化通信：MoE dispatch/combine、batch write、alltoall direct fullmesh、one-sided write/reduce 等。NVIDIA 的 GIN/NVSHMEM/DeepEP 路线也说明，GPU/kernel initiated communication 是降低控制面延迟和实现计算通信融合的重要方向。

但落到 Ascend/HCCL，第一优先级不应是“让 AICPU 裸写所有 NIC WQE”，而应是：

- 建链阶段预生成 RDMA WQE/template；
- AICPU 执行时只 patch 少数字段；
- 对规则小包通信走 fast path；
- ordering/completion/error 由 STARS/CCU/NIC firmware 保底；
- 长期把触发点从 AICPU 推到 AI Core kernel 或 CCU。

一句话：**为了性能，应该做直驱；但最优直驱不是 AICPU 变成 RDMA 驱动，而是 AICPU 生成模板，AI Core/CCU/NIC 在数据面快速触发。**
