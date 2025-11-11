# Release 0.8.0

## 主要依赖库版本描述
- python 3.9 或者更高
- [torch-mlu] 2.7.1；
- [torch-npu] 2.7.1；
- pytorch与torch-mlu或torch-npu对应；
- [triton-x] 3.2.0 或者更高；

## 重大变动
-

## 主要特性与改进
- MLU后端新增FusedCombineMatMul推理pattern，允许在MLU推理场景下，将多个shape相同的线性层操作（matmul+bias+act）融合为一个kernel，以提高利用率; #384

## Bug修复与其他改动
-

---

# Release 0.7.0

## 主要依赖库版本描述
- python 3.9 或者更高
- pytorch 2.7.1 或者更高
- [torch-mlu] 与pytorch版本对应
- [torch-npu] 与pytorch版本对应
- [triton-x] 3.2.0 或者更高

## 重大变动
- Npu后端新增默认的`vendor_compiler_config`配置，此后，我们正式约定，**None表示禁用后端编译器，{}表示使用默认后端编译器**,
  此外，先前版本中，对于Npu后端，如果传入`{}`，默认将使能`inductor`，现在已经改为了`AclGraph`，对于MLU后端，我们没有变化; #444

## Bug修复与其他改动
- 修复`vendor_compiler_config`默认为`None`导致运行失败的问题; #444

---

# Release 0.6.0

## 主要依赖库版本描述
- python 3.9 或者更高
- pytorch 2.5.1 或者更高
- [torch-mlu] 与pytorch版本对应
- [torch-npu] 与pytorch版本对应
- [triton-x] 3.2.0 或者更高

## 重大变动
-

## 主要特性与改进
- 支持训练阶段指定计算图前反向拆分策略，可以通过参数`partition_fn`指定，也可以通过环境变量`XPUGRAPH_PARTITIONER`指定。 #430
    * `"DEFAULT"`: 使用与eager行为一致的前反向拆分（`torch._functorch.partitioners.default_partition_fn`）
    * `"MINCUT"`: 使用显存优化的前反向拆分 (`torch._functorch.partitioners.min_cut_rematerialization_partition`)
    * 也可以指定自定义的partition_fn实现定制的前反向拆分策略（通过环境变量指定时使用函数的fqn进行指定）
    * 未指定时效果同`"DEFAULT"`


## Bug修复与其他改动
- 修复fold_view导致view shape推导在inductor内失效的问题。 #435
- 修复常量折叠无法处理特定dtype的tensor的问题。 #428

---

# Release 0.5.2

## 主要依赖库版本描述
- python 3.9 或者更高
- pytorch 2.5.1 或者更高
- [torch-mlu] 与pytorch版本对应
- [torch-npu] 与pytorch版本对应
- [triton-x] 3.2.0 或者更高

## 重大变动
-

## 主要特性与改进
- 现在你可以通过结构化字典的方式，将npu compiler的选项设置透传到过去，例如：
  - ```
    vendor_cfg = {'mode': 'reduce-overhead', 'aclgraph_config': {'use_custom_pool': mem_pool_handl}}
    xpu_cfg =  XpuGraphConfig(
                is_training=False,
                target=Target.npu,
                vendor_compiler_config=vendor_cfg)
    # This also works for now and will be deprecated after THREE versions.
    vendor_cfg = {'mode': 'reduce-overhead', 'use_custom_pool': mem_pool_handl}
    xpu_cfg =  XpuGraphConfig(
                is_training=False,
                target=Target.npu,
                vendor_compiler_config=vendor_cfg)

    ```
  目前我们仍然兼容原来的选项写法，但接下来三个版本后，我们将废弃原有的`use_custom_pool`字段，并不再新增npu compiler选项字段，请参考官方的选项用法。 #399
- 增加对横向融合pattern的支持，可以将多个无依赖的pointwise算子融合成一次调用，减少host开销。 #390
- 增加AddN pattern，将多次add融合为stack sum。 #248
- 在存在高阶op和subclass tensor的情况下，使用aot_autograd来进行编译，以避免dispatch失败。 #191

## Bug修复与其他改动
- 优化日志打印。 #423
- 修复fold pass带来的stride变化导致view失败的问题，将所有view操作替换为reshape。 #413

---

# Release 0.5.1

## 主要依赖库版本描述
- python 3.9 或者更高
- pytorch 2.5.1 或者更高
- [torch-mlu] 与pytorch版本对应
- [torch-npu] 与pytorch版本对应
- [triton-x] 3.2.0 或者更高

## 重大变动
- 优化推理场景动态shape支持，修复不支持动态shape的pass
- 支持部分业务使用uv包管理

## 主要特性与改进
- 增加动态shape相关的功能函数，避免pattern中产生额外的shape guard

## Bug修复与其他改动
- 修复 binary op 的 fold pattern 在存在 type promotion 时 dtype 与原结果不一致的问题
- 修复 fold 类的 pattern 在计算图中的迭代顺序
- 修复 change_tensor_like pattern 未处理 dtype 参数的问题
- 优化日志打印

---

# Release 0.5.0

## 主要依赖库版本描述
- python 3.9 或者更高
- pytorch 2.5.1 或者更高
- [torch-mlu] 与pytorch版本对应
- [torch-npu] 与pytorch版本对应
- [triton-x] 3.2.0 或者更高

## 重大变动
- 新增运行时精度监控 (#272)，可以在运行时将编译产物的前反向精度与优化前的fx graph进行对比，帮助用户定位精度问题
- 补充 Apache 2.0 协议 (#379)

## 主要特性与改进
- 优化 slicelike folding pattern：#366
  - 消除 noop-slice（Pattern: y = slice(x, dim, 0, len(x)(or inf)) -> Becomes: y = x）
  - 消除 noop-slicescatter（Pattern: y = slice_scatter(base, view, ...) -> Becomes: y = view）
- 优化 dense 类 pattern: #279
  - 增加 addmm, baddbmm, SDP attention 融合（ common pattern ）
  - 优化 denselayer structure pattern，支持后端特化的 matmul/bmm + add + activation 后融合
  - 优化 densetower structure pattern，支持后端特化的多层 FFN 融合

## Bug修复与其他改动
- 修复 mlu inductor 的 cpp_wrapper 设置，并避免默认值覆盖 TORCHINDUCTOR_CPP_WRAPPER 环境变量 （#367，#381）
- 修复了 reduce(dim=None) 编译失败的问题 （#365）

---

# Release 0.4.0

## 主要依赖库版本描述
- python 3.9 或者更高
- pytorch 2.5.1 或者更高
- [torch-mlu] 与pytorch版本对应
- [torch-npu] 与pytorch版本对应
- [triton-x] 3.2.0 或者更高

## 重大变动
- 现在默认不再忽略literal匹配了，如果需要忽略，请在注册PluginPattern时使用关键字参数`ignore_literal=True`声明；

## 主要特性与改进
- PluginPattern现在支持字面值(`float`,`bool`和`int`)类型的继承重写，只需要将其作为Pattern函数的输入参数传入即可，有以下几点需要注意：#350
  - 作为example input传入的literal，必须确保是全局唯一的，本身也是假的无意义的，因此理论上可以选择任意值来保证唯一性；
  - 对于不需要继承重写的literal，不要作为输入传入，且需要确保有意义，特别是会影响到图节点生成；
- 现在我们通过`vendor_compiler_config`的`enable_super_kernel`来打开`super kernel`优化；#345

## Bug修复与其他改动
- 修复了PyTorch2.7.0下使用`MLU CppWrapper`失败的问题；#328
- 修复了`layernorm+cast`的类型转换问题; #341
- 现在我们不再对`non-inductor-based`的后端，比如`GE`和`AclGraph`，执行图分解下降，以免引入不支持的Op类型；#351

---

# Release 0.3.0

## 主要依赖库版本描述
- python 3.9 或者更高
- pytorch 2.5.1 或者更高
- [torch-mlu] 与pytorch版本对应
- [torch-npu] 与pytorch版本对应
- [triton-x] 3.2.0

## 重大变动
-

## 主要特性与改进
- 新增`vendor_compiler_config`字段`use_custom_pool`，用户可以通过`torch.npu.graph_pool_handle()`获取内存池并传入，从而实现多图的内存复用，但切忌并发使用共享内存池的多图，以免造成不必要的内存踩踏问题; #324

## Bug修复与其他改动
- 由于`graphviz`在处理大图时速度非常慢，现在使能`dump_graph`，我们不再以`svg`格式绘制并导出，而是以`dot`格式保存，并附上绘图代码，用户可以根据需要自行绘图，请确保绘图环境里安装了`graphviz`和python模块`pydot`；此外，我们还将图表示文件的后缀从`.ll`改成了`.txt`，这是为了能够在一些在线文档中直接预览，而无需下载查看; #286

---

# Release 0.2.0

## 主要依赖库版本描述
- python 3.9 或者更高
- pytorch 2.5.1 或者更高
- [torch-mlu] 与pytorch版本对应
- [torch-npu] 与pytorch版本对应
- [byted-triton-x] 3.2.0

## 重大变动
- 现在whl包更名为`byted-xpu-graph`，并在bytedpypi源上发布，注意与旧版本的包名不同 #313;
- 现在`Target.ascend`被删除且合并到`Target.npu`中，`Target.ascend`和`Target.npu`原先语义，可以通过`vendor_compiler_mode:{compiler: 'ge'}`以及`vendor_compiler_mode:{compiler: 'inductor'}`区分 #274;

## 主要特性与改进
- 现在可以通过环境变量`XPUGRAPH_DEPRECATED_AOT_CONFIG_IS_EXPORT`来控制AOTConfig的`is_export`属性，这可以解决部分`immutable functional tensor`的问题，但副作用不明确，注意未来我们将弃用这一环境变量，使用该环境变量前务必明白使用的意义，否则可能会导致一些未知的问题 #311;
- 现在`Target.npu`支持对`torch.ops.npu.npu_quant_matmul`的权重进行`nd2nz`的转化，通过常量折叠完成编译期优化 #239;
- 现在`Target.npu`支持对`add+rmsNorm`和`silu+mul` 两个pattern基于triton算子进行融合，需要使能`OptLevel.level2`;
- 增加 rms_norm pattern，将常见的rms_norm小算子实现进行替换 #297;
- 现在`Target.mlu`下如果传入`vendor_compiler_mode=none`，则不会使用mlu的inductor编译器 #252;

## Bug修复与其他改动
- 在时间戳之后增加了进程与线程ID的打印，方便调试 #270;

---
