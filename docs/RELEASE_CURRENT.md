# Release 0.8.0

## 主要依赖库版本描述
- python 3.9 或者更高
- [torch-mlu] 2.7.1；
- [torch-npu] 2.7.1；
- pytorch与torch-mlu或torch-npu对应；
- [triton-x] 3.2.0 或者更高；

## 重大变动
- 因依赖组件的变动，从当前版本（0.8.0）开始，XPU_Graph默认支持的最低torch版本调整为2.7，新增功能均基于2.7.1版本实现。

## 主要特性与改进
- MLU后端新增FusedCombineMatMul推理pattern，允许在MLU推理场景下，将多个shape相同的线性层操作（matmul+bias+act）融合为一个kernel，以提高利用率; #384

## Bug修复与其他改动
-
