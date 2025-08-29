# Release 0.5.0

## 主要依赖库版本描述
- python 3.9 或者更高
- pytorch 2.5.1 或者更高
- [torch-mlu] 与pytorch版本对应
- [torch-npu] 与pytorch版本对应
- [triton-x] 3.2.0 或者更高

## 重大变动

## 主要特性与改进
- 优化 slicelike folding pattern：#366
  - 消除 noop-slice（Pattern: y = slice(x, dim, 0, len(x)(or inf)) -> Becomes: y = x）
  - 消除 noop-slicescatter（Pattern: y = slice_scatter(base, view, ...) -> Becomes: y = view）
- 优化 dense 类 pattern: #279
  - 增加 addmm, baddbmm, SDP attention 融合（ common pattern ）
  - 优化 denselayer structure pattern，支持后端特化的 matmul/bmm + add + activation 后融合
  - 优化 densetower structure pattern，支持后端特化的多层 FFN 融合

## Bug修复与其他改动
