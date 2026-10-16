# 空间群边轨道 tokenizer：核心研究代码

本包展示离散prototype/连续几何拆分与周期图展开，不包含完整扩散生成器或训练数据。

## 内容

- `codec/discrete_graph_prototype.py`：整数prototype、仿射位点约束和周期图展开。
- `codec/symmetry_graph_codec.py`：给定对称几何时的边轨道编码。
- `codec/audit_symmetry_graph.py`：包含从CIF构建/检查近邻图的`process`函数；其批处理入口需要原研究数据目录，不是独立命令。
- `examples/`：MP-20 `mp-30009`的prototype/几何分离文件，以及最小检查脚本。该例用于解释表示，不用于证明泛化。
- `evidence/`：全量训练版开发结果、冻结模型的独立复核结果和ASU强对照。
- `方案说明.md`：方法定义、已有工作边界与实验限制。

## 最小检查

在有NumPy的Python环境运行：

```sh
python examples/inspect_example.py
```

原实验在Burgundy现有matgl环境运行。核心展开只依赖NumPy（测试版本1.26.4）；CIF编码还使用spglib 2.6.0和pymatgen 2025.6.14。模型实验另外使用PyTorch 2.2.0；训练脚本与完整记录保留在原workspace的`experiments/wyckoff_tokenizer_search/`。

脚本用独立随机自由参数实例化同一个prototype，检查抽象周期图是否保持一致。随机实例不代表物理合理或稳定材料。几何必须与给定空间群兼容；图展开器不是材料有效性检查器。

## 证据的正确解释

相对ASU逐邻居路径，512图批次的峰值分配内存减少44.6%，但时间增加11.7%；输出与梯度保持一致。单级条件几何去噪在5343个独立保留结构上，比不做修正的MSE下降71.16%；这不是相对ASU的精度提升，也不是完整扩散采样或新材料稳定性结果。

当前prototype内联操作表，不能把节点/边记录减少直接称为最终文件字节或语言模型token压缩率。近邻边来自4.5Å半径图，不全部等同于化学键。

例子数据来自WyFormer公开MP-20数据，固定commit `e8e9add5df904284116af8a06c203555d908f290`：https://github.com/SymmetryAdvantage/WyckoffTransformer 。
