# 环境注意事项

本次目录整理没有安装依赖、修改算法超参数或重新训练模型。

## 3d-OT 的 R 初始化

当前 Python 来自 STAIR-env。若未指定 R_HOME，rpy2 会探测到系统 R
`/usr/lib/R`，但当前 conda/rpy2 组合可能报错：
`no item called "package:utils" on the search list`。

使用现有 conda R 目录即可通过入口检查：

```bash
R_HOME=/root/miniconda3/envs/STAIR-env/lib/R \
  /root/miniconda3/envs/STAIR-env/bin/python \
  benchmark/scripts/runners/clustering/run_3dot_recommended_clustering.py --help
```

该检查成功加载已有 mclust 6.1.2，没有新增依赖。今后真正重新训练时仍应
记录 R/Python 包版本及实际环境，不能把一次 --help 检查称为完整训练验证。

## 本次入口检查

九个运行/发布入口已检查。八个在默认环境中通过；3d-OT 使用上面的
R_HOME 后通过。最初默认环境的检查记录保留在 `entrypoint_smoke_test.json`，
最终验证摘要在 `layout_validation.json`，不覆盖初次检查中的失败信息。

Python 自动生成的 `__pycache__` 是可重建缓存，不是实验输出，已通过
`benchmark/.gitignore` 排除；正式代码阅读入口是脚本索引而非缓存目录。
