# 缓存临时文件写入错误与暂停记录

2026-11-03，用户报告写入 `outputs/nvidia_cache/extraction_status.json.tmp` 时发生 PermissionError。

原实现对 JSON 和 NPZ 使用固定临时文件名，没有在提取会话开始时强制独占写锁。两个 CLI 同时写同一目录时会争用临时文件；即使只改为唯一临时名，各进程持有的旧 manifest 仍可能覆盖其他进程的新增记录。因此本次同时修复两层问题：

- `save_json` 和 NPZ 保存改用目标目录内的唯一临时文件，写入、flush/关闭后再原子替换，只清理本次创建的临时文件；写入或替换失败继续抛出错误。
- 缓存 CLI 在读取 manifest、初始化 encoder 或调用 API 前取得 `.writer.lock` 独占锁，持有至提取结束。重复启动立即给出清楚错误。异常或进程退出会释放操作系统文件锁，不按锁文件是否存在判断任务状态。

独立检查结果：缓存目录可以创建并写入唯一临时文件；当前无活动提取进程；manifest 与 NPZ 均为497条，所有记录有限值、ID、维度及 checksum 校验通过，没有孤立文件。既有缓存未删除或重算。

另一次实际提取任务已在变异 `4-54288779-T-C` 因 NVIDIA 请求失败或超时退出（processed=497, failed=1）。该网络错误和用户报告的文件写入错误分开记录；文件锁修复不等于网络问题已经确认解决。

按用户要求，当前暂停 API 提取与真实 Teacher/Student 训练。已有497条可用于后续显式断点续跑；不能把缓存数量或离线测试通过当作训练完成。完整检查在 `outputs/training_run_20261103/cache_integrity_after_failure.json`。

旧源文件备份在 `work/cache_writer_fix_20261103_backup`。`.env` 未修改。

