

## Remote AutoDL server

### Server2
ssh -p 16531 root@connect.bjb2.seetacloud.com
password: 9EyLfeGrqOxs


### Server6
ssh -p 41038 root@connect.westd.seetacloud.com
password: cyi8gWNJki3l


- start server vpn: source /etc/network_turbo
- close server vpn: unset http_proxy && unset https_proxy
- github access: do remember to open vpn before you cloning anything.


### 服务器环境 （没有则创建）
- 安装 miniconda，安装到 /root/autodl-tmp 目录下， 然后创建一个软连接， 将 /root/autodl-tmp/miniconda 链接到 /root/miniconda
- conda 环境存储目录： /root/autodl-tmp/envs
- conda create -n llm python=3.12
- conda activate llm
- 模型权重目录：`/root/autodl-tmp/models/`
- 数据集目录: 视具体项目要求而定
- Screen (长期执行终端命令):
    - 创建后台会话: `screen -dmS <session_name> bash -c '<command>'`
    - 创建并进入会话: `screen -S <session_name>`
    - 查看所有会话: `screen -ls`
    - 重新连接会话: `screen -r <session_name>`
    - 断开当前会话 (不终止): 按 `Ctrl+A` 然后按 `D`
    - 终止会话: `screen -X -S <session_name> quit`
    - 常用示例:
        - 后台训练: `screen -dmS train bash -c 'conda activate llm && python train.py 2>&1 | tee train.log'`
        - 双 GPU 并行: 分别创建两个 screen，各自指定 `CUDA_VISIBLE_DEVICES=0` 和 `CUDA_VISIBLE_DEVICES=1`
        - 查看后台日志: `tail -f train.log` 或 `screen -r train` 进入查看
    - 注意事项:
        - SSH 断开后 screen 会话仍然存活，重连后用 `screen -r` 恢复
        - 建议用 `tee` 同时输出到日志文件，方便后续检查
        - 如果 screen -r 提示 "Attached"，用 `screen -d -r <name>` 强制接管
- How to download datasets: you may want use huggingface, hf-mirror, modelscope, github, or whatever source you like.



## 开发要求
- 每当开发完某一个阶段的方案/任务， 你都必须将最新的开发进度记录到项目目录下的 `docs/开发进度` 文件夹中。



## 项目部署要求
- 所有项目代码都应该持续在本地保留最新的版本， 如需上传到 remote server 进行部署（比如在开发过程中遇到需要测试、运行的情况）， 那就把本地的最新项目代码上传上去。 如果 remote server 出现 bug， 必须先在本地进行修改， 再二次上传到远端。禁止直接在远端修改代码。

- 远端的所有项目代码都要放到 /root/autodl-tmp 目录下

