FROM python:3.10-slim


# 检查 Python 版本
RUN python --version

# 设置 Python 的输出不被缓存
ENV PYTHONUNBUFFERED 1

# 设置 pip 超时时间
ENV PIP_DEFAULT_TIMEOUT=500
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 创建一个名为 user 的用户组和用户
RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output/images/thoracic-abdominal-ct \
&& chown -R user:user /opt/app /input /output


# 切换到 user 用户
USER user
# 设置工作目录为 /opt/app
WORKDIR /opt/app
# 将脚本目录添加到 PATH 环境变量中
ENV PATH="/home/user/.local/bin:$PATH"

# 设置 CUDA 库路径
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# 复制 requirements.txt 文件到工作目录，并设置权限
COPY --chown=user:user requirements.txt /opt/app/
# 复制 resources 目录到工作目录，并设置权限
COPY --chown=user:user resources /opt/app/resources/


# 安装其他 Python 依赖，包括 nnunetv2 从 GitHub 安装，并设置 pip 超时时间
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --timeout 9999 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --requirement /opt/app/requirements.txt

# 复制应用代码文件到工作目录，并设置权限
COPY --chown=user:user inference.py /opt/app/

# 设置容器启动时执行的命令
ENTRYPOINT ["python", "inference.py"]




