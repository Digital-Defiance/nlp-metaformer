FROM prefecthq/prefect:2-python3.10

RUN pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118
RUN pip install pydantic duckdb numpy pydantic-settings safetensors tiktoken


LABEL com.nvidia.volumes.needed="nvidia_driver"
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH
ENV PYTORCH_VERSION 2.2.0
