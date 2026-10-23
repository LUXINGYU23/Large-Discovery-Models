FROM ldm-copper-baseline-base:20261024
RUN pip install --no-cache-dir --index-url https://pypi.tuna.tsinghua.edu.cn/simple gstools==1.7.0 matplotlib==3.9.4 tqdm==4.67.1 numpy==1.26.4 scipy==1.14.1
COPY vendor /baseline
ENV MPLCONFIGDIR=/tmp/matplotlib
