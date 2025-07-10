# 生产环境指标监控

LMDeploy 通过 Prometheus 暴露监控指标，并通过 Grafana 提供可视化界面。

## 配置指南

本节介绍如何设置 `lmdeploy/monitoring` 目录中提供的监控套件（Prometheus + Grafana）

## 前提条件

- 已安装 [Docker](https://docs.docker.com/engine/install/) 和 [Docker Compose](https://docs.docker.com/compose/install/)

- 已启用指标系统的 LMDeploy 服务正在运行

## 使用说明 (DP = 1)

1. **启动已启用指标的 LMDeploy 服务**

```
lmdeploy serve api_server Qwen/Qwen2.5-7B-Instruct --enable-metrics
```

请根据需求替换模型路径。默认 metrics endpoint 位于 `http://<lmdeploy_server_host>:23333/metrics`。

2. **进入监控目录**

```
cd lmdeploy/monitoring
```

3. **启动监控套件**

```
docker compose up
```

此命令将在后台启动 Prometheus 和 Grafana。

4. **访问监控界面**

- Prometheus：浏览器访问 http://localhost:9090.

- Grafana：浏览器访问 http://localhost:3000.

5. **登录 Grafana**

- 默认用户名：`admin`

- 默认密码：`admin` （首次登录后会提示修改密码）

6. **查看仪表盘**

预配置的 LMDeploy 仪表盘将自动加载。

## 使用说明 (DP > 1)

1. **启动已启用指标的 LMDeploy 服务**

以模型 `Qwen/Qwen2.5-7B-Instruct` 为例，使用 `DP=2，TP=2` 启动服务：

```bash
# Proxy server
lmdeploy serve proxy --server-port 8000 --routing-strategy 'min_expected_latency' --serving-strategy Hybrid --log-level INFO

# API server
LMDEPLOY_DP_MASTER_ADDR=127.0.0.1 \
LMDEPLOY_DP_MASTER_PORT=29555 \
lmdeploy serve api_server \
    Qwen/Qwen2.5-7B-Instruct \
    --backend pytorch \
    --tp 2 \
    --dp 2 \
    --proxy-url http://0.0.0.0:8000 \
    --nnodes 1 \
    --node-rank 0 \
    --enable-metrics
```

您应该能在代理服务器列表中看到多个 API 服务实例。详细信息可以在 `lmdeploy/serve/proxy/proxy_config.json` 中找到。

例如，您可能会看到如下 API 服务地址：

```
http://$host_ip:$api_server_port1

http://$host_ip:$api_server_port2
```

2. **修改 Prometheus 配置**

当 DP > 1 时，LMDeploy 会为每个 DP Rank 启动一个 API 服务。如果你想监控其中某个 API 服务，例如：`http://$host_ip:$api_server_port1`，请修改配置文件 `lmdeploy/monitoring/prometheus.yaml` 如下所示。

> 注意：这里应使用实际主机的 IP 地址而非 127.0.0.1，因为当 DP > 1 时，LMDeploy 是通过实际主机 IP 启动 API 服务的。

```
global:
  scrape_interval: 5s
  evaluation_interval: 30s

scrape_configs:
  - job_name: lmdeploy
    static_configs:
      - targets:
          - '$host_ip:$api_server_port1' # <= 修改此处
```

3. **进入监控目录并执行上述相同步骤**

## 故障排除

1. **端口冲突**

检查端口 `23333` (LMDeploy 服务端口)、`9090` (Prometheus 端口) 或 `3000` (Grafana 端口) 是否被占用。解决方案，关闭冲突的端口或如下修改配置文件：

- 修改 Prometheus 抓取的 LMDeploy 服务端口

在 `lmdeploy/monitoring/prometheus.yaml` 中

```
global:
  scrape_interval: 5s
  evaluation_interval: 30s

scrape_configs:
  - job_name: lmdeploy
    static_configs:
      - targets:
          - '127.0.0.1:23333' # <= 修改此处的 LMDeploy 服务端口 23333，需与实际运行端口一致
```

- 修改 Prometheus 端口

在 `lmdeploy/monitoring/grafana/datasources/datasource.yaml` 中

```
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://localhost:9090 # <= 修改此处的 Prometheus 接口端口 9090
    isDefault: true
    editable: false
```

- 修改 Grafana 端口

在 `lmdeploy/monitoring/docker-compose.yaml` 中操作（例如改为 3090 端口）:

方案一：在环境变量中添加 `GF_SERVER_HTTP_PORT`

```
  environment:
- GF_AUTH_ANONYMOUS_ENABLED=true
- GF_SERVER_HTTP_PORT=3090  # <= 添加此行
```

方案二：使用端口映射

```
grafana:
  image: grafana/grafana:latest
  container_name: grafana
  ports:
  - "3090:3000"  # <= 主机端口:容器端口映射
```

- **仪表盘无数据**

尝试向 LMDeploy 服务发送请求生成流量：

```
python3 benchmark/profile_restful_api.py --backend lmdeploy --num-prompts 5000 --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json
```

刷新后仪表盘应显示数据。
