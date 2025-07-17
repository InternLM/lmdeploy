# Production Metrics

LMDeploy exposes a set of metrics via Prometheus, and provides visualization via Grafana.

## Setup Guide

This section describes how to set up the monitoring stack (Prometheus + Grafana) provided in the `lmdeploy/monitoring` directory.

## Prerequisites

- [Docker](https://docs.docker.com/engine/install/) and [Docker Compose](https://docs.docker.com/compose/install/) installed

- LMDeploy server running with metrics system enabled

## Usage (DP = 1)

1. **Start your LMDeploy server with metrics enabled**

```
lmdeploy serve api_server Qwen/Qwen2.5-7B-Instruct --enable-metrics
```

Replace the model path according to your needs.
By default, the metrics endpoint will be available at `http://<lmdeploy_server_host>:23333/metrics`.

2. **Navigate to the monitoring directory**

```
cd lmdeploy/monitoring
```

3. **Start the monitoring stack**

```
docker compose up
```

This command will start Prometheus and Grafana in the background.

4. **Access the monitoring interfaces**

- Prometheus: Open your web browser and go to http://localhost:9090.

- Grafana: Open your web browser and go to http://localhost:3000.

5. **Log in to Grafana**

- Default Username: `admin`

- Default Password: `admin` You will be prompted to change the password upon your first login.

6. **View the Dashboard**

The LMDeploy dashboard is pre-configured and should be available automatically.

## Usage (DP > 1)

1. **Start your LMDeploy server with metrics enabled**

As an example, we use the model `Qwen/Qwen2.5-7B-Instruct` with `DP=2, TP=2`. Start the service as follows:

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

You should be able to see multiple API servers added to the proxy server list. Details can be found in `lmdeploy/serve/proxy/proxy_config.json`.

For example, you may have the following API servers:

```
http://$host_ip:$api_server_port1

http://$host_ip:$api_server_port2
```

2. **Modify the Prometheus configuration**

When `DP > 1`, LMDeploy will launch one API server for each DP rank. If you want to monitor a specific API server, e.g. `http://$host_ip:$api_server_port1`, modify the configuration file `lmdeploy/monitoring/prometheus.yaml` as follows.

> Note that you should use the actual host machine IP instead of `127.0.0.1` here, since LMDeploy starts the API server using the actual host IP when `DP > 1`

```
global:
  scrape_interval: 5s
  evaluation_interval: 30s

scrape_configs:
  - job_name: lmdeploy
    static_configs:
      - targets:
          - '$host_ip:$api_server_port1' # <= Modify this
```

3. **Navigate to the monitoring folder and perform the same steps as described above**

## Troubleshooting

1. **Port conflicts**

Check if any services are occupying ports `23333` (LMDeploy server port), `9090` (Prometheus port), or `3000` (Grafana port). You can either stop the conflicting running ports or modify the config files as follows:

- Modify LMDeploy server port for Prometheus scrape

In `lmdeploy/monitoring/prometheus.yaml`

```
global:
  scrape_interval: 5s
  evaluation_interval: 30s

scrape_configs:
  - job_name: lmdeploy
    static_configs:
      - targets:
          - '127.0.0.1:23333' # <= Modify this LMDeploy server port 23333, need to match the running server port
```

- Modify Prometheus port

In `lmdeploy/monitoring/grafana/datasources/datasource.yaml`

```
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://localhost:9090 # <= Modify this Prometheus interface port 9090
    isDefault: true
    editable: false
```

- Modify Grafana port:

In `lmdeploy/monitoring/docker-compose.yaml`, for example, change the port to `3090`

Option 1: Add `GF_SERVER_HTTP_PORT` to the environment section.

```
  environment:
- GF_AUTH_ANONYMOUS_ENABLED=true
- GF_SERVER_HTTP_PORT=3090  # <= Add this line
```

Option 2: Use port mapping.

```
grafana:
  image: grafana/grafana:latest
  container_name: grafana
  ports:
  - "3090:3000"  # <= Host:Container port mapping
```

2. **No data on the dashboard**

- Create traffic

Try to send some requests to the LMDeploy server to create certain traffic

```
python3 benchmark/profile_restful_api.py --backend lmdeploy --num-prompts 5000 --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json
```

After refreshing, you should be able to see data on the dashboard.
