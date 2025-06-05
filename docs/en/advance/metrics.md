# Production Metrics

LMDeploy exposes a set of metrics via Prometheus, and provides visualization via Grafana.

## Setup Guide

This section describes how to set up the monitoring stack (Prometheus + Grafana) provided in the `lmdeploy/monitoring` directory.

## Prerequisites

- [Docker](https://docs.docker.com/engine/install/) and [Docker Compose](https://docs.docker.com/compose/install/) installed

- LMDeploy server running with metrics system enabled

## Usage

1. **Start your LMDeploy server with metrics enabled**

```
lmdeploy serve api_server models--Qwen--Qwen2.5-7B-Instruct --server-port 30000 --enable-metrics
```

Replace the model path according to your needs.
By default, the metrics endpoint will be available at `http://<lmdeploy_server_host>:30000/metrics`.

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

## Troubleshooting

1. **Port conflicts**

Check if any services are occupying ports `30000` (LMDeploy server port), `9090` (Prometheus port), or `3000` (Grafana port). You can either stop the conflicting running ports or modify the config files as follows:

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
          - '127.0.0.1:30000' # <= Modify this LMDeploy server port 30000, need to match the running server port
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

- **No data on the dashboard**

Try to send some requests to the LMDeploy server to create certain traffic

```
python3 benchmark/profile_pipeline_api.py ShareGPT_V3_unfiltered_cleaned_split.json models--Qwen--Qwen2.5-7B-Instruct
```

After refreshing, you should be able to see data on the dashboard.
