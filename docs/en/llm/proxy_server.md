# Request Distributor Server

The request distributor service can parallelize multiple api_server services. Users only need to access the proxy URL, and they can indirectly access different api_server services. The proxy service will automatically distribute requests internally, achieving load balancing.

## Startup

Start the proxy service:

```shell
python3 -m lmdeploy.serve.proxy.proxy --server_name {server_name} --server_port {server_port} --strategy "min_expected_latency"
```

After startup is successful, the URL of the proxy service will also be printed by the script. Access this URL in your browser to open the Swagger UI.

## API

Through Swagger UI, we can see multiple APIs. Those related to api_server node management include:

- /nodes/status
- /nodes/add
- /nodes/remove

They respectively represent viewing all api_server service nodes, adding a certain node, and deleting a certain node.

APIs related to usage include:

- /v1/models
- /v1/chat/completions
- /v1/completions

The usage of these APIs is the same as that of api_server.

## Dispatch Strategy

The current distribution strategies of the proxy service are as follows:

- random： dispatches based on the ability of each api_server node provided by the user to process requests. The greater the request throughput, the more likely it is to be allocated. Nodes that do not provide throughput are treated according to the average throughput of other nodes.
- min_expected_latency： allocates based on the number of requests currently waiting to be processed on each node, and the throughput capability of each node, calculating the expected time required to complete the response. The shortest one gets allocated. Nodes that do not provide throughput are treated similarly.
- min_observed_latency： allocates based on the average time required to handle a certain number of past requests on each node. The one with the shortest time gets allocated.
