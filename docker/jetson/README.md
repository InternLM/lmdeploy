# Jetson Support (beta)

Build LMDeploy for NVIDIA Jetson platforms by docker.

```sh
export JETPACK_L4T="36.2.0" # Jetpack 6.0 DP

cd lmdeploy
docker build -t lmdeploy_jetson:r$JETPACK_L4T \
    --build-arg JETPACK_VERSION=$JETPACK_L4T \
    -f docker/jetson/Dockerfile .
```

Version Corresponding List:

| $JETPACK_L4T | Jetpack Version | Python Version | Torch Version | CUDA VERSION |                                           Support Boards                                           |
| :----------: | :-------------: | :------------: | :-----------: | :----------: | :------------------------------------------------------------------------------------------------: |
|    35.2.1    |       5.1       |   Python 3.8   |     2.0.0     |     11.4     |            AGX Orin NX 32GB,<br>Orin NX 16GB,<br>Xavier NX series,<br>AGX Xavier Series            |
|    35.3.1    |      5.1.1      |   Python 3.8   |     2.0.0     |     11.4     | AGX Orin Series,<br>Orin NX Series,<br>Orin Nano Series,<br>Xavier NX Series,<br>AGX Xavier Series |
|    35.4.1    |      5.1.2      |   Python 3.8   |     2.1.0     |     11.4     | AGX Orin Series,<br>Orin NX Series,<br>Orin Nano Series,<br>Xavier NX Series,<br>AGX Xavier Series |
|    36.2.0    |     6.0 DP      |  Python 3.10   |     2.2.0     |     12.2     |                      AGX Orin Series,<br>Orin NX Series,<br>Orin Nano Series                       |
|    36.3.0    |       6.0       |  Python 3.10   |     2.4.0     |     12.2     |                      AGX Orin Series,<br>Orin NX Series,<br>Orin Nano Series                       |
