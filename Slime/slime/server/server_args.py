from dataclasses import dataclass


@dataclass
class ServerArgs:
    host: str
    port: int
