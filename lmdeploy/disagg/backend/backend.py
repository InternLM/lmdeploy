from lmdeploy.disagg.messages import MigrationBackend


MIGRATION_BACKENDS = {}


def register_migration_backend(backend_name: MigrationBackend):
    def register(cls):
        MIGRATION_BACKENDS[backend_name] = cls
        return cls

    return register
