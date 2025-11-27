class MoEBackend:
    def __init__(self):
        """Initialize moe backend."""
        self._use_deepep_moe_backend = False

    def set_deepep_moe_backend(self):
        """Set deepep moe backend"""
        self._use_deepep_moe_backend = True

    def use_deepep_moe_backend(self):
        """Get deepep moe backend"""
        return self._use_deepep_moe_backend


MOE_BACKEND = None

def get_moe_backend():
    global MOE_BACKEND
    if MOE_BACKEND is None:
        MOE_BACKEND = MoEBackend()

    return MOE_BACKEND

moe_backend = get_moe_backend()
