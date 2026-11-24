# Copyright (c) OpenMMLab. All rights reserved.
"""Process-start trace flags shared by engine and service code; no service
imports."""
import os


def _flag(name):
    return os.getenv(name, '0').strip().lower() in ('1', 'true', 'yes', 'on', 'y')


LOG_ENABLED = _flag('LMDEPLOY_TRACE_ENABLED')
OTLP_ENABLED = bool(os.getenv('OTEL_EXPORTER_OTLP_TRACES_ENDPOINT')
                    or os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT'))
ENABLED = LOG_ENABLED or OTLP_ENABLED
