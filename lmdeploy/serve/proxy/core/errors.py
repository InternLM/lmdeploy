# Copyright (c) OpenMMLab. All rights reserved.

import enum


class ErrorCodes(enum.Enum):
    """Error codes."""

    MODEL_NOT_FOUND = 10400
    SERVICE_UNAVAILABLE = 10401
    API_TIMEOUT = 10402


err_msg = {
    ErrorCodes.MODEL_NOT_FOUND: 'The request model name does not exist in the model list.',
    ErrorCodes.SERVICE_UNAVAILABLE: 'The service is unavailable now. May retry later.',
    ErrorCodes.API_TIMEOUT: 'Failed to get response after a period of time',
}
