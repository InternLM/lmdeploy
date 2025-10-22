# Copyright (c) OpenMMLab. All rights reserved.
from logging import Logger
from typing import List

from lmdeploy.utils import can_colorize, get_logger

RED_COLOR = '\033[31m'
RESET_COLOR = '\033[0m'


def _red_text(text: str):
    """Red text."""
    if not can_colorize():
        return text
    return f'{RED_COLOR}{text}{RESET_COLOR}'


class BaseChecker:
    """Base checker."""

    def __init__(self, logger: Logger = None):
        if logger is None:
            logger = get_logger('lmdeploy')
        self.logger = logger
        self._is_passed = False
        self._required_checker: List[BaseChecker] = list()

    def get_logger(self):
        """Get logger."""
        return self.logger

    def register_required_checker(self, checker: 'BaseChecker'):
        """register_required."""
        self._required_checker.append(checker)

    def handle(self):
        """Handle check."""
        is_passed = getattr(self, '_is_passed', False)
        if not is_passed:
            checker_name = type(self).__name__
            self.logger.debug(f'Checking <{checker_name}>:')
            for checker in self._required_checker:
                checker.handle()
            self.check()
            self.is_passed = True

    def log_and_exit(self, e: Exception = None, mod_name: str = None, message: str = None):
        logger = self.logger
        if mod_name is None:
            mod_name = type(self).__name__
        if message is None:
            message = 'Please check your environment.'
        logger.debug('Exception', exc_info=1)
        if e is not None:
            logger.error(f'{type(e).__name__}: {e}')
        logger.error(f'<{mod_name}> check failed!\n{_red_text(message)}')
        exit(1)

    def check(self):
        """check."""
        raise NotImplementedError('check not implemented.')
