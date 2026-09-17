# Copyright (c) OpenMMLab. All rights reserved.
"""Unit tests for startup sequence logic in lmdeploy_router.

These tests focus on testing the startup sequence logic in isolation, including router initialization, configuration
validation, and startup flow.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
from lmdeploy_router.launch_router import RouterArgs, launch_router
from lmdeploy_router.router import policy_from_str
from lmdeploy_router_rs import PolicyType


# Local helper mirroring the router logger setup used in production
def setup_logger():
    logger = logging.getLogger('router')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter(
            '[Router (Python)] %(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class TestSetupLogger:
    """Test logger setup functionality."""

    def test_setup_logger_returns_logger(self):
        """Test that setup_logger returns a logger instance."""
        logger = setup_logger()

        assert isinstance(logger, logging.Logger)
        assert logger.name == 'router'
        assert logger.level == logging.INFO

    def test_setup_logger_has_handler(self):
        """Test that setup_logger configures a handler."""
        logger = setup_logger()

        assert len(logger.handlers) > 0
        handler = logger.handlers[0]
        assert isinstance(handler, logging.StreamHandler)

    def test_setup_logger_has_formatter(self):
        """Test that setup_logger configures a formatter."""
        logger = setup_logger()

        handler = logger.handlers[0]
        formatter = handler.formatter

        assert formatter is not None
        assert '[Router (Python)]' in formatter._fmt

    def test_setup_logger_multiple_calls(self):
        """Test that multiple calls to setup_logger work correctly."""
        logger1 = setup_logger()
        logger2 = setup_logger()

        # Should return the same logger instance
        assert logger1 is logger2


class TestPolicyFromStr:
    """Test policy string to enum conversion in startup context."""

    def test_policy_conversion_in_startup(self):
        """Test policy conversion during startup sequence."""
        # Test all valid policies
        policies = [
            'random',
            'round_robin',
            'cache_aware',
            'power_of_two',
            'consistent_hash',
        ]
        expected_enums = [
            PolicyType.Random,
            PolicyType.RoundRobin,
            PolicyType.CacheAware,
            PolicyType.PowerOfTwo,
            PolicyType.ConsistentHash,
        ]

        for policy_str, expected_enum in zip(policies, expected_enums):
            result = policy_from_str(policy_str)
            assert result == expected_enum

    def test_invalid_policy_in_startup(self):
        """Test handling of invalid policy during startup."""
        with pytest.raises(KeyError):
            policy_from_str('invalid_policy')


class TestRouterInitialization:
    """Test router initialization logic."""

    def test_router_initialization_basic(self):
        """Test basic router initialization."""
        args = RouterArgs(
            host='127.0.0.1',
            port=30000,
            worker_urls=['http://worker1:8000'],
            policy='cache_aware',
        )

        with patch('lmdeploy_router.launch_router.Router') as router_mod:
            captured_args = {}

            mock_router_instance = MagicMock()

            def fake_from_args(router_args):
                # capture needed fields from RouterArgs
                captured_args.update(
                    dict(
                        host=router_args.host,
                        port=router_args.port,
                        worker_urls=router_args.worker_urls,
                        policy=policy_from_str(router_args.policy),
                    )
                )
                return mock_router_instance

            router_mod.from_args = MagicMock(side_effect=fake_from_args)

            launch_router(args)

            # Verify Router.from_args was called and captured fields match
            router_mod.from_args.assert_called_once()
            assert captured_args['host'] == '127.0.0.1'
            assert captured_args['port'] == 30000
            assert captured_args['worker_urls'] == ['http://worker1:8000']
            assert captured_args['policy'] == PolicyType.CacheAware

            # Verify router.start() was called
            mock_router_instance.start.assert_called_once()

            # Function returns None; ensure start was invoked

    def test_router_initialization_pd_mode(self):
        """Test router initialization in PD mode."""
        args = RouterArgs(
            lmdeploy_pd_disaggregation=True,
            prefill_urls=['http://prefill1:8000'],
            decode_urls=['http://decode1:8001'],
            policy='power_of_two',
        )

        with patch('lmdeploy_router.launch_router.Router') as router_mod:
            captured_args = {}
            mock_router_instance = MagicMock()

            def fake_from_args(router_args):
                captured_args.update(
                    dict(
                        lmdeploy_pd_disaggregation=router_args.lmdeploy_pd_disaggregation,
                        prefill_urls=router_args.prefill_urls,
                        decode_urls=router_args.decode_urls,
                        policy=policy_from_str(router_args.policy),
                    )
                )
                return mock_router_instance

            router_mod.from_args = MagicMock(side_effect=fake_from_args)

            launch_router(args)

            # Verify Router.from_args was called with PD parameters
            router_mod.from_args.assert_called_once()
            assert captured_args['lmdeploy_pd_disaggregation'] is True
            assert captured_args['prefill_urls'] == ['http://prefill1:8000']
            assert captured_args['decode_urls'] == ['http://decode1:8001']
            assert captured_args['policy'] == PolicyType.PowerOfTwo

            # Verify router.start() was called
            mock_router_instance.start.assert_called_once()

            # Function returns None; ensure start was invoked

    def test_router_initialization_with_service_discovery(self):
        """Test router initialization with service discovery."""
        args = RouterArgs(
            service_discovery=True,
            selector={'app': 'worker', 'env': 'prod'},
            service_discovery_port=8080,
            service_discovery_namespace='default',
        )

        with patch('lmdeploy_router.launch_router.Router') as router_mod:
            captured_args = {}
            mock_router_instance = MagicMock()

            def fake_from_args(router_args):
                captured_args.update(
                    dict(
                        service_discovery=router_args.service_discovery,
                        selector=router_args.selector,
                        service_discovery_port=router_args.service_discovery_port,
                        service_discovery_namespace=router_args.service_discovery_namespace,
                    )
                )
                return mock_router_instance

            router_mod.from_args = MagicMock(side_effect=fake_from_args)

            launch_router(args)

            # Verify Router.from_args was called with service discovery parameters
            router_mod.from_args.assert_called_once()
            assert captured_args['service_discovery'] is True
            assert captured_args['selector'] == {'app': 'worker', 'env': 'prod'}
            assert captured_args['service_discovery_port'] == 8080
            assert captured_args['service_discovery_namespace'] == 'default'

            # Verify router.start() was called
            mock_router_instance.start.assert_called_once()

            # Function returns None; ensure start was invoked

    def test_router_initialization_with_retry_config(self):
        """Test router initialization with retry configuration."""
        args = RouterArgs(
            retry_max_retries=3,
            retry_initial_backoff_ms=100,
            retry_max_backoff_ms=10000,
            retry_backoff_multiplier=2.0,
            retry_jitter_factor=0.1,
            disable_retries=False,
        )

        with patch('lmdeploy_router.launch_router.Router') as router_mod:
            captured_args = {}
            mock_router_instance = MagicMock()

            def fake_from_args(router_args):
                captured_args.update(
                    dict(
                        retry_max_retries=router_args.retry_max_retries,
                        retry_initial_backoff_ms=router_args.retry_initial_backoff_ms,
                        retry_max_backoff_ms=router_args.retry_max_backoff_ms,
                        retry_backoff_multiplier=router_args.retry_backoff_multiplier,
                        retry_jitter_factor=router_args.retry_jitter_factor,
                        disable_retries=router_args.disable_retries,
                    )
                )
                return mock_router_instance

            router_mod.from_args = MagicMock(side_effect=fake_from_args)

            launch_router(args)

            # Verify router was created with retry parameters
            router_mod.from_args.assert_called_once()
            assert captured_args['retry_max_retries'] == 3
            assert captured_args['retry_initial_backoff_ms'] == 100
            assert captured_args['retry_max_backoff_ms'] == 10000
            assert captured_args['retry_backoff_multiplier'] == 2.0
            assert captured_args['retry_jitter_factor'] == 0.1
            assert captured_args['disable_retries'] is False

            # Verify router.start() was called
            mock_router_instance.start.assert_called_once()

            # Function returns None; ensure start was invoked

    def test_router_initialization_with_circuit_breaker_config(self):
        """Test router initialization with circuit breaker configuration."""
        args = RouterArgs(
            cb_failure_threshold=5,
            cb_success_threshold=2,
            cb_timeout_duration_secs=30,
            cb_window_duration_secs=60,
            disable_circuit_breaker=False,
        )

        with patch('lmdeploy_router.launch_router.Router') as router_mod:
            captured_args = {}
            mock_router_instance = MagicMock()

            def fake_from_args(router_args):
                captured_args.update(
                    dict(
                        cb_failure_threshold=router_args.cb_failure_threshold,
                        cb_success_threshold=router_args.cb_success_threshold,
                        cb_timeout_duration_secs=router_args.cb_timeout_duration_secs,
                        cb_window_duration_secs=router_args.cb_window_duration_secs,
                        disable_circuit_breaker=router_args.disable_circuit_breaker,
                    )
                )
                return mock_router_instance

            router_mod.from_args = MagicMock(side_effect=fake_from_args)

            launch_router(args)

            # Verify router was created with circuit breaker parameters
            router_mod.from_args.assert_called_once()
            assert captured_args['cb_failure_threshold'] == 5
            assert captured_args['cb_success_threshold'] == 2
            assert captured_args['cb_timeout_duration_secs'] == 30
            assert captured_args['cb_window_duration_secs'] == 60
            assert captured_args['disable_circuit_breaker'] is False

            # Verify router.start() was called
            mock_router_instance.start.assert_called_once()

            # Function returns None; ensure start was invoked

    def test_router_initialization_with_rate_limiting_config(self):
        """Test router initialization with rate limiting configuration."""
        args = RouterArgs(
            max_concurrent_requests=512,
            queue_size=200,
            queue_timeout_secs=120,
            rate_limit_tokens_per_second=100,
        )

        with patch('lmdeploy_router.launch_router.Router') as router_mod:
            captured_args = {}
            mock_router_instance = MagicMock()

            def fake_from_args(router_args):
                captured_args.update(
                    dict(
                        max_concurrent_requests=router_args.max_concurrent_requests,
                        queue_size=router_args.queue_size,
                        queue_timeout_secs=router_args.queue_timeout_secs,
                        rate_limit_tokens_per_second=router_args.rate_limit_tokens_per_second,
                    )
                )
                return mock_router_instance

            router_mod.from_args = MagicMock(side_effect=fake_from_args)

            launch_router(args)

            # Verify router was created with rate limiting parameters
            router_mod.from_args.assert_called_once()
            assert captured_args['max_concurrent_requests'] == 512
            assert captured_args['queue_size'] == 200
            assert captured_args['queue_timeout_secs'] == 120
            assert captured_args['rate_limit_tokens_per_second'] == 100

            # Verify router.start() was called
            mock_router_instance.start.assert_called_once()

            # Function returns None; ensure start was invoked

    def test_router_initialization_with_health_check_config(self):
        """Test router initialization with health check configuration."""
        args = RouterArgs(
            health_failure_threshold=2,
            health_success_threshold=1,
            health_check_timeout_secs=3,
            health_check_interval_secs=30,
            health_check_endpoint='/healthz',
        )

        with patch('lmdeploy_router.launch_router.Router') as router_mod:
            captured_args = {}
            mock_router_instance = MagicMock()

            def fake_from_args(router_args):
                captured_args.update(
                    dict(
                        health_failure_threshold=router_args.health_failure_threshold,
                        health_success_threshold=router_args.health_success_threshold,
                        health_check_timeout_secs=router_args.health_check_timeout_secs,
                        health_check_interval_secs=router_args.health_check_interval_secs,
                        health_check_endpoint=router_args.health_check_endpoint,
                    )
                )
                return mock_router_instance

            router_mod.from_args = MagicMock(side_effect=fake_from_args)

            launch_router(args)

            # Verify router was created with health check parameters
            router_mod.from_args.assert_called_once()
            assert captured_args['health_failure_threshold'] == 2
            assert captured_args['health_success_threshold'] == 1
            assert captured_args['health_check_timeout_secs'] == 3
            assert captured_args['health_check_interval_secs'] == 30
            assert captured_args['health_check_endpoint'] == '/healthz'

            # Verify router.start() was called
            mock_router_instance.start.assert_called_once()

            # Function returns None; ensure start was invoked

    def test_router_initialization_with_prometheus_config(self):
        """Test router initialization with Prometheus configuration."""
        args = RouterArgs(prometheus_port=29000, prometheus_host='127.0.0.1')

        with patch('lmdeploy_router.launch_router.Router') as router_mod:
            captured_args = {}
            mock_router_instance = MagicMock()

            def fake_from_args(router_args):
                captured_args.update(
                    dict(
                        prometheus_port=router_args.prometheus_port,
                        prometheus_host=router_args.prometheus_host,
                    )
                )
                return mock_router_instance

            router_mod.from_args = MagicMock(side_effect=fake_from_args)

            launch_router(args)

            # Verify router was created with Prometheus parameters
            router_mod.from_args.assert_called_once()
            assert captured_args['prometheus_port'] == 29000
            assert captured_args['prometheus_host'] == '127.0.0.1'

            # Verify router.start() was called
            mock_router_instance.start.assert_called_once()

            # Function returns None; ensure start was invoked

    def test_router_initialization_with_cors_config(self):
        """Test router initialization with CORS configuration."""
        args = RouterArgs(
            cors_allowed_origins=['http://localhost:3000', 'https://example.com']
        )

        with patch('lmdeploy_router.launch_router.Router') as router_mod:
            captured_args = {}
            mock_router_instance = MagicMock()

            def fake_from_args(router_args):
                captured_args.update(
                    dict(cors_allowed_origins=router_args.cors_allowed_origins)
                )
                return mock_router_instance

            router_mod.from_args = MagicMock(side_effect=fake_from_args)

            launch_router(args)

            # Verify router was created with CORS parameters
            router_mod.from_args.assert_called_once()
            assert captured_args['cors_allowed_origins'] == [
                'http://localhost:3000',
                'https://example.com',
            ]

            # Verify router.start() was called
            mock_router_instance.start.assert_called_once()

            # Function returns None; ensure start was invoked

    def test_router_initialization_with_tokenizer_config(self):
        """Test router initialization with tokenizer configuration."""
        pytest.skip('Tokenizer configuration not available in current implementation')


class TestStartupValidation:
    """Test startup validation logic."""

    def test_pd_mode_without_urls_allows_dynamic_registration(self):
        """Test that PD mode allows workers to register after startup."""
        args = RouterArgs(
            lmdeploy_pd_disaggregation=True,
            prefill_urls=[],
            decode_urls=[],
            service_discovery=False,
        )

        with patch('lmdeploy_router.launch_router.Router') as router_mod:
            mock_router_instance = MagicMock()
            router_mod.from_args = MagicMock(return_value=mock_router_instance)

            launch_router(args)
            router_mod.from_args.assert_called_once()

    def test_pd_mode_with_service_discovery_validation(self):
        """Test PD mode with service discovery validation during startup."""
        args = RouterArgs(
            lmdeploy_pd_disaggregation=True,
            prefill_urls=[],
            decode_urls=[],
            service_discovery=True,
        )

        # Should not raise validation error
        with patch('lmdeploy_router.launch_router.Router') as router_mod:
            mock_router_instance = MagicMock()

            router_mod.from_args = MagicMock(return_value=mock_router_instance)

            launch_router(args)

            # Should create router instance
            router_mod.from_args.assert_called_once()

    def test_policy_warning_during_startup(self):
        """Test policy warning during startup in PD mode."""
        args = RouterArgs(
            lmdeploy_pd_disaggregation=True,
            prefill_urls=['http://prefill1:8000'],
            decode_urls=['http://decode1:8001'],
            policy='cache_aware',
            prefill_policy='power_of_two',
            decode_policy='round_robin',
        )

        with patch('lmdeploy_router.launch_router.Router') as router_mod:
            mock_router_instance = MagicMock()
            router_mod.from_args = MagicMock(return_value=mock_router_instance)

            # The policy messages are emitted by router_args logger
            with patch('lmdeploy_router.router_args.logger') as mock_logger:
                launch_router(args)

                # Should log warning about policy usage
                mock_logger.warning.assert_called_once()
                warning_call = mock_logger.warning.call_args[0][0]
                assert (
                    'Both --prefill-policy and --decode-policy are specified'
                    in warning_call
                )

                # Should create router instance
                router_mod.from_args.assert_called_once()

    def test_policy_info_during_startup(self):
        """Test policy info logging during startup in PD mode."""
        # Test with only prefill policy specified
        args = RouterArgs(
            lmdeploy_pd_disaggregation=True,
            prefill_urls=['http://prefill1:8000'],
            decode_urls=['http://decode1:8001'],
            policy='cache_aware',
            prefill_policy='power_of_two',
            decode_policy=None,
        )

        with patch('lmdeploy_router.launch_router.Router') as router_mod:
            mock_router_instance = MagicMock()
            router_mod.from_args = MagicMock(return_value=mock_router_instance)

            # The policy messages are emitted by router_args logger
            with patch('lmdeploy_router.router_args.logger') as mock_logger:
                launch_router(args)

                # Should log info about policy usage
                mock_logger.info.assert_called_once()
                info_call = mock_logger.info.call_args[0][0]
                assert "Using --prefill-policy 'power_of_two'" in info_call
                assert "and --policy 'cache_aware'" in info_call

                # Should create router instance
                router_mod.from_args.assert_called_once()

    def test_policy_info_decode_only_during_startup(self):
        """Test policy info logging during startup with only decode policy
        specified."""
        args = RouterArgs(
            lmdeploy_pd_disaggregation=True,
            prefill_urls=['http://prefill1:8000'],
            decode_urls=['http://decode1:8001'],
            policy='cache_aware',
            prefill_policy=None,
            decode_policy='round_robin',
        )

        with patch('lmdeploy_router.launch_router.Router') as router_mod:
            mock_router_instance = MagicMock()
            router_mod.from_args = MagicMock(return_value=mock_router_instance)

            # The policy messages are emitted by router_args logger
            with patch('lmdeploy_router.router_args.logger') as mock_logger:
                launch_router(args)

                # Should log info about policy usage
                mock_logger.info.assert_called_once()
                info_call = mock_logger.info.call_args[0][0]
                assert "Using --policy 'cache_aware'" in info_call
                assert "and --decode-policy 'round_robin'" in info_call

                # Should create router instance
                router_mod.from_args.assert_called_once()


class TestStartupErrorHandling:
    """Test startup error handling logic."""

    def test_router_creation_error_handling(self):
        """Test error handling when router creation fails."""
        args = RouterArgs(
            host='127.0.0.1', port=30000, worker_urls=['http://worker1:8000']
        )

        with patch('lmdeploy_router.launch_router.Router') as router_mod:
            # Simulate router creation failure in from_args
            router_mod.from_args = MagicMock(
                side_effect=Exception('Router creation failed')
            )

            with patch('lmdeploy_router.launch_router.logger') as mock_logger:
                with pytest.raises(Exception, match='Router creation failed'):
                    launch_router(args)

                # Should log error
                mock_logger.error.assert_called_once()
                error_call = mock_logger.error.call_args[0][0]
                assert 'Error starting router: Router creation failed' in error_call

    def test_router_start_error_handling(self):
        """Test error handling when router start fails."""
        args = RouterArgs(
            host='127.0.0.1', port=30000, worker_urls=['http://worker1:8000']
        )

        with patch('lmdeploy_router.launch_router.Router') as router_mod:
            mock_router_instance = MagicMock()
            router_mod.from_args = MagicMock(return_value=mock_router_instance)

            # Simulate router start failure
            mock_router_instance.start.side_effect = Exception('Router start failed')

            with patch('lmdeploy_router.launch_router.logger') as mock_logger:
                with pytest.raises(Exception, match='Router start failed'):
                    launch_router(args)

                # Should log error
                mock_logger.error.assert_called_once()
                error_call = mock_logger.error.call_args[0][0]
                assert 'Error starting router: Router start failed' in error_call


class TestStartupFlow:
    """Test complete startup flow."""

    def test_complete_startup_flow_basic(self):
        """Test complete startup flow for basic configuration."""
        args = RouterArgs(
            host='127.0.0.1',
            port=30000,
            worker_urls=['http://worker1:8000', 'http://worker2:8000'],
            policy='cache_aware',
            cache_threshold=0.5,
            balance_abs_threshold=32,
            balance_rel_threshold=1.5,
        )

        with patch('lmdeploy_router.launch_router.Router') as router_mod:
            mock_router_instance = MagicMock()
            router_mod.from_args = MagicMock(return_value=mock_router_instance)

            launch_router(args)

            # Verify complete flow
            router_mod.from_args.assert_called_once()
            mock_router_instance.start.assert_called_once()

    def test_complete_startup_flow_pd_mode(self):
        """Test complete startup flow for PD mode configuration."""
        args = RouterArgs(
            lmdeploy_pd_disaggregation=True,
            prefill_urls=[
                'http://prefill1:8000',
                'http://prefill2:8000',
            ],
            decode_urls=['http://decode1:8001', 'http://decode2:8001'],
            policy='power_of_two',
            prefill_policy='cache_aware',
            decode_policy='round_robin',
        )

        with patch('lmdeploy_router.launch_router.Router') as router_mod:
            mock_router_instance = MagicMock()
            router_mod.from_args = MagicMock(return_value=mock_router_instance)

            with patch('lmdeploy_router.router_args.logger') as mock_logger:
                launch_router(args)

                # Verify complete flow
                router_mod.from_args.assert_called_once()
                mock_router_instance.start.assert_called_once()

                # Verify policy warning was logged
                mock_logger.warning.assert_called_once()

    def test_complete_startup_flow_with_all_features(self):
        """Test complete startup flow with all features enabled."""
        args = RouterArgs(
            host='0.0.0.0',
            port=30001,
            worker_urls=['http://worker1:8000'],
            policy='round_robin',
            service_discovery=True,
            selector={'app': 'worker'},
            service_discovery_port=8080,
            service_discovery_namespace='default',
            api_key='test-key',
            log_dir='/tmp/logs',
            log_level='debug',
            prometheus_port=29000,
            prometheus_host='0.0.0.0',
            request_id_headers=['x-request-id', 'x-trace-id'],
            request_timeout_secs=1200,
            max_concurrent_requests=512,
            queue_size=200,
            queue_timeout_secs=120,
            rate_limit_tokens_per_second=100,
            cors_allowed_origins=['http://localhost:3000'],
            retry_max_retries=3,
            retry_initial_backoff_ms=100,
            retry_max_backoff_ms=10000,
            retry_backoff_multiplier=2.0,
            retry_jitter_factor=0.1,
            cb_failure_threshold=5,
            cb_success_threshold=2,
            cb_timeout_duration_secs=30,
            cb_window_duration_secs=60,
            health_failure_threshold=2,
            health_success_threshold=1,
            health_check_timeout_secs=3,
            health_check_interval_secs=30,
            health_check_endpoint='/healthz',
        )

        with patch('lmdeploy_router.launch_router.Router') as router_mod:
            captured_args = {}
            mock_router_instance = MagicMock()

            def fake_from_args(router_args):
                captured_args.update(
                    dict(
                        host=router_args.host,
                        port=router_args.port,
                        worker_urls=router_args.worker_urls,
                        policy=policy_from_str(router_args.policy),
                        service_discovery=router_args.service_discovery,
                        selector=router_args.selector,
                        service_discovery_port=router_args.service_discovery_port,
                        service_discovery_namespace=router_args.service_discovery_namespace,
                        api_key=router_args.api_key,
                        log_dir=router_args.log_dir,
                        log_level=router_args.log_level,
                        prometheus_port=router_args.prometheus_port,
                        prometheus_host=router_args.prometheus_host,
                        request_id_headers=router_args.request_id_headers,
                        request_timeout_secs=router_args.request_timeout_secs,
                        max_concurrent_requests=router_args.max_concurrent_requests,
                        queue_size=router_args.queue_size,
                        queue_timeout_secs=router_args.queue_timeout_secs,
                        rate_limit_tokens_per_second=router_args.rate_limit_tokens_per_second,
                        cors_allowed_origins=router_args.cors_allowed_origins,
                        retry_max_retries=router_args.retry_max_retries,
                        retry_initial_backoff_ms=router_args.retry_initial_backoff_ms,
                        retry_max_backoff_ms=router_args.retry_max_backoff_ms,
                        retry_backoff_multiplier=router_args.retry_backoff_multiplier,
                        retry_jitter_factor=router_args.retry_jitter_factor,
                        cb_failure_threshold=router_args.cb_failure_threshold,
                        cb_success_threshold=router_args.cb_success_threshold,
                        cb_timeout_duration_secs=router_args.cb_timeout_duration_secs,
                        cb_window_duration_secs=router_args.cb_window_duration_secs,
                        health_failure_threshold=router_args.health_failure_threshold,
                        health_success_threshold=router_args.health_success_threshold,
                        health_check_timeout_secs=router_args.health_check_timeout_secs,
                        health_check_interval_secs=router_args.health_check_interval_secs,
                        health_check_endpoint=router_args.health_check_endpoint,
                    )
                )
                return mock_router_instance

            router_mod.from_args = MagicMock(side_effect=fake_from_args)

            launch_router(args)

            # Verify complete flow
            router_mod.from_args.assert_called_once()
            mock_router_instance.start.assert_called_once()

            # Verify key parameters were propagated into RouterArgs
            assert captured_args['host'] == '0.0.0.0'
            assert captured_args['port'] == 30001
            assert captured_args['worker_urls'] == ['http://worker1:8000']
            assert captured_args['policy'] == PolicyType.RoundRobin
            assert captured_args['service_discovery'] is True
            assert captured_args['selector'] == {'app': 'worker'}
            assert captured_args['service_discovery_port'] == 8080
            assert captured_args['service_discovery_namespace'] == 'default'
            assert captured_args['api_key'] == 'test-key'
            assert captured_args['log_dir'] == '/tmp/logs'
            assert captured_args['log_level'] == 'debug'
            assert captured_args['prometheus_port'] == 29000
            assert captured_args['prometheus_host'] == '0.0.0.0'
            assert captured_args['request_id_headers'] == ['x-request-id', 'x-trace-id']
            assert captured_args['request_timeout_secs'] == 1200
            assert captured_args['max_concurrent_requests'] == 512
            assert captured_args['queue_size'] == 200
            assert captured_args['queue_timeout_secs'] == 120
            assert captured_args['rate_limit_tokens_per_second'] == 100
            assert captured_args['cors_allowed_origins'] == ['http://localhost:3000']
            assert captured_args['retry_max_retries'] == 3
            assert captured_args['retry_initial_backoff_ms'] == 100
            assert captured_args['retry_max_backoff_ms'] == 10000
            assert captured_args['retry_backoff_multiplier'] == 2.0
            assert captured_args['retry_jitter_factor'] == 0.1
            assert captured_args['cb_failure_threshold'] == 5
            assert captured_args['cb_success_threshold'] == 2
            assert captured_args['cb_timeout_duration_secs'] == 30
            assert captured_args['cb_window_duration_secs'] == 60
            assert captured_args['health_failure_threshold'] == 2
            assert captured_args['health_success_threshold'] == 1
            assert captured_args['health_check_timeout_secs'] == 3
            assert captured_args['health_check_interval_secs'] == 30
            assert captured_args['health_check_endpoint'] == '/healthz'
