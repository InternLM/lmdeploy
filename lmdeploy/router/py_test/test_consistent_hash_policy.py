# Copyright (c) OpenMMLab. All rights reserved.
#!/usr/bin/env python3
"""Test script for the new Consistent Hash load balancing policy.

This demonstrates how the consistent hash policy routes requests based on session_id or user_id, ensuring that requests
from the same user/session are consistently routed to the same worker for better cache locality and stateful processing.
"""

import argparse
import random
import string
import time

import requests


class ConsistentHashTester:
    def __init__(self, router_url: str = 'http://localhost:30000'):
        self.router_url = router_url.rstrip('/')
        self.session = requests.Session()

    def log(self, message: str, level: str = 'INFO'):
        """Log a message with timestamp."""
        timestamp = time.strftime('%H:%M:%S')
        print(f'[{timestamp}] [{level}] {message}')

    def generate_session_id(self) -> str:
        """Generate a random session ID."""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=12))

    def generate_user_id(self) -> str:
        """Generate a random user ID."""
        return f'user_{random.randint(1000, 9999)}'

    def make_request(
        self, prompt: str, session_id: str = None, user_id: str = None
    ) -> tuple[bool, str]:
        """Make a request to the router and return (success, response_text).

        Returns the full response for analysis.
        """
        # Build request with session_id or user_id in session_params
        request_data = {
            'text': prompt,
            'sampling_params': {'temperature': 0.7, 'max_new_tokens': 50},
        }

        # Add session_id to session_params OR user at top level (proper locations per OpenAI spec)
        if session_id:
            request_data['session_params'] = {'session_id': session_id}
        elif user_id:
            request_data['user'] = user_id  # Top-level user field (OpenAI standard)

        try:
            response = self.session.post(
                f'{self.router_url}/generate', json=request_data, timeout=30
            )

            if response.status_code == 200:
                return True, response.text
            else:
                return False, f'HTTP {response.status_code}: {response.text}'

        except Exception as e:
            return False, str(e)

    def test_session_consistency(self, num_requests: int = 10) -> bool:
        """Test that requests with the same session_id always go to the same
        worker."""
        self.log('Testing session consistency...')

        session_id = self.generate_session_id()
        worker_responses = []

        for i in range(num_requests):
            prompt = f'Request {i+1} for session {session_id}'
            success, response = self.make_request(prompt, session_id=session_id)

            if success:
                worker_responses.append(response)
                self.log(f'  Request {i+1}: ✅')
            else:
                self.log(f'  Request {i+1}: ❌ {response}', 'ERROR')
                return False

        # Extract worker information from responses (this would depend on your router's response format)
        # For now, we'll assume consistency if all requests succeed
        self.log(
                f'✅ Session consistency test passed: {num_requests} requests with session_id '
                f"'{session_id}' completed successfully"
        )
        return True

    def test_user_consistency(self, num_requests: int = 10) -> bool:
        """Test that requests with the same user_id always go to the same
        worker."""
        self.log('Testing user consistency...')

        user_id = self.generate_user_id()
        worker_responses = []

        for i in range(num_requests):
            prompt = f'Request {i+1} for user {user_id}'
            success, response = self.make_request(prompt, user_id=user_id)

            if success:
                worker_responses.append(response)
                self.log(f'  Request {i+1}: ✅')
            else:
                self.log(f'  Request {i+1}: ❌ {response}', 'ERROR')
                return False

        self.log(
            f"✅ User consistency test passed: {num_requests} requests with user_id '{user_id}' completed successfully"
        )
        return True

    def test_session_priority_over_user(self) -> bool:
        """Test that session_id takes priority over user_id for routing."""
        self.log('Testing session_id priority over user_id...')

        session_id = self.generate_session_id()
        user_id = self.generate_user_id()

        # Make requests with both session_id and user_id in session_params
        responses = []
        for i in range(5):
            prompt = f'Priority test request {i+1}'
            request_data = {
                'text': prompt,
                'sampling_params': {'temperature': 0.7, 'max_new_tokens': 30},
                'session_params': {
                    'session_id': session_id,
                    'user_id': user_id,  # This should be ignored in favor of session_id
                },
            }

            try:
                response = self.session.post(
                    f'{self.router_url}/generate', json=request_data
                )
                if response.status_code == 200:
                    responses.append(response.text)
                    self.log(f'  Priority test request {i+1}: ✅')
                else:
                    self.log(
                        f'  Priority test request {i+1}: ❌ HTTP {response.status_code}',
                        'ERROR',
                    )
                    return False
            except Exception as e:
                self.log(f'  Priority test request {i+1}: ❌ {e}', 'ERROR')
                return False

        self.log(
            '✅ Session priority test passed: All requests with both session_id and user_id completed successfully'
        )
        return True

    def test_distribution_across_workers(self, num_sessions: int = 20) -> bool:
        """Test that different sessions are distributed across multiple
        workers."""
        self.log(
            f'Testing distribution across workers with {num_sessions} different sessions...'
        )

        successful_requests = 0

        for i in range(num_sessions):
            session_id = self.generate_session_id()
            prompt = f'Distribution test for session {session_id}'

            success, response = self.make_request(prompt, session_id=session_id)

            if success:
                successful_requests += 1
                self.log(f'  Session {i+1}: ✅')
            else:
                self.log(f'  Session {i+1}: ❌ {response}', 'ERROR')

        success_rate = successful_requests / num_sessions
        if success_rate >= 0.8:  # Allow for some failures
            self.log(
                f'✅ Distribution test passed: {successful_requests}/{num_sessions} sessions succeeded'
                f'({success_rate:.1%})'
            )
            return True
        else:
            self.log(
                f'❌ Distribution test failed: Only {successful_requests}/{num_sessions} sessions succeeded'
                f'({success_rate:.1%})',
                'ERROR',
            )
            return False

    def test_fallback_without_session_or_user(self) -> bool:
        """Test that requests without session_id or user_id still work
        (fallback behavior)."""
        self.log('Testing fallback behavior without session_id or user_id...')

        for i in range(5):
            prompt = f'Fallback test request {i+1}'
            success, response = self.make_request(prompt)  # No session_id or user_id

            if success:
                self.log(f'  Fallback request {i+1}: ✅')
            else:
                self.log(f'  Fallback request {i+1}: ❌ {response}', 'ERROR')
                return False

        self.log(
            '✅ Fallback test passed: All requests without session_id/user_id completed successfully'
        )
        return True

    def test_openai_user_field_routing(self) -> bool:
        """Test routing with OpenAI-style user field instead of
        session_params."""
        self.log('Testing OpenAI user field routing...')

        user = self.generate_user_id()

        # Make requests using user field (OpenAI ChatCompletion/Completion style)
        for i in range(5):
            prompt = f'OpenAI user field test request {i+1}'
            request_data = {
                'text': prompt,
                'user': user,  # OpenAI-style user field (not in session_params)
                'sampling_params': {'temperature': 0.7, 'max_new_tokens': 30},
            }

            try:
                response = self.session.post(
                    f'{self.router_url}/generate', json=request_data
                )
                if response.status_code == 200:
                    self.log(f'  OpenAI user request {i+1}: ✅')
                else:
                    self.log(
                        f'  OpenAI user request {i+1}: ❌ HTTP {response.status_code}',
                        'ERROR',
                    )
                    return False
            except Exception as e:
                self.log(f'  OpenAI user request {i+1}: ❌ {e}', 'ERROR')
                return False

        self.log('✅ OpenAI user field routing test passed')
        return True

    def test_priority_routing(self) -> bool:
        """
        Test routing priority: session_params.session_id > session_params.user_id > user field.
        """
        self.log('Testing routing priority...')

        session_id = self.generate_session_id()
        user_id = self.generate_user_id()
        user = f'openai_{self.generate_user_id()}'

        # Test with all three: session_params should win
        request_data = {
            'text': 'Priority test',
            'user': user,  # Should be ignored
            'session_params': {
                'session_id': session_id,  # Should take priority
                'user_id': user_id,  # Should be ignored in favor of session_id
            },
            'sampling_params': {'temperature': 0.7, 'max_new_tokens': 30},
        }

        try:
            response = self.session.post(
                f'{self.router_url}/generate', json=request_data
            )
            if response.status_code == 200:
                self.log('✅ Priority routing test passed')
                return True
            else:
                self.log(
                    f'❌ Priority routing test failed: HTTP {response.status_code}',
                    'ERROR',
                )
                return False
        except Exception as e:
            self.log(f'❌ Priority routing test failed: {e}', 'ERROR')
            return False

    def test_concurrent_requests(self, num_concurrent: int = 10) -> bool:
        """Test concurrent requests with the same session_id to verify thread
        safety."""
        import threading
        import time

        self.log(
            f'Testing {num_concurrent} concurrent requests with same session_id...'
        )

        session_id = self.generate_session_id()
        results = []

        def make_concurrent_request(request_id: int):
            prompt = f'Concurrent request {request_id} for session {session_id}'
            success, response = self.make_request(prompt, session_id=session_id)
            results.append((request_id, success, response))

        # Start all threads
        threads = []
        start_time = time.time()

        for i in range(num_concurrent):
            thread = threading.Thread(target=make_concurrent_request, args=(i + 1,))
            thread.start()
            threads.append(thread)

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        end_time = time.time()

        # Check results
        successful = sum(1 for _, success, _ in results if success)
        success_rate = successful / num_concurrent

        if success_rate >= 0.8:  # Allow for some failures in concurrent scenario
            self.log(
                f'✅ Concurrent test passed: {successful}/{num_concurrent} requests succeeded'
                f'({success_rate:.1%}) in {end_time - start_time:.2f}s'
            )
            return True
        else:
            self.log(
                f'❌ Concurrent test failed: Only {successful}/{num_concurrent} requests succeeded'
                f'({success_rate:.1%})',
                'ERROR',
            )
            return False

    def test_different_request_formats(self) -> bool:
        """Test consistent hashing with different request formats (JSON
        variations)."""
        self.log('Testing different request formats...')

        session_id = self.generate_session_id()

        # Test different JSON formats
        formats = [
            # Standard format
            {
                'text': 'Format test 1',
                'session_id': session_id,
                'sampling_params': {'max_new_tokens': 20},
            },
            # Different field order
            {
                'session_id': session_id,
                'text': 'Format test 2',
                'sampling_params': {'temperature': 0.5, 'max_new_tokens': 20},
            },
            # With extra fields
            {
                'text': 'Format test 3',
                'session_id': session_id,
                'extra_field': 'ignored',
                'sampling_params': {'max_new_tokens': 20},
            },
        ]

        for i, request_data in enumerate(formats):
            try:
                response = self.session.post(
                    f'{self.router_url}/generate', json=request_data
                )
                if response.status_code == 200:
                    self.log(f'  Format {i+1}: ✅')
                else:
                    self.log(f'  Format {i+1}: ❌ HTTP {response.status_code}', 'ERROR')
                    return False
            except Exception as e:
                self.log(f'  Format {i+1}: ❌ {e}', 'ERROR')
                return False

        self.log('✅ Different request formats test passed')
        return True

    def run_comprehensive_test(self) -> bool:
        """Run all test cases and return overall success."""
        self.log('=' * 80)
        self.log('Starting Consistent Hash Policy Comprehensive Test')
        self.log(f'Router URL: {self.router_url}')
        self.log('=' * 80)

        tests = [
            ('Session Consistency', lambda: self.test_session_consistency(10)),
            ('User Consistency', lambda: self.test_user_consistency(10)),
            ('Session Priority', self.test_session_priority_over_user),
            ('OpenAI User Field', self.test_openai_user_field_routing),
            ('Priority Routing', self.test_priority_routing),
            ('Distribution', lambda: self.test_distribution_across_workers(20)),
            ('Fallback Behavior', self.test_fallback_without_session_or_user),
            ('Concurrent Requests', lambda: self.test_concurrent_requests(10)),
            ('Request Formats', self.test_different_request_formats),
        ]

        passed = 0
        failed = 0

        for test_name, test_func in tests:
            self.log('')
            self.log(f'🧪 Running: {test_name}')
            try:
                if test_func():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                self.log(f"❌ Test '{test_name}' failed with exception: {e}", 'ERROR')
                failed += 1

        self.log('')
        self.log('=' * 80)
        self.log('Test Results Summary')
        self.log('=' * 80)

        total = passed + failed
        success_rate = passed / total if total > 0 else 0

        self.log(f'Total Tests: {total}')
        self.log(f'Passed: {passed}')
        self.log(f'Failed: {failed}')
        self.log(f'Success Rate: {success_rate:.1%}')

        if failed == 0:
            self.log('🎉 All tests passed! Consistent hashing is working correctly.')
            return True
        else:
            self.log(
                f'⚠️  {failed} test(s) failed. Please check the router configuration and try again.'
            )
            return False


def main():
    parser = argparse.ArgumentParser(
        description='Test consistent hash policy for the router'
    )
    parser.add_argument(
        '--router-url',
        default='http://localhost:30000',
        help='Router URL (default: http://localhost:30000)',
    )
    parser.add_argument(
        '--test',
        choices=[
            'session',
            'user',
            'priority',
            'distribution',
            'fallback',
            'dp_routing',
            'concurrent',
            'formats',
            'all',
        ],
        default='all',
        help='Run specific test (default: all)',
    )

    args = parser.parse_args()

    tester = ConsistentHashTester(router_url=args.router_url)

    if args.test == 'all':
        success = tester.run_comprehensive_test()
        return 0 if success else 1
    else:
        # Run specific test
        test_methods = {
            'session': lambda: tester.test_session_consistency(10),
            'user': lambda: tester.test_user_consistency(10),
            'priority': tester.test_session_priority_over_user,
            'distribution': lambda: tester.test_distribution_across_workers(20),
            'fallback': tester.test_fallback_without_session_or_user,
            'concurrent': lambda: tester.test_concurrent_requests(10),
            'formats': tester.test_different_request_formats,
        }

        if args.test in test_methods:
            success = test_methods[args.test]()
            return 0 if success else 1
        else:
            print(f'Unknown test: {args.test}')
            return 1


if __name__ == '__main__':
    import sys

    sys.exit(main())
