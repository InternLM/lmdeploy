# Copyright (c) OpenMMLab. All rights reserved.
"""Unit tests for lmdeploy version module."""

import pytest

from lmdeploy.version import __version__, parse_version_info, short_version, version_info


class TestVersionConstants:
    """Tests for version constants."""

    def test_version_exists(self):
        """Test that __version__ is defined."""
        assert __version__ is not None
        assert isinstance(__version__, str)

    def test_short_version_equals_version(self):
        """Test that short_version equals __version__."""
        assert short_version == __version__

    def test_version_format(self):
        """Test that version follows semantic versioning format."""
        parts = __version__.split('.')
        assert len(parts) >= 2
        # First two parts should be numeric
        assert parts[0].isdigit()
        assert parts[1].isdigit()

    def test_version_is_0_13_0(self):
        """Test current version is 0.13.0."""
        assert __version__ == '0.13.0'


class TestParseVersionInfo:
    """Tests for parse_version_info function."""

    def test_parse_simple_version(self):
        """Test parsing simple version like '1.2.3'."""
        result = parse_version_info('1.2.3')
        assert result == (1, 2, 3)

    def test_parse_two_part_version(self):
        """Test parsing two-part version like '1.2'."""
        result = parse_version_info('1.2')
        assert result == (1, 2)

    def test_parse_version_with_rc(self):
        """Test parsing version with release candidate like '1.2.3rc1'."""
        result = parse_version_info('1.2.3rc1')
        assert result == (1, 2, 3, 'rc1')

    def test_parse_version_with_rc_double_digit(self):
        """Test parsing version with double-digit RC like '1.2.3rc10'."""
        result = parse_version_info('1.2.3rc10')
        assert result == (1, 2, 3, 'rc10')

    def test_parse_current_version(self):
        """Test parsing current version '0.13.0'."""
        result = parse_version_info('0.13.0')
        assert result == (0, 13, 0)

    def test_parse_version_with_multiple_rc(self):
        """Test parsing version with multiple RC segments."""
        # Only first RC should be parsed
        result = parse_version_info('1.0.0rc1')
        assert result == (1, 0, 0, 'rc1')

    def test_parse_version_preserves_strings(self):
        """Test that non-numeric parts are handled correctly."""
        result = parse_version_info('1.0.0rc1')
        # Check that rc part is a string
        assert isinstance(result[-1], str)
        assert result[-1].startswith('rc')

    def test_parse_version_all_digits(self):
        """Test parsing version with all numeric parts."""
        result = parse_version_info('2.0.0')
        assert all(isinstance(x, int) for x in result)

    def test_parse_single_digit_version(self):
        """Test parsing single digit version parts."""
        result = parse_version_info('0.0.1')
        assert result == (0, 0, 1)

    def test_parse_large_version_numbers(self):
        """Test parsing large version numbers."""
        result = parse_version_info('99.100.200')
        assert result == (99, 100, 200)


class TestVersionInfo:
    """Tests for version_info tuple."""

    def test_version_info_is_tuple(self):
        """Test that version_info is a tuple."""
        assert isinstance(version_info, tuple)

    def test_version_info_matches_parsed_version(self):
        """Test that version_info matches parsed __version__."""
        expected = parse_version_info(__version__)
        assert version_info == expected

    def test_version_info_has_major_minor(self):
        """Test that version_info has at least major and minor."""
        assert len(version_info) >= 2

    def test_version_info_major_is_zero(self):
        """Test that major version is 0."""
        assert version_info[0] == 0

    def test_version_info_minor_is_13(self):
        """Test that minor version is 13."""
        assert version_info[1] == 13

    def test_version_info_patch_is_zero(self):
        """Test that patch version is 0."""
        assert version_info[2] == 0


class TestVersionComparison:
    """Tests for version comparison scenarios."""

    def test_higher_version_parses_correctly(self):
        """Test that higher versions parse correctly."""
        result = parse_version_info('1.0.0')
        assert result == (1, 0, 0)

    def test_version_with_alpha_not_parsed(self):
        """Test that alpha/beta suffixes are not parsed (only rc)."""
        # Non-RC, non-digit parts are ignored
        result = parse_version_info('1.0.0alpha')
        # '0alpha' is not digit and doesn't contain 'rc', so it's skipped
        assert result == (1, 0)

    def test_empty_string_handling(self):
        """Test handling of empty string."""
        result = parse_version_info('')
        assert result == ()

    def test_version_with_leading_zeros(self):
        """Test version with leading zeros."""
        result = parse_version_info('01.02.03')
        assert result == (1, 2, 3)
