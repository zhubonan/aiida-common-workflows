# -*- coding: utf-8 -*-
# pylint: disable=unused-argument
"""Configuration and fixtures for unit test suite."""
import pytest

pytest_plugins = ['aiida.manage.tests.pytest_fixtures']  # pylint: disable=invalid-name


@pytest.fixture
def with_database(aiida_profile):
    """Alias for the `aiida_profile` fixture from `aiida-core`."""
    yield


@pytest.fixture
def clear_database(clear_database_before_test):
    """Alias for the `clear_database_before_test` fixture from `aiida-core`."""
    yield
