import pytest
from flask import Flask

from app import create_app


@pytest.fixture()
def app():
    app = create_app()
    yield app


def test_create_app(app):
    """
    Test the app creation.

    This test checks if the returned object from create_app is an instance of Flask and
    if the debug mode is set to False.

    :param app: The Flask application instance.
    """
    assert isinstance(app, Flask)
    assert app.debug is False
