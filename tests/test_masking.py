import glob
import io
import os

import imagehash
import pytest
from PIL import Image
from pdf2image import convert_from_bytes, convert_from_path

from app import create_app

url_prefix = '/api/v1/mask'


@pytest.fixture()
def app():
    app = create_app()
    yield app


@pytest.fixture
def client(app):
    return app.test_client()


def test_mask_pdf_no_file(client):
    """
    Test the mask_pdf endpoint with no file in the request.

    This test checks if the correct HTTP status code and response body are returned when
    the mask_pdf endpoint is called without a file in the request.

    :param client: The test client used to send requests to the application.
    """
    response = client.post(url_prefix + '/pdf')
    assert response.status_code == 400
    assert response.json == {
        "success": False,
        "message": "No file uploaded.",
        "code": 400
    }


def test_mask_pdf_wrong_file_type(client):
    """
    Test the mask_pdf endpoint with an unsupported file type in the request.

    This test checks if the correct HTTP status code and response body are returned when
    the mask_pdf endpoint is called with a file of an unsupported type.

    :param client: The test client used to send requests to the application.
    """
    data = {'file': (io.BytesIO(b"file content"), 'test.txt')}
    response = client.post(url_prefix + '/pdf', data=data, content_type='multipart/form-data')
    assert response.status_code == 415
    assert response.json == {
        "success": False,
        "message": "Unsupported file type. Only PDF files are allowed.",
        "code": 415
    }


def test_mask_image_no_file(client):
    """
    Test the mask_image endpoint with no file in the request.

    This test checks if the correct HTTP status code and response body are returned when
    the mask_image endpoint is called without a file in the request.

    :param client (FlaskClient): The test client used to send requests to the application.
    """
    response = client.post(url_prefix + '/image')
    assert response.status_code == 400
    assert response.json == {
        "success": False,
        "message": "No file uploaded.",
        "code": 400
    }


def test_mask_image_wrong_file_type(client):
    """
    Test the mask_image endpoint with an unsupported file type in the request.

    This test checks if the correct HTTP status code and response body are returned when
    the mask_image endpoint is called with a file of an unsupported type.

    :param client: The test client used to send requests to the application.
    """
    data = {'file': (io.BytesIO(b"file content"), 'test.txt')}
    response = client.post(url_prefix + '/image', data=data, content_type='multipart/form-data')
    assert response.status_code == 415
    assert response.json == {
        "success": False,
        "message": "Unsupported file type. Only JPEG, PNG, and GIF images are allowed.",
        "code": 415
    }


def test_mask_image_with_examples(client):
    """
    Test the mask_image endpoint with all jpg image files in the data directory.

    This test checks if the correct HTTP status code and response body are returned when
    the mask_image endpoint is called with each jpg image file in the data directory. It also checks if the
    response body contains an image by checking the content type of the response.

    The test then compares the returned image with an existing image by comparing
    their perceptual hashes.

    :param client: The test client used to send requests to the application.
    """
    for image_path in glob.glob("data/images/*.jpg"):
        if "_expected.jpg" in image_path:
            continue
        with open(image_path, "rb") as image_file:
            data = {'file': (image_file, os.path.basename(image_path))}
            response = client.post('/api/v1/mask/image', data=data, content_type='multipart/form-data')

        assert response.status_code == 200
        assert response.headers['Content-Type'] in ['image/jpeg', 'image/png', 'image/gif']

        expected_image_path = image_path.replace('.jpg', '_expected.jpg')

        if not os.path.exists(expected_image_path):
            # use this for test setup if new images are added
            response_image = Image.open(io.BytesIO(response.data))
            response_image.save(expected_image_path)

        response_image = io.BytesIO(response.data)
        hash0 = imagehash.average_hash(Image.open(response_image))
        hash1 = imagehash.average_hash(Image.open(expected_image_path))
        similarity = 1 - (hash0 - hash1) / len(hash0.hash) ** 2
        assert similarity > 0.95


def test_mask_pdf_from_different_tools(client):
    """
    Test the mask_pdf endpoint with all pdf files in the data directory.

    This test checks if the correct HTTP status code and response body are returned when
    the mask_pdf endpoint is called with each pdf file in the data directory.

    :param client: The test client used to send requests to the application.
    """
    for pdf_path in glob.glob("data/*.pdf"):
        if "_expected.pdf" in pdf_path:
            continue
        with open(pdf_path, "rb") as pdf_file:
            data = {'file': (pdf_file, os.path.basename(pdf_path))}
            response = client.post('/api/v1/mask/pdf', data=data, content_type='multipart/form-data')

        assert response.status_code == 200
        assert response.headers['Content-Type'] == 'application/pdf'

        expected_pdf_path = pdf_path.replace('.pdf', '_expected.pdf')

        if not os.path.exists(expected_pdf_path):
            # use this for test setup if new pdfs are added
            with open(expected_pdf_path, 'wb') as f:
                f.write(response.data)

        similarity_scores = compare_pdfs(response.data, expected_pdf_path)
        assert all(score > 0.95 for score in similarity_scores)


def test_mask_pdf_with_examples(client):
    """
    Test the mask_pdf endpoint with all pdf files in the data directory.

    This test checks if the correct HTTP status code and response body are returned when
    the mask_pdf endpoint is called with each pdf file in the data directory.

    :param client: The test client used to send requests to the application.
    """
    for pdf_path in glob.glob("data/pdfs/*.pdf"):
        if "_expected.pdf" in pdf_path:
            continue
        with open(pdf_path, "rb") as pdf_file:
            data = {'file': (pdf_file, os.path.basename(pdf_path))}
            response = client.post('/api/v1/mask/pdf', data=data, content_type='multipart/form-data')

        assert response.status_code == 200
        assert response.headers['Content-Type'] == 'application/pdf'

        expected_pdf_path = pdf_path.replace('.pdf', '_expected.pdf')

        if not os.path.exists(expected_pdf_path):
            # use this for test setup if new pdfs are added
            with open(expected_pdf_path, 'wb') as f:
                f.write(response.data)

        similarity_scores = compare_pdfs(response.data, expected_pdf_path)
        assert all(score > 0.95 for score in similarity_scores)


def test_mask_pdf_with_full_image_examples(client):
    """
    Test the mask_pdf endpoint with all pdf files in the data directory.

    This test checks if the correct HTTP status code and response body are returned when
    the mask_pdf endpoint is called with each pdf file in the data directory.

    :param client: The test client used to send requests to the application.
    """
    for pdf_path in glob.glob("data/pdfs_full_image/*.pdf"):
        if "_expected.pdf" in pdf_path:
            continue
        with open(pdf_path, "rb") as pdf_file:
            data = {'file': (pdf_file, os.path.basename(pdf_path))}
            response = client.post('/api/v1/mask/pdf', data=data, content_type='multipart/form-data')

        assert response.status_code == 200
        assert response.headers['Content-Type'] == 'application/pdf'

        expected_pdf_path = pdf_path.replace('.pdf', '_expected.pdf')

        if not os.path.exists(expected_pdf_path):
            # use this for test setup if new pdfs are added
            with open(expected_pdf_path, 'wb') as f:
                f.write(response.data)

        similarity_scores = compare_pdfs(response.data, expected_pdf_path)
        assert all(score > 0.95 for score in similarity_scores)


def compare_pdfs(pdf1_content, pdf2_path):
    # Convert PDFs to images
    images1 = convert_from_bytes(pdf1_content)
    images2 = convert_from_path(pdf2_path)

    # Compare images using image hashing
    similarity_scores = []
    num_pages = min(len(images1), len(images2))
    for i in range(num_pages):
        # Calculate image hashes
        hash0 = imagehash.average_hash(images1[i])
        hash1 = imagehash.average_hash(images2[i])

        # Compare the hashes
        similarity_score = 1 - (hash0 - hash1) / len(hash0.hash) ** 2
        similarity_scores.append(similarity_score)

    return similarity_scores
