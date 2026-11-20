import io
from PIL import Image
from server.forensics import inspect_image


def test_descriptive_inspection_no_authenticity_claim():
    image = Image.new('RGB', (64, 48), (90, 125, 160))
    data = inspect_image(image, source_format='PNG', byte_count=120, exif_present=False)
    assert data['file']['width'] == 64
    assert data['file']['height'] == 48
    assert data['file']['format'] == 'PNG'
    assert data['metrics']['brightness'] > 0
    assert data['metrics']['grayscale_entropy_bits'] == 0
    assert data['ela_data_url'].startswith('data:image/png;base64,')
    assert 'NOT evidence' in data['interpretation']


def test_odd_image_shapes():
    for size in [(1, 1), (1, 23), (257, 10)]:
        assert inspect_image(Image.new('RGB', size, 'white'),
                             source_format='JPEG', byte_count=100, exif_present=False)['file']['width'] == size[0]
