import csv
from pathlib import Path
import pytest
from PIL import Image
from ml.data import read_manifest
from ml.metrics import evaluate, find_threshold, source_report


def test_threshold_only_uses_validation_labels():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.49, 0.55, 0.9]
    t = find_threshold(labels, scores, max_fpr=0)
    assert t > 0.49
    result = evaluate(labels, scores, t)
    assert result['false_positive_rate'] == 0
    assert result['confusion_matrix']['fp'] == 0


def test_one_class_group_does_not_invent_auc():
    result = source_report([1, 1], [0.8, 0.9], ['diffusion', 'diffusion'], 0.5)
    assert result['diffusion']['roc_auc'] is None


def test_cross_split_identity_leakage(tmp_path):
    for name in ('a', 'b'):
        Image.new('RGB', (8, 8)).save(tmp_path / f'{name}.png')
    manifest = tmp_path / 'manifest.csv'
    with manifest.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['path', 'label', 'split', 'source', 'group'])
        w.writeheader()
        w.writerows([{'path':'a.png','label':'real','split':'train','source':'camera','group':'person1'},
                     {'path':'b.png','label':'fake','split':'test','source':'generator1','group':'person1'}])
    with pytest.raises(ValueError, match='leakage'):
        read_manifest(manifest)


def test_jpeg_transform_and_untrained_model_forward():
    torch = pytest.importorskip('torch')
    from ml.model import RandomJPEG, build_model, transform
    image = Image.new('RGB', (256, 256), (50, 100, 150))
    assert RandomJPEG(probability=1)(image).size == image.size
    batch = transform(False)(image).unsqueeze(0)
    model = build_model(pretrained=False)
    with torch.inference_mode():
        output = model(batch)
    assert output.shape == (1, 1)


def test_checkpoint_predictor_v2(tmp_path):
    torch = pytest.importorskip('torch')
    from ml.model import build_model
    from server.core import Predictor
    checkpoint = tmp_path / 'v2.pth'
    torch.save({'architecture':'resnet18-v2','model_state_dict':build_model(pretrained=False).state_dict(),
                'threshold':.6,'class_to_index':{'real':0,'fake':1}}, checkpoint)
    outcome = Predictor(checkpoint).predict(Image.new('RGB', (256, 256), (60, 70, 80)))
    assert outcome['model'] == 'resnet18-v2'
    assert 0 <= outcome['fake_score'] <= 1
    assert outcome['decision_threshold'] == .6
    assert outcome['saliency_data_url'].startswith('data:image/png;base64,')
