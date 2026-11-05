import json
from unittest.mock import Mock
import pytest
from tasks.copper_inversion.core.execution import digest
from tasks.copper_inversion.experiments.v2.freeze_service_selection import freeze_from_service


def test_trusted_selection_uses_only_sealed_private_receipts(tmp_path,monkeypatch):
    identity=dict(configuration=dict(manifest='/private/train',seconds=10,image='sha256:'+'a'*64))
    receipts={'b'*64:dict(source_path='/private/source',metrics={'mean_iou':.5})}
    seal=dict(contract_digest=digest(identity),selected_receipts=receipts,source_sha256s=['b'*64])
    (tmp_path/'identity.json').write_text(json.dumps(identity))
    (tmp_path/'training-sealed.json').write_text(json.dumps(seal))
    monkeypatch.setattr('tasks.copper_inversion.experiments.v2.freeze_service_selection.eligible',lambda *args:receipts)
    frozen=Mock(return_value={'status':'test'})
    monkeypatch.setattr('tasks.copper_inversion.experiments.v2.freeze_service_selection.freeze',frozen)
    freeze_from_service(tmp_path,tmp_path/'selection','/private/val','/private/id','/private/ood')
    assert frozen.call_args.args[1]==['/private/source']
    assert frozen.call_args.kwargs['training_run']==tmp_path
    receipts['b'*64]['metrics']['mean_iou']=.9
    with pytest.raises(ValueError,match='receipts changed'):
        freeze_from_service(tmp_path,tmp_path/'selection','/private/val','/private/id','/private/ood')
    assert frozen.call_count==1
