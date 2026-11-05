import json
import shutil
from pathlib import Path
import pytest
from tasks.copper_inversion.v2.public_catalog import SCHEMA, file_hash, load_catalog, catalog_digest, observation_digest
from tasks.copper_inversion.v2.data import save_problem
from tasks.copper_inversion.v2.fixtures import fixtures
from tasks.copper_inversion.experiments.v2.publish_training import publish


def catalog(root):
    problem = next(iter(fixtures()))
    save_problem(problem, root/'cases/0000')
    data = dict(schema=SCHEMA, stage='development', split='train', cases=[dict(
        case_id=problem['case_id'], directory='cases/0000',
        sha256={name:file_hash(root/'cases/0000'/name) for name in ('problem.json','problem.npz')})])
    path=root/'catalog.json';path.write_text(json.dumps(data))
    return path,data


@pytest.mark.parametrize('mutation',['truth','parent','escape','digest','duplicate','split'])
def test_reject_private_fields_and_changed_catalog(tmp_path,mutation):
    path,data=catalog(tmp_path)
    assert load_catalog(path)==data
    row=data['cases'][0]
    if mutation=='truth':row['truth']='secret'
    if mutation=='parent':row['parent_id']='private'
    if mutation=='escape':row['directory']='../elsewhere'
    if mutation=='digest':row['sha256']['problem.npz']='0'*64
    if mutation=='duplicate':data['cases'].append(dict(row))
    if mutation=='split':data['split']='test_id'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):load_catalog(path)


def test_symlink_file_rejected(tmp_path):
    path,data=catalog(tmp_path)
    file=tmp_path/'cases/0000/problem.json'
    target=tmp_path/'copy.json';file.rename(target);file.symlink_to(target)
    with pytest.raises(ValueError,match='path escapes'):load_catalog(path)


def test_real_publication_works_without_private_tree(tmp_path):
    manifest=Path('suites/integration-train/manifest.json')
    if not manifest.exists():pytest.skip('remote integration suite required')
    published=publish(manifest,tmp_path/'release')
    original=load_catalog(published)
    from tasks.copper_inversion.v2.suite import load_manifest
    from tasks.copper_inversion.v2.data import read_problem
    private=load_manifest(manifest)
    assert catalog_digest(published)==observation_digest(private['stage'],
        (read_problem(manifest.parent/row['public']) for row in private['cases']))
    relocated=tmp_path/'relocated';shutil.copytree(published.parent,relocated)
    shutil.rmtree(published.parent)
    assert load_catalog(relocated/'catalog.json')==original
    assert not list(relocated.rglob('truth.npy'))
    assert len(list(relocated.rglob('*.*')))==5
    with pytest.raises(FileExistsError):publish(manifest,relocated)
