import pytest

from src.core.parser import CustomParser as CP


@pytest.mark.parametrize("args, q, k, data_dir, embeddings_path, index_path", [
    (['--query', 'test query'], 'test query', CP.DEFAULT_K, CP.DEFAULT_DATA_DIR, CP.DEFAULT_EMBEDDINGS_PATH, CP.DEFAULT_FAISS_INDEX_PATH),
    (['-q', 'test query'], 'test query', CP.DEFAULT_K, CP.DEFAULT_DATA_DIR, CP.DEFAULT_EMBEDDINGS_PATH, CP.DEFAULT_FAISS_INDEX_PATH),
    (['-q', 'test query', '--data-dir', 'd'], 'test query', CP.DEFAULT_K, 'd', CP.DEFAULT_EMBEDDINGS_PATH, CP.DEFAULT_FAISS_INDEX_PATH),
    (['-q', 'test query', '-k', '10'], 'test query', 10, CP.DEFAULT_DATA_DIR, CP.DEFAULT_EMBEDDINGS_PATH, CP.DEFAULT_FAISS_INDEX_PATH)
])
def test_parse_args_valid(monkeypatch, args, q, k, data_dir, embeddings_path, index_path):
    monkeypatch.setattr('sys.argv', ['script_name'] + args)
    parsed_args = CP.parse_args()
    assert parsed_args['query'] == q
    assert parsed_args['k'] == k
    assert parsed_args['data_dir'] == data_dir
    assert parsed_args['embeddings_path'] == embeddings_path
    assert parsed_args['faiss_index_path'] == index_path


@pytest.mark.parametrize("invalid_args", [
    (['-k', 10]),  # no query
    (['--query', 'test query', '-k', 'string k']),  # k as string
    (['--data-dir', 'd', '-k', 'string k'])  # no query and k as string
])
def test_parse_args_invalid(invalid_args, monkeypatch):
    with pytest.raises(SystemExit):
        monkeypatch.setattr('sys.argv', ['script_name'] + [invalid_args])
        CP.parse_args()  # This should raise SystemExit due to missing required argument
