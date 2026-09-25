import nbformat

from revllm.notebooks import build, colab_url

URL = "https://github.com/someone/reversible-llm.git"


def test_build_writes_four_valid_notebooks(tmp_path):
    paths = build(URL, tmp_path)
    assert sorted(p.name for p in paths) == ["01_baseline.ipynb", "02_reversible_same_batch.ipynb",
                                             "03_reversible_max_batch.ipynb", "04_analysis.ipynb"]
    for p in paths:
        nbformat.validate(nbformat.read(p, as_version=4))
    src = "".join(c.source for c in nbformat.read(tmp_path / "01_baseline.ipynb", as_version=4).cells)
    assert URL in src and "TrainConfig(" in src


def test_colab_url():
    assert colab_url(URL, "x.ipynb") == \
        "https://colab.research.google.com/github/someone/reversible-llm/blob/main/notebooks/x.ipynb"


def test_project_in_repo_subfolder(tmp_path):
    url = "https://github.com/someone/assignments.git"
    assert colab_url(url, "x.ipynb", subdir="rev-13") == \
        "https://colab.research.google.com/github/someone/assignments/blob/main/rev-13/notebooks/x.ipynb"
    build(url, tmp_path, subdir="rev-13")
    src = "".join(c.source for c in nbformat.read(tmp_path / "01_baseline.ipynb", as_version=4).cells)
    assert "/content/assignments/rev-13" in src
    assert "/content/reversible-llm" not in src
