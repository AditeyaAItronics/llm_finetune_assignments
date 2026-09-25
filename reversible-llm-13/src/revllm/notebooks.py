"""Generates the Colab notebooks from source so they stay reviewable.
Usage: python -m revllm.notebooks https://github.com/<you>/<repo>.git [subfolder-of-repo]"""
import sys
from pathlib import Path

import nbformat as nbf


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


def colab_url(repo_url: str, name: str, subdir: str = "") -> str:
    slug = repo_url.removeprefix("https://github.com/").removesuffix(".git")
    prefix = f"{subdir}/" if subdir else ""
    return f"https://colab.research.google.com/github/{slug}/blob/main/{prefix}notebooks/{name}"


def _setup(repo_url, subdir=""):
    clone_dir = "/content/" + repo_url.rstrip("/").removesuffix(".git").rsplit("/", 1)[-1]
    project = f"{clone_dir}/{subdir}" if subdir else clone_dir
    return [
        md("## Setup\nRuntime → Change runtime type → **T4 GPU**. Results are written to Google Drive, "
           "so a disconnect does not lose finished runs."),
        code("!nvidia-smi"),
        code("""
from google.colab import drive
drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive/reversible-llm'
import os
os.makedirs(f'{DRIVE}/data', exist_ok=True)
os.makedirs(f'{DRIVE}/results', exist_ok=True)
"""),
        code(f"""
import os, sys
if not os.path.exists('{clone_dir}'):
    !git clone -q {repo_url} {clone_dir}
%cd {project}
!git pull -q
!pip install -q tiktoken datasets
sys.path.insert(0, '{project}/src')
"""),
        code("""
from revllm.data import prepare_tinystories
prepare_tinystories(f'{DRIVE}/data')  # tokenizes once (~55M tokens), cached on Drive
!mkdir -p /content/data && cp -n {DRIVE}/data/*.bin /content/data/
"""),
        code("""
import pytest
# includes the GPU memory test that is skipped on CPU
assert pytest.main(['-q', 'tests']) == 0, 'tests failed: fix before training'
"""),
    ]


def nb_baseline(repo_url, subdir=""):
    return [
        md("# 01: Baseline GPT (19.3M params, 50M tokens, fixed batch size)\n"
           "Standard pre-LN transformer. We first measure the largest batch that fits (for reporting), "
           "then train at a fixed **B=64** (32,768 tokens/step), a size that trains well within the 50M-token budget."),
        *_setup(repo_url, subdir),
        code("""
import json
from revllm.config import ModelConfig, TrainConfig
from revllm.batch_search import find_max_batch
from revllm.train import train

mc = ModelConfig(arch='baseline')
search = find_max_batch(mc)
json.dump(search, open(f'{DRIVE}/batch_search_baseline.json', 'w'))
search
"""),
        code("""
B = 64 if search['max_batch'] >= 64 else search['recommended']
res = train(mc, TrainConfig(run_name='01_baseline', batch_size=B,
                            data_dir='/content/data', out_dir=f'{DRIVE}/results'))
{k: res[k] for k in ['params', 'batch_size', 'tokens_seen', 'final_train_loss', 'final_val_loss',
                     'tokens_per_sec', 'peak_mem_allocated_gb', 'wall_time_s']}
"""),
    ]


def nb_same_batch(repo_url, subdir=""):
    return [
        md("# 02: Reversible transformer at the same batch size\n"
           "**Step 1: variant pilots.** Each variant (plus a baseline reference) is trained for 5M tokens at the baseline batch size:\n"
           "- `euler`: two-stream additive coupling (RevNet/Reformer, a symplectic-Euler step), h=1\n"
           "- `midpoint`: leapfrog/explicit-midpoint step `(a, b) → (b, a + 2h·F(b))`, h=0.5 and h=0.25\n\n"
           "**Step 2:** full 50M-token run of the variant with the best pilot validation loss."),
        *_setup(repo_url, subdir),
        code("""
import json
from revllm.config import ModelConfig, TrainConfig
from revllm.train import train

B = json.load(open(f'{DRIVE}/results/01_baseline.json'))['batch_size']
PILOTS = [('baseline', None), ('euler', 1.0), ('midpoint', 0.5), ('midpoint', 0.25)]
pilots = {}
for arch, h in PILOTS:
    mc = ModelConfig(arch=arch, h=h)
    name = f'02_pilot_{arch}_h{mc.h}'
    pilots[name] = train(mc, TrainConfig(run_name=name, total_tokens=5_000_000, batch_size=B,
                                         eval_every_tokens=1_000_000,
                                         data_dir='/content/data', out_dir=f'{DRIVE}/results'))
KEYS = ['diverged', 'final_val_loss', 'tokens_per_sec', 'peak_mem_allocated_gb', 'recon_error']
{n: {k: r[k] for k in KEYS} for n, r in pilots.items()}
"""),
        code("""
ok = [r for r in pilots.values() if r['arch'] != 'baseline' and not r['diverged']]
if not ok:
    raise RuntimeError('all reversible pilots diverged - rerun 02 with lower lr or h')
winner = min(ok, key=lambda r: r['final_val_loss'])
json.dump({'arch': winner['arch'], 'h': winner['h']}, open(f'{DRIVE}/winner.json', 'w'))
print('Winner:', winner['arch'], 'h =', winner['h'])
"""),
        code("""
mc = ModelConfig(arch=winner['arch'], h=winner['h'])
res = train(mc, TrainConfig(run_name=f"02_reversible_{winner['arch']}_same_batch", batch_size=B,
                            data_dir='/content/data', out_dir=f'{DRIVE}/results'))
{k: res[k] for k in ['batch_size', 'tokens_seen', 'final_train_loss', 'final_val_loss',
                     'tokens_per_sec', 'peak_mem_allocated_gb', 'wall_time_s', 'recon_error']}
"""),
    ]


def nb_max_batch(repo_url, subdir=""):
    return [
        md("# 03: Reversible transformer at the maximum batch size\n"
           "Search for the largest batch the winning variant can run, back off 10% for headroom, and train "
           "on the same 50M tokens. The learning rate is scaled by √(B/B_base) and capped at 3e-3."),
        *_setup(repo_url, subdir),
        code("""
import json, math
from revllm.config import ModelConfig, TrainConfig
from revllm.batch_search import find_max_batch
from revllm.train import train

w = json.load(open(f'{DRIVE}/winner.json'))
B_base = json.load(open(f'{DRIVE}/results/01_baseline.json'))['batch_size']
mc = ModelConfig(arch=w['arch'], h=w['h'])
search = find_max_batch(mc, start=B_base)
json.dump(search, open(f"{DRIVE}/batch_search_{w['arch']}.json", 'w'))
search
"""),
        code("""
B_max = search['recommended']
lr = min(3e-3, 1e-3 * math.sqrt(B_max / B_base))
res = train(mc, TrainConfig(run_name=f"03_reversible_{w['arch']}_max_batch", batch_size=B_max,
                            lr=lr, min_lr=lr / 10,
                            data_dir='/content/data', out_dir=f'{DRIVE}/results'))
{k: res[k] for k in ['batch_size', 'steps', 'tokens_seen', 'final_train_loss', 'final_val_loss',
                     'tokens_per_sec', 'peak_mem_allocated_gb', 'wall_time_s']}
"""),
        md("**Isolating the memory saving.** `find_max_batch` above uses the memory-saving reversible "
           "backward. Re-running it with `use_rev_fn=False` (plain autograd, no activation reconstruction) "
           "at the same starting batch isolates how much of the memory saving comes from the custom "
           "backward's activation reconstruction, versus the two-stream reversible architecture itself."),
        code("""
search_no_revfn = find_max_batch(mc, start=B_base, use_rev_fn=False)
json.dump(search_no_revfn, open(f"{DRIVE}/batch_search_{w['arch']}_no_revfn.json", 'w'))
search_no_revfn
"""),
    ]


def nb_analysis():
    return [
        md("# 04: Analysis\nRun locally (or in Colab) after copying the Drive `results/*.json` into the repo's `results/`."),
        code("""
import os
from pathlib import Path
root = Path.cwd() if (Path.cwd() / 'results').exists() else Path.cwd().parent
os.chdir(root)
import sys; sys.path.insert(0, str(root / 'src'))
from revllm.report import load_results, results_table, plot_all
from IPython.display import Image, Markdown, display

rs = load_results('results')
display(Markdown(results_table(rs)))
for p in plot_all(rs, 'results/plots'):
    display(Image(p))
"""),
    ]


def build(repo_url: str, out_dir, subdir: str = "") -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    specs = {
        "01_baseline.ipynb": nb_baseline(repo_url, subdir),
        "02_reversible_same_batch.ipynb": nb_same_batch(repo_url, subdir),
        "03_reversible_max_batch.ipynb": nb_max_batch(repo_url, subdir),
        "04_analysis.ipynb": nb_analysis(),
    }
    paths = []
    for name, cells in specs.items():
        nb = nbf.v4.new_notebook(cells=cells)
        nb.metadata["accelerator"] = "GPU"
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        path = out / name
        nbf.write(nb, path)
        paths.append(path)
    return paths


if __name__ == "__main__":
    url = sys.argv[1]
    sub = sys.argv[2] if len(sys.argv) > 2 else ""
    for p in build(url, Path(__file__).resolve().parents[2] / "notebooks", subdir=sub):
        print(f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url(url, p.name, sub)}) `{p.name}`")
