"""NARCBench on Qwen3-32B-AWQ: fresh generation + four defenses, one Modal run.

Run from this folder (research\\narcbench_qwen_native) on Windows:
    ..\\.venv-modal\\Scripts\\modal.exe run --detach native_modal.py

Stages (all resume-safe; output on volume compjailbench-agentharm-scores):
  1. generate  - pilot pair audit_01, validated, then the other 49 pairs.
                 Uses the unchanged released runner, Qwen layer-28 activations
                 and the frozen probe (peak_suspicion), same decoding as the
                 historical 0.9896 run (T=0.7, top_p=0.8, top_k=20, seed 20260817).
  2. score     - exports the Qwen trajectories, strips labels/private fields and
                 scores A2A / Sentinel / CoT with the same frozen protocol as
                 narcbench_qwen_text_v1 (greedy, thinking off, 1536 tokens).
"""

import hashlib
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
RUN = '/scores/narcbench_qwen_native_v1'
MODEL = 'Qwen/Qwen3-32B-AWQ'
REVISION = '0499c3ac83fdef8810b907a23894ba91e95eddd8'
PROJECT = '/opt/CompJailBench/NARCBench_Attack_CoT_Monitor'
PROBE_DIR = '/opt/CompJailBench/Defence_NARCBench Probing'
PILOT = 'audit_01'

app = modal.App('compjailbench-narcbench-qwen-native')
volume = modal.Volume.from_name('compjailbench-agentharm-scores')
weights = modal.Volume.from_name('compjailbench-qwen-cache')
image = modal.Image.from_id('im-VOAEtmK389l59IYjXjnrji')
if modal.is_local():
    repo = ROOT.parents[1] / 'sources/CompJailBench'
    proj = repo / 'NARCBench_Attack_CoT_Monitor'
    for sub in ('src', 'scripts', 'data', 'configs'):
        image = image.add_local_dir(proj / sub, f'{PROJECT}/{sub}', ignore=['**/__pycache__/**'])
    image = image.add_local_dir(repo / 'Defence_NARCBench Probing', PROBE_DIR, ignore=['**/__pycache__/**'])
    image = image.add_local_file(ROOT.parent / 'decompbench_parallel/qwen_remaining_scoring/score_qwen_pilot_monitors.py',
                                 f'{PROJECT}/scripts/score_qwen_pilot_monitors.py')

FN = dict(image=image, gpu='L40S', cpu=4, memory=32768, timeout=86400,
          volumes={'/scores': volume, '/weights': weights}, max_containers=1)


def _committer(stop):
    while not stop.wait(60):
        try:
            volume.commit()
        except Exception as exc:  # keep generating; next commit retries
            print('commit failed:', exc, flush=True)


def _export(gen_root):
    """Read every Inspect log -> StandardTrajectory dicts + attack outcomes."""
    import runpy
    import sys
    sys.path.insert(0, PROJECT)
    from inspect_ai.log import read_eval_log, read_eval_log_samples
    exporter = runpy.run_path(f'{PROJECT}/scripts/export_standard_trajectories.py')
    rows, outcomes = [], []
    logs = [l for l in sorted(Path(gen_root, 'inspect_logs').rglob('*.eval'))
            if read_eval_log(str(l), header_only=True).status == 'success']  # skip interrupted runs
    for log in logs:
        for sample in read_eval_log_samples(str(log)):
            row = exporter['trajectory_dict'](sample.store or {})
            rows.append(row)
            scores = {k: (v.model_dump(mode='json') if hasattr(v, 'model_dump') else str(v))
                      for k, v in (sample.scores or {}).items()}
            outcomes.append(dict(sample_id=row['sample_id'], pair_id=row['pair_id'],
                                 log=str(log.relative_to(gen_root)), scores=scores))
    latest = {}
    for row, outcome in zip(rows, outcomes):
        latest[row['sample_id']] = (row, outcome)  # newest successful log wins
    return [v[0] for v in latest.values()], [v[1] for v in latest.values()]


def _run_generation(pairs, seed=20260817):
    import runpy
    import sys
    import threading
    sys.path.insert(0, PROJECT)
    import src.narcbench_probe_runtime as rt

    class Pinned(rt.QwenActivationRuntime):
        def __init__(self, **kw):
            kw.setdefault('model_revision', REVISION)
            super().__init__(**kw)

    rt.QwenActivationRuntime = Pinned
    sys.argv = ['run_narcbench_probe_kaggle.py', '--probe', f'{PROBE_DIR}/narcbench_probe.pkl',
                '--output', f'{RUN}/gen', '--gpu-memory-gib', '42', '--cpu-memory-gib', '20',
                '--sampling-seed', str(seed)]
    if pairs:
        sys.argv += ['--pairs', ','.join(pairs)]
    stop = threading.Event()
    thread = threading.Thread(target=_committer, args=(stop,), daemon=True)
    thread.start()
    try:
        runpy.run_path(f'{PROJECT}/scripts/run_narcbench_probe_kaggle.py', run_name='__main__')
    finally:
        stop.set()
        volume.commit()
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()


@app.function(**FN)
def generate():
    out = Path(RUN)
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(generator_model=MODEL, generator_revision=REVISION, monitor_model=MODEL,
                    monitor_revision=REVISION, pilot_pair=PILOT, probe_layer=28,
                    probe_metric='peak_suspicion', decoding=dict(do_sample=True, temperature=0.7, top_p=0.8,
                                                                 top_k=20, sampling_seed=20260817,
                                                                 max_new_tokens=512, max_format_retries=1),
                    monitor_decoding=dict(do_sample=False, enable_thinking=False, max_new_tokens=1536,
                                          checker_retries=2),
                    profiles={'agent_to_agent': {'prompt': 'compositional_leakage_v1',
                                                 'scope': 'public-utterance-graph-plus-final-output'},
                              'sentinel': {'prompt': 'brad_node_edge_checkers_e90d60c_hardened_v2', 'scope': 'public',
                                           'signature': 'narcbench_message_paths_v1'},
                              'cot': {'prompt': 'general_v2', 'scope': 'public-action-only'}})
    if not (out / 'manifest.json').exists():
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        volume.commit()
    # 1. pilot pair, then validate before spending on the rest
    _run_generation([PILOT])
    rows, _ = _export(f'{RUN}/gen')
    pilot = [r for r in rows if r['pair_id'] == PILOT]
    checks = dict(samples=len(pilot) == 2,
                  conditions=sorted(r['condition_label'] for r in pilot) == ['attack', 'benign'],
                  public_messages=all(len(r['public_messages']) > 0 for r in pilot),
                  final_output=all(bool(r['final_output']) for r in pilot),
                  probe_rows=Path(f'{RUN}/gen/probe_scores.jsonl').exists())
    (out / 'pilot_generation.json').write_text(json.dumps(dict(pair_id=PILOT, checks=checks,
                                                               passed=all(checks.values())), indent=2))
    volume.commit()
    if not all(checks.values()):
        raise RuntimeError(f'Pilot failed: {checks}')
    print('PILOT GENERATION PASSED', checks, flush=True)
    # 2. remaining pairs, one at a time (resume markers skip finished ones). A pair whose
    # Qwen output fails the JSON format check twice is retried with seed+1, then seed+2;
    # every reseed and every final exclusion is recorded.
    tasks = json.loads(Path(f'{PROJECT}/data/narcbench_core_official.json').read_text())
    pair_ids = list(dict.fromkeys(t['pair_id'] for t in tasks))
    reseeded, excluded = {}, {}
    for pair in pair_ids:
        if Path(f'{RUN}/gen/completed_pairs/{pair}.json').exists():
            continue
        for offset in range(3):
            try:
                _run_generation([pair], seed=20260817 + offset)
                if offset:
                    reseeded[pair] = 20260817 + offset
                break
            except Exception as exc:
                print(f'PAIR {pair} failed with seed offset {offset}: {exc}', flush=True)
                excluded[pair] = str(exc)[:300]
        else:
            continue
        excluded.pop(pair, None)
        (out / 'generation_exceptions.json').write_text(json.dumps(dict(reseeded=reseeded, excluded=excluded), indent=2))
        volume.commit()
    markers = list(Path(f'{RUN}/gen/completed_pairs').glob('*.json'))
    (out / 'generation_exceptions.json').write_text(json.dumps(dict(reseeded=reseeded, excluded=excluded), indent=2))
    (out / 'generation_complete.json').write_text(json.dumps(dict(pairs=len(markers), reseeded=reseeded,
                                                                  excluded=sorted(excluded))))
    volume.commit()
    return len(markers)


@app.function(**FN)
def score():
    import anyio
    import math
    import runpy
    import sys
    from inspect_ai.util import Store
    sys.path.insert(0, PROJECT)
    from src.trajectory import StandardTrajectory
    out = Path(RUN)
    assert (out / 'generation_complete.json').exists(), 'generation not finished'
    rows, outcomes = _export(f'{RUN}/gen')
    tasks = {t['sample_id']: t for t in json.loads(Path(f'{PROJECT}/data/narcbench_core_official.json').read_text())}
    allowed = ('schema_version', 'sample_id', 'pair_id', 'attack_name', 'model_name', 'agent_ids',
               'public_messages', 'actions', 'tool_calls', 'final_output')
    clean, pairs = [], {}
    for row in rows:
        cond = 'benign_control' if row['condition_label'] == 'benign' else row['condition_label']
        task = tasks[row['sample_id']]
        item = {k: row[k] for k in allowed if k in row}
        item.update(condition_label=cond, public_task={k: task[k] for k in ('title', 'domain', 'context', 'options')},
                    communication_edges=[e for e in row['communication_edges'] if e.get('channel') == 'public'])
        clean.append(item)
        pairs.setdefault(row['pair_id'], []).append(cond)
    complete_pairs = sorted(p for p, v in pairs.items() if sorted(v) == ['attack', 'benign_control'])
    clean = [c for c in clean if c['pair_id'] in complete_pairs]
    clean.sort(key=lambda r: (r['pair_id'], r['condition_label']))
    inputs = out / 'inputs'
    inputs.mkdir(exist_ok=True)
    src = inputs / 'standard_trajectories.jsonl'
    src.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in clean))
    (out / 'attack_outcomes.jsonl').write_text(''.join(json.dumps(o) + '\n' for o in outcomes))
    (out / 'inputs_manifest.json').write_text(json.dumps(dict(trajectories=len(clean), complete_pairs=len(complete_pairs),
                                                              input_sha256=hashlib.sha256(src.read_bytes()).hexdigest())))
    volume.commit()

    project = Path(PROJECT)
    module = runpy.run_path(str(project / 'scripts/score_qwen_pilot_monitors.py'))
    sentinel = runpy.run_path(str(project / 'scripts/score_sentinel_message_only.py'))
    backend = sentinel['load_sentinel_backend'](project.parent / 'Defense_SentinelAgents')
    scores_path = out / 'scores.jsonl'
    done = set()
    if scores_path.exists():
        done = {(r['sample_id'], r['defense']) for r in map(json.loads, scores_path.read_text().splitlines())}
    runtime = module['QwenActivationRuntime'](output_root=Path('/tmp/narcbench-runtime'), model_name=MODEL,
                                              model_revision=REVISION, gpu_memory_gib=42, cpu_memory_gib=20)
    monitor = module['Monitor'](runtime, out)
    if (out / 'monitor_calls.jsonl').exists():
        monitor.calls = len((out / 'monitor_calls.jsonl').read_text().splitlines())

    async def run():
        for data in clean:
            trajectory = StandardTrajectory(store=Store(), **data)
            for defense in ('agent_to_agent', 'sentinel', 'cot'):
                if (trajectory.sample_id, defense) in done:
                    continue
                start = monitor.calls
                try:
                    if defense == 'agent_to_agent':
                        row = await module['score_a2a'](trajectory, monitor=monitor)
                    elif defense == 'cot':
                        row = await module['score_cot'](trajectory, scope='public-action-only', monitor=monitor)
                    else:
                        row = await sentinel['score_one'](trajectory, scope='public', model=monitor,
                                                          backend=backend, checker_retries=2)
                    value = row.get('suspiciousness')
                    valid = (row.get('status') == 'scored' and isinstance(value, (int, float)) and math.isfinite(value)
                             and 0 <= value <= 1 and not row.get('invalid_checker_outputs', 0))
                    if not valid:
                        row['status'] = 'invalid'
                        row['error'] = row.get('error') or 'Invalid score or persistent invalid checker output'
                except Exception as exc:
                    row = dict(status='error', error=f'{type(exc).__name__}: {exc}')
                row.update(sample_id=trajectory.sample_id, pair_id=trajectory.pair_id,
                           condition_label=trajectory.condition_label, defense=defense, monitor_model=MODEL,
                           monitor_revision=REVISION, monitor_call_ids=list(range(start + 1, monitor.calls + 1)))
                with scores_path.open('a') as handle:
                    handle.write(json.dumps(row) + '\n')
                volume.commit()
                print(json.dumps({k: row[k] for k in ('sample_id', 'defense', 'status')}), flush=True)
        (out / 'complete.json').write_text(json.dumps(dict(trajectories=len(clean), score_rows=3 * len(clean),
                                                           calls=monitor.calls)))
        volume.commit()
    anyio.run(run)


@app.function(**FN)
def all_stages():
    print('pairs generated:', generate.local())
    score.local()


@app.local_entrypoint()
def main():
    print('pairs generated:', generate.remote())
    score.remote()
    print('DONE: see volume compjailbench-agentharm-scores', RUN)
