"""Truth-curated TRAIN examples to debug finite-track extraction.

Selection is intentionally favourable and must not be used as a population
quality estimate. It is separate from the fixed validation audit.
"""
import json
from pathlib import Path
import joblib
import numpy as np
from .core import projections_from_hits
from .star_reconstruct import run_event
from .track_topology import infer_topology
from .topology_truth import visible_truth
from .topology_audit import match_tracks, baseline_tracks, summarize


def main():
    out = Path('results/topology_v2/development'); out.mkdir(exist_ok=True)
    labels = [json.loads(s) for s in Path('data/derived/labels.jsonl').read_text().splitlines()]
    model = joblib.load('results/star_model.joblib')
    records, scanned = [], {}
    for kind, suffix in (('proton', ''), ('antiproton', ' (2)')):
        data = np.load(f'data/raw/calorimeter_response{suffix}.npy', allow_pickle=True).item()
        secondary = np.load(f'data/raw/secondary{suffix}.npy', allow_pickle=True).item()['secondary']
        pool = [r for r in labels if r['split'] == 'train' and r['kind'] == kind and r['branch_count'] in (2, 3) and r['star'] == 1]
        rng = np.random.default_rng(404); rng.shuffle(pool)
        found = 0
        for index, label in enumerate(pool):
            event_id = label['event_id']; xz, yz = projections_from_hits(data, event_id)
            truth = visible_truth(secondary[event_id], label, xz, yz)
            if not truth['simple_star']:
                continue
            baseline = run_event(model, xz, yz)
            topology = infer_topology(xz, yz)
            btracks = [baseline_tracks(baseline, axis) for axis in (0, 1)]
            matches = {name: [match_tracks(tracks[a], truth, a) for a in (0, 1)]
                       for name, tracks in [('v1', btracks), ('segments', [topology['x']['tracks'], topology['y']['tracks']])]}
            records.append(dict(kind=kind, event_id=event_id, category=label['category'],
                                stratum='two_three_long', true_vertex_mm=label['vertex_mm'],
                                long_branches=label['branch_count'], truth=truth, topology=topology,
                                baseline_tracks=btracks, metrics=matches, flags=[], errors={}))
            np.savez_compressed(out/f'{kind}_{event_id}.npz', xz=xz, yz=yz)
            found += 1
            print(kind, event_id, 'selected', found, 'scanned', index+1, flush=True)
            if found == 10: break
        scanned[kind] = dict(scanned=index+1, selected=found, pool=len(pool))
    (out/'events.json').write_text(json.dumps(records, indent=2)+'\n')
    (out/'selection.json').write_text(json.dumps(dict(seed=404, split='train', truth_curated=True, counts=scanned,
                                                      criteria='2-3 co-born long branches; each including primary visible and straight; >=70% measured strips explained within +/-1 strip'), indent=2)+'\n')
    summarize(records, out)


if __name__ == '__main__': main()
