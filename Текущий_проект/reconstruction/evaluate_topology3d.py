"""Refresh truth diagnostics and evaluate abstaining v2 on the two audit sets."""
import json
from pathlib import Path
import numpy as np
import joblib
from .topology_truth import visible_truth
from .track_topology import vertex_proposals
from .topology_reconstruct import reconstruct_topology, serializable
from .topology_audit import summarize
from .star_reconstruction_benchmark import truth44, iou
from .star_reconstruct import run_event
from .protocol import sha256, verify_protocol


def main():
    verify_protocol()  # v1 remains byte-identical
    model = joblib.load('results/star_model.joblib')
    labels = {(r['kind'], r['event_id']): r for r in
              (json.loads(l) for l in Path('data/derived/labels.jsonl').read_text().splitlines())}
    raw, secondary = {}, {}
    for kind, suffix in (('proton', ''), ('antiproton', ' (2)')):
        raw[kind] = np.load(f'data/raw/calorimeter_response{suffix}.npy', allow_pickle=True).item()
        secondary[kind] = np.load(f'data/raw/secondary{suffix}.npy', allow_pickle=True).item()['secondary']
    for root in (Path('results/topology_v2'), Path('results/topology_v2/development')):
        rows = json.loads((root/'events.json').read_text())
        results = []
        for r in rows:
            a = np.load(root/f"{r['kind']}_{r['event_id']}.npz")
            r['topology']['vertex_candidates'] = vertex_proposals(r['topology']['x'], r['topology']['y'])
            prediction = reconstruct_topology(a['xz'], a['yz'], r['topology'])
            baseline = run_event(model, a['xz'], a['yz'])
            # Only now access evaluation coordinates and trajectories.
            gt = truth44(raw[r['kind']], r['event_id'])
            r['truth'] = visible_truth(secondary[r['kind']][r['event_id']], labels[r['kind'], r['event_id']], a['xz'], a['yz'])
            v = r['true_vertex_mm']
            proposed = r['topology']['vertex_candidates']
            r['errors']['first_topology_vertex_mm'] = float(np.linalg.norm(np.array(proposed[0]['vertex_mm'])-v)) if v is not None and proposed else None
            r['errors']['oracle_best_proposed_vertex_mm'] = min([float(np.linalg.norm(np.array(c['vertex_mm'])-v)) for c in proposed], default=None) if v is not None else None
            result = dict(kind=r['kind'], event_id=r['event_id'], category=r['category'], simple_star=r['truth']['simple_star'],
                          **serializable(prediction), v1_iou44=0., v2_iou44=0.)
            if baseline['reconstruction'] and baseline['reconstruction']['hypotheses']:
                h = baseline['reconstruction']['hypotheses'][0]
                result['v1_iou44'] = iou(h['mask'], gt > .01)
                if v is not None:
                    result['v1_vertex_error_mm'] = float(np.linalg.norm(np.array(baseline['reconstruction']['vertex_mm'])-v))
            if prediction['hypotheses']:
                h = prediction['hypotheses'][0]
                result['v2_iou44'] = iou(h['mask'], gt > .01)
                if v is not None:
                    result['v2_vertex_error_mm'] = float(np.linalg.norm(np.array(h['vertex_mm'])-v))
            result['hybrid_iou44'] = result['v2_iou44'] if prediction['status'] == 'ok' else result['v1_iou44']
            results.append(result)
        (root/'events.json').write_text(json.dumps(rows, indent=2)+'\n')
        (root/'reconstruction.json').write_text(json.dumps(results, indent=2)+'\n')
        summaries = {}
        for name, part in [('all', results), ('stars', [r for r in results if r['category'] == 'multi_prong']),
                           ('simple_visible', [r for r in results if r['simple_star']])]:
            covered = [r for r in part if r['status'] == 'ok']
            summaries[name] = dict(n=len(part), reconstructed=len(covered),
                v1_iou_mean_misses_zero=float(np.mean([r['v1_iou44'] for r in part])) if part else None,
                v2_iou_mean_misses_zero=float(np.mean([r['v2_iou44'] for r in part])) if part else None,
                v1_iou_on_v2_covered=float(np.mean([r['v1_iou44'] for r in covered])) if covered else None,
                v2_iou_on_v2_covered=float(np.mean([r['v2_iou44'] for r in covered])) if covered else None,
                v2_vertex_median_mm=float(np.median([r['v2_vertex_error_mm'] for r in covered if 'v2_vertex_error_mm' in r])) if any('v2_vertex_error_mm' in r for r in covered) else None)
            difference = np.array([r['hybrid_iou44']-r['v1_iou44'] for r in part])
            rng = np.random.default_rng(771)
            summaries[name]['hybrid_iou_mean_misses_zero'] = float(np.mean([r['hybrid_iou44'] for r in part])) if part else None
            summaries[name]['hybrid_minus_v1_paired_bootstrap95'] = np.quantile([rng.choice(difference, len(difference), replace=True).mean() for _ in range(1000)], [.025, .975]).tolist() if part else None
        (root/'reconstruction_summary.json').write_text(json.dumps(summaries, indent=2)+'\n')
        summarize(rows, root)
        print(root, json.dumps(summaries, indent=2), flush=True)
    sources = ['track_topology.py', 'topology_truth.py', 'topology_audit.py', 'simple_star_development.py',
               'topology_reconstruct.py', 'evaluate_topology3d.py']
    meta = dict(stage='development; validation already used in iteration, not a new test',
                visibility='>=3 supported planes AND >=2 separated planes in EACH view; separation +/-1 strip from other primary/co-born long tracks; signal >.01 MeV',
                simple='2-3 branches; all including primary visible; straightness P90 <=3.66mm; >=70% active strips explained in each view',
                match='Hungarian one-to-one; >=50% reference samples within 4.88mm and predicted range; >=50% predicted z-span overlaps truth',
                caveats=['ParentID unavailable', 'no track ownership of deposited energy', 'separation ignores other shower trajectories',
                         'v1 includes detector gate; segment extraction does not', 'both-view recall is NOT correct 3D-pairing recall',
                         'hybrid uses v2 if status ok, otherwise v1; development rule, not test-validated',
                         'rejecting an event is counted as zero IoU in all-event means'],
                sources_sha256={p:sha256(Path('reconstruction')/p) for p in sources})
    Path('results/topology_v2/protocol.json').write_text(json.dumps(meta, indent=2)+'\n')


if __name__ == '__main__': main()
