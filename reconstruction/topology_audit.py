"""80 fixed validation cards and matched comparison of projected tracks.

The new extractor is a development diagnostic, not a frozen-test result.
Truth is accessed after both inference paths finish, except sample selection.
"""
import json
from pathlib import Path
import numpy as np
import joblib
from scipy.optimize import linear_sum_assignment
from .core import projections_from_hits
from .physical import PLANE_Z, STRIP_CENTERS
from .star_reconstruct import run_event
from .track_topology import infer_topology
from .topology_truth import visible_truth


def match_tracks(predictions, truth, axis):
    # Match all labelled co-born tracks, including primary, before reporting
    # visible-branch recall. Extra shower tracks can remain unmatched; these
    # are not automatically false physical tracks.
    eligible = [t for t in truth['tracks'] if len(t['views'][axis]['samples']) >= 3]
    costs = np.ones((len(eligible), len(predictions)))*100
    for i, t in enumerate(eligible):
        samples = t['views'][axis]['samples']
        z = np.array([PLANE_Z[2*r+1-axis] for r, _ in samples])
        value = np.array([STRIP_CENTERS[s] for _, s in samples])
        for j, p in enumerate(predictions):
            in_range = (z >= p['z_min']-4.1) & (z <= p['z_max']+4.1)
            residual = np.abs(value-(p['slope']*z+p['intercept']))
            covered = in_range & (residual <= 4.88)
            recall = covered.mean()
            # Prevent a very long spurious line matching a tiny real branch.
            overlap = max(0, min(p['z_max'], z.max())-max(p['z_min'], z.min()))
            span_precision = overlap/max(p['z_max']-p['z_min'], 1e-9)
            if recall >= .5 and span_precision >= .5:
                costs[i, j] = 1-recall+.001*np.mean(np.minimum(residual, 100))
    ii, jj = linear_sum_assignment(costs)
    matches = [(eligible[i]['track_id'], int(j)) for i, j in zip(ii, jj) if costs[i, j] < 10]
    visible = {t['track_id'] for t in truth['tracks'] if t['visible'] and not t['primary']}
    recovered = sorted(visible & {i for i, _ in matches})
    return dict(matches=matches, visible_branch_ids=sorted(visible), recovered_visible_branch_ids=recovered,
                unmatched_segments=len(predictions)-len(matches))


def baseline_tracks(report, axis):
    recon = report['reconstruction']
    if not recon or not recon['hypotheses']:
        return []
    tracks = []
    for a, b in recon['hypotheses'][0]['segments']:
        dz = b[2]-a[2]
        if abs(dz) < 1e-8:
            continue
        slope = (b[axis]-a[axis])/dz
        tracks.append(dict(slope=slope, intercept=a[axis]-slope*a[2], z_min=min(a[2], b[2]), z_max=max(a[2], b[2])))
    return tracks


def main():
    out = Path('results/topology_v2'); out.mkdir(exist_ok=True)
    labels = [json.loads(l) for l in Path('data/derived/labels.jsonl').read_text().splitlines()]
    rng = np.random.default_rng(240924)
    manifest = []
    for kind in ('proton', 'antiproton'):
        for stratum, size, condition in (
            ('two_three_long', 20, lambda r: r['star'] == 1 and r['branch_count'] in (2, 3)),
            ('many_long', 10, lambda r: r['star'] == 1 and r['branch_count'] > 3),
            ('other_interaction', 5, lambda r: r['category'] == 'other_interaction'),
            ('transportation', 5, lambda r: r['category'] == 'transportation')):
            pool = [r for r in labels if r['split'] == 'validation' and r['kind'] == kind and r['both_views'] and condition(r)]
            for i in sorted(rng.choice(len(pool), size, replace=False)):
                manifest.append(dict(kind=kind, event_id=pool[i]['event_id'], stratum=stratum))
    (out/'selection.json').write_text(json.dumps(dict(seed=240924, split='validation', events=manifest), indent=2)+'\n')
    model = joblib.load('results/star_model.joblib')
    records = []
    for kind, suffix in (('proton', ''), ('antiproton', ' (2)')):
        data = np.load(f'data/raw/calorimeter_response{suffix}.npy', allow_pickle=True).item()
        secondary = np.load(f'data/raw/secondary{suffix}.npy', allow_pickle=True).item()['secondary']
        for selected in [r for r in manifest if r['kind'] == kind]:
            event_id = selected['event_id']
            label = next(r for r in labels if r['kind'] == kind and r['event_id'] == event_id)
            xz, yz = projections_from_hits(data, event_id)
            baseline = run_event(model, xz, yz)
            topology = infer_topology(xz, yz)
            truth = visible_truth(secondary[event_id], label, xz, yz)
            btracks = [baseline_tracks(baseline, axis) for axis in (0, 1)]
            matches = {name: [match_tracks(tracks[a], truth, a) for a in (0, 1)]
                       for name, tracks in [('v1', btracks), ('segments', [topology['x']['tracks'], topology['y']['tracks']])]}
            vertex = label['vertex_mm']
            v1_vertex = (baseline['reconstruction'] or {}).get('vertex_mm')
            errors = dict(v1_vertex_mm=float(np.linalg.norm(np.array(v1_vertex)-vertex)) if vertex is not None and v1_vertex is not None else None,
                          first_topology_vertex_mm=float(np.linalg.norm(np.array(topology['vertex_candidates'][0]['vertex_mm'])-vertex)) if vertex is not None and topology['vertex_candidates'] else None,
                          oracle_best_proposed_vertex_mm=min([float(np.linalg.norm(np.array(c['vertex_mm'])-vertex)) for c in topology['vertex_candidates']], default=None) if vertex is not None else None)
            flags = []
            if label['star'] and not baseline['proposal']['is_star']: flags.append('detector_miss')
            if errors['v1_vertex_mm'] is not None and errors['v1_vertex_mm'] > 16.18: flags.append('vertex_error_over_16mm')
            if truth['visible_branches'] < (label['branch_count'] or 0): flags.append('long_tracks_not_two_view_separable')
            if any(set(m['visible_branch_ids'])-set(m['recovered_visible_branch_ids']) for m in matches['v1']): flags.append('v1_missing_projected_branches')
            if baseline['reconstruction'] and baseline['reconstruction'].get('near_optimal_pairings', 0) > 1: flags.append('v1_ambiguous_pairing')
            record = dict(**selected, category=label['category'], true_vertex_mm=vertex,
                          long_branches=label['branch_count'], truth=truth, topology=topology,
                          baseline_tracks=btracks, metrics=matches, errors=errors, flags=flags)
            records.append(record)
            np.savez_compressed(out/f'{kind}_{event_id}.npz', xz=xz, yz=yz)
            print(kind, event_id, 'visible', truth['visible_branches'], 'simple', truth['simple_star'], flush=True)
    (out/'events.json').write_text(json.dumps(records, indent=2)+'\n')
    summarize(records, out)


def summarize(records, out):
    from collections import Counter
    summary = dict(n=len(records), flags=dict(Counter(f for r in records for f in r['flags'])), groups={})
    for name, group in [('all', records), ('two_three_long', [r for r in records if r['stratum'] == 'two_three_long']),
                        ('simple_visible', [r for r in records if r['truth']['simple_star']])]:
        result = dict(n=len(group), methods={})
        for method in ('v1', 'segments'):
            total = sum(len(m['visible_branch_ids']) for r in group for m in r['metrics'][method])
            recovered = sum(len(m['recovered_visible_branch_ids']) for r in group for m in r['metrics'][method])
            both = sum(len(set(r['metrics'][method][0]['recovered_visible_branch_ids']) & set(r['metrics'][method][1]['recovered_visible_branch_ids'])) for r in group)
            result['methods'][method] = dict(visible_branches=total//2, recovered_in_views=recovered,
                                             view_recall=recovered/total if total else None,
                                             recovered_in_both_views=both, both_view_recall=both/(total/2) if total else None,
                                             mean_unmatched_segments=float(np.mean([m['unmatched_segments'] for r in group for m in r['metrics'][method]])) if group else None)
        summary['groups'][name] = result
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
