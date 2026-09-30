"""Evaluate actual X/Y connections using independently matched MC track IDs.

Unmatched/merged projected segments are unassessable, not counted correct.
This conservative diagnostic inherits the projection matching tolerances.
"""
import json
from pathlib import Path


def main():
    for root in (Path('results/topology_v2'), Path('results/topology_v2/development')):
        events = {(r['kind'],r['event_id']):r for r in json.loads((root/'events.json').read_text())}
        records=[]; total_visible=0
        for result in json.loads((root/'reconstruction.json').read_text()):
            r=events[result['kind'],result['event_id']]
            visible={t['track_id'] for t in r['truth']['tracks'] if t['visible'] and not t['primary']}
            total_visible+=len(visible)
            maps=[{int(pred):int(truth) for truth,pred in r['metrics']['segments'][axis]['matches']} for axis in (0,1)]
            pairs=[]; recovered=set()
            if result['hypotheses']:
                best=result['hypotheses'][0]
                for x,y in zip(best['x_source_track_ids'],best['y_source_track_ids']):
                    tx,ty=maps[0].get(x),maps[1].get(y)
                    assessable=tx not in (None,1) and ty not in (None,1)
                    correct=assessable and tx==ty
                    if correct and tx in visible: recovered.add(tx)
                    pairs.append(dict(x_segment=x,y_segment=y,truth_x_track=tx,truth_y_track=ty,
                                      assessable=assessable,correct=correct))
            records.append(dict(kind=r['kind'],event_id=r['event_id'],pairs=pairs,recovered_visible_ids=sorted(recovered)))
        pairs=[p for r in records for p in r['pairs']]
        summary=dict(visible_branches=total_visible, proposed_3d_pairs=len(pairs),
                     assessable_pairs=sum(p['assessable'] for p in pairs),
                     correct_assessable_pairs=sum(p['correct'] for p in pairs),
                     incorrect_assessable_pairs=sum(p['assessable'] and not p['correct'] for p in pairs),
                     unassessable_pairs=sum(not p['assessable'] for p in pairs),
                     recovered_visible_3d_branches=sum(len(r['recovered_visible_ids']) for r in records),
                     caution='Only top-ranked hypothesis; merged/primary/unmatched segments unassessable; abstentions count as no recovered branches',events=records)
        (root/'pairing_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(root,{k:v for k,v in summary.items() if k!='events'})


if __name__=='__main__': main()
