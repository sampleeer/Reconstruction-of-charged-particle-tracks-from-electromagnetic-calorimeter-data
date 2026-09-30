"""Run the abstaining topology prototype on one trusted local Geant4 file."""
import argparse
import json
from pathlib import Path
import numpy as np
from .core import projections_from_hits
from .track_topology import infer_topology
from .topology_reconstruct import reconstruct_topology, reconstruct_hybrid, serializable


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('file',type=Path);p.add_argument('--event-id',type=int,required=True)
    p.add_argument('--out',type=Path,default=Path('results/topology_event'))
    p.add_argument('--fallback-v1', action='store_true', help='Use the saved v1 model on topology abstentions')
    a=p.parse_args()
    data=np.load(a.file,allow_pickle=True).item()
    x,y=projections_from_hits(data,a.event_id)
    topology=infer_topology(x,y)
    if a.fallback_v1:
        import joblib
        result=reconstruct_hybrid(x,y,joblib.load('results/star_model.joblib'),topology)
    else:
        result=reconstruct_topology(x,y,topology)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.with_suffix('.json').write_text(json.dumps(dict(event_id=a.event_id,topology=topology,
                                                       reconstruction=serializable(result)),indent=2)+'\n')
    if result['hypotheses']:
        h=result['hypotheses'][0]
        np.savez_compressed(a.out.with_suffix('.npz'),xz=x,yz=y,energy=h['energy'],mask=h['mask'])
    print(json.dumps(dict(status=result['status'],vertex_proposals=result['vertex_proposal_count'],
                         hypotheses=len(result['hypotheses'])),indent=2))


if __name__ == '__main__':main()
