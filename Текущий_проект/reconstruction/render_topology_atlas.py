"""Render auditable event cards and a local review UI; no external service."""
import argparse
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .physical import PLANE_Z, STRIP_CENTERS


def render_card(record, root):
    stem = f"{record['kind']}_{record['event_id']}"
    arrays = np.load(root/f'{stem}.npz')
    fig, axes = plt.subplots(3, 2, figsize=(9, 9), sharex=True, sharey=True, constrained_layout=True)
    for axis, key in enumerate(('xz', 'yz')):
        image = arrays[key]; rr, cc = np.nonzero(image > .01)
        z = PLANE_Z[1-axis::2]
        for row in range(3):
            ax = axes[row, axis]
            ax.scatter(STRIP_CENTERS[cc], z[rr], s=11, c=np.log1p(image[rr, cc]), cmap='Greys', vmin=0, vmax=np.log1p(image.max()), marker='s', alpha=.75)
            ax.set(xlim=(-123, 123), ylim=(178, 0), xlabel='X, mm' if axis == 0 else 'Y, mm', ylabel='Z, mm')
            ax.grid(alpha=.15)
        ax = axes[0, axis]
        for i, t in enumerate(record['truth']['tracks']):
            samples = t['views'][axis]['samples']
            if samples:
                coords = np.array([(STRIP_CENTERS[s], z[r]) for r, s in samples])
                ax.plot(coords[:, 0], coords[:, 1], '.-', lw=1, ms=2, label=f"MC {t['track_id']}"+(' primary' if t['primary'] else ''))
        if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=6, loc='best', ncol=2)
        ax.set_title('MC reference / '+key.upper(), fontsize=10)
        for row, tracks, title in [(1, record['baseline_tracks'][axis], 'v1: vertex-anchored rays'),
                                    (2, record['topology'][('x','y')[axis]]['tracks'], 'v2: finite projected segments')]:
            for j, t in enumerate(tracks):
                zs = np.array([t['z_min'], t['z_max']]); values = t['slope']*zs+t['intercept']
                axes[row, axis].plot(values, zs, '-', color='#db6334', lw=1.4)
                axes[row, axis].text(values.mean(), zs.mean(), str(j), fontsize=6, color='#a33312', clip_on=True)
            axes[row, axis].set_title(title, fontsize=10)
        if record['true_vertex_mm']:
            for row in range(3):
                axes[row, axis].scatter(record['true_vertex_mm'][axis], record['true_vertex_mm'][2], marker='x', c='#137b51', s=40, zorder=4)
    both = {m: len(set(record['metrics'][m][0]['recovered_visible_branch_ids']) & set(record['metrics'][m][1]['recovered_visible_branch_ids'])) for m in ('v1','segments')}
    fig.suptitle(f"{record['kind']} #{record['event_id']} | long={record['long_branches']}, two-view visible={record['truth']['visible_branches']}\nRecovered in both views: v1={both['v1']}, segments={both['segments']} | green x = MC vertex (evaluation only)", fontsize=10)
    fig.savefig(root/f'{stem}.png', dpi=105)
    plt.close(fig)


PAGE = '''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Разбор топологии звёзд</title>
<style>body{font:16px system-ui;background:#f3f5f7;color:#172530;margin:24px}main{max-width:1300px;margin:auto}h1{font-size:26px}button,select,input,textarea{font:inherit;padding:8px;border:1px solid #bac6cf;border-radius:6px}button{cursor:pointer;background:white}nav{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0}.layout{display:grid;grid-template-columns:minmax(450px,2fr) minmax(270px,1fr);gap:20px}img{width:100%;background:white}aside{background:white;padding:18px;border-radius:8px;height:fit-content}label{display:block;margin:12px 0}textarea{box-sizing:border-box;width:100%;height:120px}.muted{color:#596977;font-size:14px}pre{white-space:pre-wrap;font-size:13px} @media(max-width:800px){.layout{display:block}}</style>
<main><h1>Разбор топологии звёзд · __TITLE__</h1><p>Серые точки — измерения; верхний ряд — траектории Geant4; средний — v1; нижний — новый поиск конечных отрезков. Зелёный крест — истинная вершина, показанная только для оценки.</p>
<p class="muted">Автоматические признаки ошибок требуют проверки человеком. Отрезок, не сопоставленный с прямой вторичной ветвью, может принадлежать ливню. Цвета MC и номера восстановленных отрезков не означают установленного родства.</p>
<nav><select id="filter"><option value="all">Все события</option><option value="simple">Простые и видимые</option><option value="vertex_error_over_16mm">Ошибка вершины &gt;16 мм</option><option value="v1_missing_projected_branches">Потерянные проекционные ветви</option><option value="long_tracks_not_two_view_separable">Не все длинные ветви разделимы</option></select><button id="prev">←</button><select id="event"></select><button id="next">→</button><button id="download">Скачать свои пометки</button></nav>
<div class="layout"><img id="card"><aside><h2 id="name"></h2><div id="details"></div><p class="muted">«Найдена в двух видах» ещё не означает правильного соединения ветви в 3D.</p><label><input id="reviewed" type="checkbox"> Я просмотрел событие</label><label>Разметка видимости<select id="verdict"><option value="unreviewed">Не проверена</option><option value="agree">Согласен</option><option value="disagree">Нужна корректировка</option><option value="unclear">Неоднозначно</option></select></label><label>Заметка<textarea id="note" placeholder="Какая ветвь потеряна, где излом или ложное соединение?"></textarea></label><p class="muted" id="storage">Пометки сохраняются в этом браузере. Для передачи сохраните JSON кнопкой выше.</p><pre id="metrics"></pre></aside></div></main>
<script>const records=__DATA__;const namespace='diploma-topology-__TITLE__';let notes={};try{notes=JSON.parse(localStorage.getItem(namespace)||'{}')}catch(e){}let chosen=[],index=0;const el=id=>document.getElementById(id);const key=r=>r.kind+'_'+r.event_id;
function save(){const r=chosen[index];if(!r)return;notes[key(r)]={reviewed:el('reviewed').checked,visibility:el('verdict').value,note:el('note').value};try{localStorage.setItem(namespace,JSON.stringify(notes))}catch(e){el('storage').textContent='Браузер не разрешает локальное сохранение. Скачайте JSON перед закрытием.'}}
function show(){const r=chosen[index];if(!r){el('name').textContent='Нет событий по этому фильтру';el('card').removeAttribute('src');return}el('event').value=String(index);el('name').textContent=r.kind+' #'+r.event_id;el('card').src=key(r)+'.png';el('details').textContent='Длинных ветвей: '+r.long_branches+'; разделимых в двух видах: '+r.truth.visible_branches+'. Признаки: '+(r.flags.join(', ')||'нет автоматических признаков');el('metrics').textContent=JSON.stringify({errors:r.errors,metrics:r.metrics},null,2);const n=notes[key(r)]||{};el('reviewed').checked=!!n.reviewed;el('verdict').value=n.visibility||'unreviewed';el('note').value=n.note||''}
function filter(){chosen=records.filter(r=>el('filter').value==='all'||(el('filter').value==='simple'?r.truth.simple_star:r.flags.includes(el('filter').value)));index=0;el('event').replaceChildren(...chosen.map((r,i)=>new Option(r.kind+' #'+r.event_id,String(i))));show()}
el('filter').onchange=filter;el('event').onchange=()=>{index=Number(el('event').value);show()};el('prev').onclick=()=>{if(chosen.length){index=(index+chosen.length-1)%chosen.length;show()}};el('next').onclick=()=>{if(chosen.length){index=(index+1)%chosen.length;show()}};for(const id of ['reviewed','verdict','note'])el(id).oninput=save;el('download').onclick=()=>{const a=document.createElement('a');const u=URL.createObjectURL(new Blob([JSON.stringify({sample:namespace,annotations:notes},null,2)],{type:'application/json'}));a.href=u;a.download='topology_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};filter();</script></html>'''


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, default=Path('results/topology_v2'))
    parser.add_argument('--skip-images', action='store_true'); args=parser.parse_args()
    records = json.loads((args.root/'events.json').read_text())
    reconstruction_path = args.root/'reconstruction.json'
    if reconstruction_path.exists():
        predictions = {(r['kind'], r['event_id']):r for r in json.loads(reconstruction_path.read_text())}
        for record in records:
            r = predictions.get((record['kind'], record['event_id']), {})
            record['errors']['reconstruction3d'] = {k:v for k,v in r.items() if k in ('status','v1_iou44','v2_iou44','hybrid_iou44','v2_vertex_error_mm')}
    for i, record in enumerate(records):
        if not args.skip_images: render_card(record, args.root)
        if i % 10 == 0: print('Rendered', i+1, '/', len(records), flush=True)
    name = 'train-curated' if args.root.name == 'development' else 'validation-80'
    page = PAGE.replace('__TITLE__', name).replace('__DATA__', json.dumps(records).replace('<','\\u003c'))
    (args.root/'index.html').write_text(page)


if __name__ == '__main__': main()
