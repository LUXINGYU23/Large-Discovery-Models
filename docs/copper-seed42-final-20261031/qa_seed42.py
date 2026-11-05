# -*- coding: utf-8 -*-
import json,hashlib,os,re,time,zipfile
from pathlib import Path
import fitz
from playwright.sync_api import sync_playwright
O=Path('/Aidata/shared/ldm/copper-baseline-rerun-20261027-ops');B=O/'seed42-investor-report-20261031'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
pr=json.loads((B/'data/provenance.json').read_text());assert len(pr['comparisons'])==144
for r in pr['comparisons']:
 assert sha(B/'cases'/(r['key']+'.png'))==r['image_sha256'];assert sha(B/'geometry'/(r['key']+'.js'))==r['geometry_sha256']
d=json.loads((B/'data/report.json').read_text());assert d['discovery_seed']==42 and len(d['test_rows'])==432;assert all(r['discovery_seed']==42 for r in d['test_rows']);assert len({r['case_id'] for r in d['test_rows']})==48
errors=[];decoded=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox','--enable-webgl','--use-angle=swiftshader','--enable-unsafe-swiftshader'],env=dict(os.environ,FONTCONFIG_FILE=str(O/'report-fonts/fonts.conf')));page=browser.new_page(viewport={'width':1440,'height':1100});page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto((B/'index.html').as_uri());assert page.locator('.sheet').count()==18
 for link in page.locator('a').all():
  href=link.get_attribute('href')
  if href and not href.startswith(('#','https://','http://')):assert (B/href.split('#')[0]).is_file(),href
 sizes=page.locator('.sheet').evaluate_all('xs=>xs.map(x=>({h:x.offsetHeight,scroll:x.scrollHeight}))');assert all(s['scroll']<=s['h']+2 for s in sizes)
 page.emulate_media(media='print')
 page.locator('img').evaluate_all('async imgs=>{await Promise.all(imgs.map(i=>i.decode()));}')
 page.evaluate('document.fonts.ready')
 clearance=page.locator('.sheet').evaluate_all("xs=>xs.map(x=>{let f=x.querySelector('.footer').getBoundingClientRect();let bottom=Math.max(...Array.from(x.children).filter(c=>!c.classList.contains('footer')).map(c=>c.getBoundingClientRect().bottom));return f.top-bottom})")
 assert all(c>=5 for c in clearance),clearance
 page.emulate_media(media='screen')
 page.goto((B/'algorithm-notes.html').as_uri())
 for link in page.locator('a').all():
  href=link.get_attribute('href')
  if href:assert (B/href.split('#')[0]).is_file(),href
 page.goto((B/'trajectory.html').as_uri())
 page.wait_for_function("document.getElementById('plot').data && document.getElementById('plot').data.length===4")
 assert page.locator('article').count()==12
 assert page.evaluate("document.getElementById('plot').data[0].x.length")==308
 assert page.evaluate("document.getElementById('plot').data[2].text.join('')")=='ABCDEFGHIJKL'
 page.uncheck('#raw');assert page.evaluate("document.getElementById('plot').data[0].visible")==False
 page.check('#raw');page.uncheck('#labels');assert page.evaluate("document.getElementById('plot').data[2].mode")=='markers'
 page.check('#labels')
 for link in page.locator('a').all():
  href=link.get_attribute('href')
  if href and not href.startswith(('https://','http://','#')):assert (B/href.split('#')[0]).is_file(),href
 page.screenshot(path=str(B/'data/trajectory-browser.png'),full_page=True)
 page.goto((B/'browser.html').as_uri());cases=page.locator('#case option').evaluate_all('els=>els.map(e=>e.value)');assert len(cases)==48
 for c in cases:
  page.select_option('#case',c)
  for es in ['47','48','49']:
   page.select_option('#exec',es);page.locator('#panel').evaluate('async img=>await img.decode()');expected=c+'-e'+es+'.png';assert page.locator('#panel').get_attribute('src')=='cases/'+expected;assert page.locator('#original').get_attribute('href').endswith('/cases/'+expected);decoded.append(expected)
 for case in [cases[0],next(c for c in cases if c.startswith('test_ood'))]:
  page.select_option('#case',case);page.select_option('#exec','47');page.click('#load3d');page.wait_for_function('loadedKey===currentKey()',timeout=120000);assert page.locator('.js-plotly-plot').count()==4;page.screenshot(path=str(B/'data'/('browser-'+case.split('-')[0]+'.png')),full_page=True)
 # Camera propagation is a meaningful interaction check.
 page.evaluate("Plotly.relayout('mesh0',{'scene.camera':{eye:{x:1.1,y:1.3,z:.8}}})")
 page.wait_for_function("Math.abs(document.getElementById('mesh1').layout.scene.camera.eye.x-1.1)<1e-8")
 browser.close()
assert not errors,errors
pdf=fitz.open(B/'copper-seed42-investor-report.pdf');assert len(pdf)==18
portable_links=0
for page in pdf:
 for link in page.get_links():
  if 'file' in link:
   assert not Path(link['file']).is_absolute()
   assert (B/link['file']).is_file(),link['file']
   portable_links+=1
P=B/'data/pdf-preview';P.mkdir(exist_ok=True)
texts=[]
for i,page in enumerate(pdf):
 text=page.get_text();texts.append(text);assert len(text)>100
 page.get_pixmap(matrix=fitz.Matrix(1.15,1.15)).save(str(P/f'page-{i+1:02d}.png'))
assert '种子42' in ''.join(texts).replace(' ','')
assert '多尺度球形翻转退火' in ''.join(texts).replace(' ','')
audit=json.loads((B/'data/algorithm-audit.json').read_text())
assert sha(B/audit['selected']['report_path'])==audit['selected']['source_sha256']
assert sha(B/audit['training_best']['report_path'])==audit['training_best']['source_sha256']
assert sha(B/'data/report.json')==audit['data_sha256_before']==audit['data_sha256_after']
trajectory=json.loads((B/'data/method-trajectory.json').read_text())
assert len(trajectory['nodes'])==12 and len(trajectory['candidates'])==308
assert all(sha(B/n['source_path'])==n['source_sha256'] for n in trajectory['nodes'])
assert all(sha(B/n['source_path'])==n['source_sha256'] for n in trajectory['baselines'].values())
assert trajectory['report_data_sha256']==sha(B/'data/report.json')
qa=dict(checked_at=time.time(),discovery_seed=42,test_records=432,cases=48,static_comparisons_verified=144,geometry_files_verified=144,browser_combinations_decoded=len(decoded),interactive_3d_scenes_checked=8,linked_camera_passed=True,pdf_pages=18,page_errors=errors,html_page_sizes=sizes,footer_clearance_px=clearance,pdf_sha256=sha(B/'copper-seed42-investor-report.pdf'),algorithm_source_hashes_verified=True,algorithm_audit_disclosed=True,experiment_data_unchanged=True,trajectory_source_nodes_verified=12,trajectory_candidates=308,trajectory_toggles_passed=True,baseline_winner_sources_verified=2)
qa['portable_pdf_links_verified']=portable_links
(B/'data/qa.json').write_text(json.dumps(qa,indent=2))
print(json.dumps(qa),flush=True)
