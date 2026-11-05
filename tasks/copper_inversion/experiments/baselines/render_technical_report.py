"""Render the repository report as an offline HTML document on the remote host."""
from pathlib import Path
import base64
import re
import markdown

root=Path('/Aidata/shared/ldm/copper-technical-report-20261024')
source=(root/'report.md').read_text()
equation='<div class="equation">L<sub>o</sub> = w<sub>o</sub> [ Σ<sub>y=0</sub> softplus(φ) + Σ<sub>y&gt;0</sub> softplus(−φ) + mean(φ<sub>C</sub>)² + mean(φ<sub>C</sub>²) ]</div>'
source=re.sub(r'\\\[.*?\\\]',lambda _:equation,source,flags=re.S)
body=markdown.markdown('[TOC]\n\n'+source,extensions=['tables','fenced_code','toc'],output_format='html')
def embed(match):
    path=root/match.group(1)
    if path.exists() and path.suffix=='.png':
        return 'src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'"'
    return match.group(0)
body=re.sub(r'src="([^"]+)"',embed,body)
style='''
:root {color-scheme:light;--ink:#16313e;--accent:#087e8b;--line:#dbe5e8}
*{box-sizing:border-box}body{margin:0;color:var(--ink);background:#edf3f4;font-family:system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;line-height:1.8}
main{max-width:1180px;margin:36px auto;padding:52px 64px;background:white;box-shadow:0 8px 40px #16313e12;border-top:7px solid var(--accent)}
h1{font-size:34px;line-height:1.35;margin:0 0 24px}h2{font-size:25px;margin-top:48px;padding-bottom:10px;border-bottom:2px solid var(--line)}h3{font-size:20px;margin-top:30px}
a{color:var(--accent);text-underline-offset:3px}p,li{font-size:16px}strong{color:#102b36}img{width:100%;height:auto;border:1px solid var(--line);margin:12px 0}
table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.6;margin:24px 0}th{background:#e4f1f2;text-align:left}th,td{padding:10px;border:1px solid var(--line)}tr:nth-child(even){background:#f8fafb}
blockquote{margin:24px 0;padding:14px 22px;border-left:4px solid var(--accent);background:#edf7f8}code{font-size:13px;background:#f0f4f6;padding:2px 4px;overflow-wrap:anywhere}.equation{font-family:serif;font-size:19px;padding:24px;background:#f0f5f6;overflow-x:auto}
.toc{border:1px solid var(--line);padding:18px 28px;margin-bottom:32px;background:#f8fafb}.toc ul{margin:4px 0;padding-left:20px}.toc li{font-size:13px}.toc>ul>li>ul{columns:2}.toc a{text-decoration:none}
@media(max-width:800px){main{margin:0;padding:25px 18px}.toc>ul>li>ul{columns:1}table{display:block;overflow-x:auto}h1{font-size:28px}}
@media print{body{background:white}main{box-shadow:none;margin:0;padding:0;max-width:none}.toc{display:none}h2{break-before:page}img,table{break-inside:avoid}a{color:inherit}}
'''
(root/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LDM 铜矿几何反演技术报告</title><style>'+style+'</style><main>'+body+'</main></html>')
assert '<!-- HOLDOUT_RESULTS -->' not in source
assert 'data:image/png;base64,' in (root/'report.html').read_text()
print('Rendered offline HTML with embedded static figures')
