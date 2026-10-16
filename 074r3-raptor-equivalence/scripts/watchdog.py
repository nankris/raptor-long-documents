import time,subprocess,pathlib
p=pathlib.Path('/workspace/work/tig/074r3-raptor-equivalence/results')
time.sleep(9800)
if not (p/'stopped_at.txt').exists():
 r=subprocess.run(['bash','-lc','source /workspace/tools/setup.sh; colab stop -s r3074'],capture_output=True,text=True)
 (p/'watchdog.log').write_text(r.stdout+r.stderr)
 (p/'stopped_at.txt').write_text(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
