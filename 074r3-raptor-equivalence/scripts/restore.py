# Upload results/checkpoint.tar.gz to /content/checkpoint.tar.gz, then execute
# this file BEFORE init.py on a replacement runtime. No model calls here.
import tarfile,pathlib
with tarfile.open('/content/checkpoint.tar.gz','r:gz') as tf:
 for member in tf.getmembers():
  p=pathlib.PurePosixPath(member.name)
  if '..' in p.parts or p.is_absolute() or p.parts[0]!='checkpoint':raise ValueError('invalid checkpoint member')
 tf.extractall('/content/restore',filter='data')
import shutil
shutil.copytree('/content/restore/checkpoint','/content/r3074',dirs_exist_ok=True)
print('RESTORED')
