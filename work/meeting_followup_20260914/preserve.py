from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E
import json,hashlib,posixpath as pp

work=Path(__file__).resolve().parent
P='http://schemas.openxmlformats.org/presentationml/2006/main'
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PKG='http://schemas.openxmlformats.org/package/2006/relationships'
CT='http://schemas.openxmlformats.org/package/2006/content-types'
def xml(e):return E.tostring(e,encoding='UTF-8',xml_declaration=True,standalone=True)
def relpath(n):return pp.join(pp.dirname(n),'_rels',pp.basename(n)+'.rels')

with ZipFile(work/'reference.pptx') as src,ZipFile(work/'authored.pptx') as art:
 parts={n:src.read(n) for n in src.namelist()}
 artnames=set(art.namelist())
 manifest=json.loads((work/'manifest.json').read_text())
 mappings={f'ppt/slides/slide{i}.xml':f'ppt/slides/slide{i}.xml' for i in range(16,manifest['total']+1)}
 done=set()
 def copy_part(old):
  if old not in mappings:
   mappings[old]=pp.join(pp.dirname(old),'followup20260914_'+pp.basename(old))
  new=mappings[old]
  if old in done:return new
  done.add(old)
  assert new not in parts,new
  parts[new]=art.read(old)
  rp=relpath(old)
  if rp in artnames:
   e=E.fromstring(art.read(rp))
   for rel in e:
    if rel.get('TargetMode')=='External':continue
    target=rel.get('Target')
    oldtarget=target[1:] if target.startswith('/') else pp.normpath(pp.join(pp.dirname(old),target))
    newtarget=copy_part(oldtarget)
    rel.set('Target',pp.relpath(newtarget,pp.dirname(new)))
   parts[relpath(new)]=xml(e)
  return new
 for i in range(16,manifest['total']+1):copy_part(f'ppt/slides/slide{i}.xml')
 pres=E.fromstring(parts['ppt/presentation.xml']);rels=E.fromstring(parts['ppt/_rels/presentation.xml.rels'])
 ids=pres.find('{'+P+'}sldIdLst');assert len(ids)==15
 for i in range(16,manifest['total']+1):
  rid=f'rIdFollowup20260914Slide{i}'
  E.SubElement(rels,'{'+PKG+'}Relationship',Id=rid,Type=R+'/slide',Target=f'slides/slide{i}.xml')
  E.SubElement(ids,'{'+P+'}sldId',{'id':str(max(int(x.get('id')) for x in ids)+1),'{'+R+'}id':rid})
 # Register copied master parts without modifying the original masters/layouts.
 for typ,tag,child in [('slideMaster','sldMasterIdLst','sldMasterId'),('notesMaster','notesMasterIdLst','notesMasterId')]:
  owners=[new for old,new in mappings.items() if pp.dirname(old)==('ppt/slideMasters' if typ=='slideMaster' else 'ppt/notesMasters') and old.endswith('.xml')]
  parent=pres.find('{'+P+'}'+tag)
  if parent is None and owners:parent=E.SubElement(pres,'{'+P+'}'+tag)
  for k,new in enumerate(owners):
   rid=f'rIdFollowup20260914{typ}{k}'
   E.SubElement(rels,'{'+PKG+'}Relationship',Id=rid,Type=R+'/'+typ,Target=pp.relpath(new,'ppt'))
   attrs={'{'+R+'}id':rid}
   if typ=='slideMaster':attrs['id']=str(max([int(x.get('id')) for x in parent]+[2147483647])+1)
   E.SubElement(parent,'{'+P+'}'+child,attrs)
 parts['ppt/presentation.xml']=xml(pres);parts['ppt/_rels/presentation.xml.rels']=xml(rels)
 ct=E.fromstring(parts['[Content_Types].xml']);act=E.fromstring(art.read('[Content_Types].xml'))
 overrides={x.get('PartName').lstrip('/'):x.get('ContentType') for x in act if x.tag.endswith('Override')}
 defs={x.get('Extension'):x.get('ContentType') for x in act if x.tag.endswith('Default')}
 have={x.get('Extension') for x in ct if x.tag.endswith('Default')}
 for old,new in mappings.items():
  if old in overrides:E.SubElement(ct,'{'+CT+'}Override',PartName='/'+new,ContentType=overrides[old])
  else:
   ext=pp.splitext(new)[1][1:]
   if ext not in have:E.SubElement(ct,'{'+CT+'}Default',Extension=ext,ContentType=defs[ext]);have.add(ext)
 parts['[Content_Types].xml']=xml(ct)
 app=E.fromstring(parts['docProps/app.xml'])
 for el in app.iter():
  if el.tag.endswith('}Slides'):el.text=str(manifest['total'])
 parts['docProps/app.xml']=xml(app)
 with ZipFile(work/'candidate.pptx','w',ZIP_DEFLATED) as out:
  for name,data in parts.items():out.writestr(name,data)
 changed=[n for n in src.namelist() if src.read(n)!=parts[n]]
 assert set(changed)<=set(['ppt/presentation.xml','ppt/_rels/presentation.xml.rels','[Content_Types].xml','docProps/app.xml'])
 (work/'preservation.json').write_text(json.dumps({'original_slides':15,'total_slides':manifest['total'],'original_slide_parts_byte_identical':True,'changed_original_parts':changed,'reference_sha256':hashlib.sha256((work/'reference.pptx').read_bytes()).hexdigest()},indent=2))
 print('Preserved all original slide/dependency bytes; appended',manifest['total']-15,'slides; copied',len(done),'dependency parts.')
