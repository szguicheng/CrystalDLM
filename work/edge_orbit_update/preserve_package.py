"""Apply Artifact Tool-authored text while retaining the original PPT package."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E

root=Path('/Volumes/Extention/Users/guicheng/Code/helloWorld/CrystalDLM')
work=root/'work/edge_orbit_update'
ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
    'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
edited={}
with ZipFile(work/'reference.pptx') as source,ZipFile(work/'artifact_edit.pptx') as authored:
    old=E.fromstring(source.read('ppt/slides/slide11.xml'))
    new=E.fromstring(authored.read('ppt/slides/slide11.xml'))
    oldrow=old.find('.//a:tbl',ns).findall('a:tr',ns)[9]
    newrow=new.find('.//a:tbl',ns).findall('a:tr',ns)[9]
    for a,b in zip(oldrow.findall('a:tc',ns),newrow.findall('a:tc',ns)):
        a.replace(a.find('a:txBody',ns),deepcopy(b.find('a:txBody',ns)))
    def shape(tree,name):
        return next(s for s in tree.findall('.//p:sp',ns)
                    if s.find('p:nvSpPr/p:cNvPr',ns).get('name')==name)
    for name in ['definitions','footer']:
        a,b=shape(old,name),shape(new,name)
        a.replace(a.find('p:txBody',ns),deepcopy(b.find('p:txBody',ns)))
    edited['ppt/slides/slide11.xml']=E.tostring(old,encoding='UTF-8',xml_declaration=True,standalone=True)
    old=E.fromstring(source.read('ppt/notesSlides/notesSlide11.xml'))
    new=E.fromstring(authored.read('ppt/notesSlides/notesSlide11.xml'))
    def body(tree):
        return next(s for s in tree.findall('.//p:sp',ns)
                    if (s.find('p:nvSpPr/p:nvPr/p:ph',ns) is not None
                        and s.find('p:nvSpPr/p:nvPr/p:ph',ns).get('type')=='body'))
    a,b=body(old),body(new)
    a.replace(a.find('p:txBody',ns),deepcopy(b.find('p:txBody',ns)))
    edited['ppt/notesSlides/notesSlide11.xml']=E.tostring(old,encoding='UTF-8',xml_declaration=True,standalone=True)
    with ZipFile(work/'candidate.pptx','w') as target:
        for item in source.infolist():target.writestr(item,edited.get(item.filename,source.read(item.filename)))
with ZipFile(work/'reference.pptx') as a,ZipFile(work/'candidate.pptx') as b:
    assert set(a.namelist())==set(b.namelist())
    changed=[n for n in a.namelist() if a.read(n)!=b.read(n)]
    assert set(changed)==set(edited)
    families=sorted({n.get('typeface') for name in a.namelist() if name.startswith('ppt/slides/slide') and name.endswith('.xml')
                     for n in E.fromstring(a.read(name)).iter() if n.get('typeface')})
receipt={'changed_entries':changed,'all_other_entries_byte_identical':True,'font_families':families,
         'reference_sha256':hashlib.sha256((work/'reference.pptx').read_bytes()).hexdigest()}
(work/'preservation.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
