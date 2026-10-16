from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
import hashlib,json

work=Path(__file__).resolve().parent
P='http://schemas.openxmlformats.org/presentationml/2006/main'
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PKG='http://schemas.openxmlformats.org/package/2006/relationships'
CT='http://schemas.openxmlformats.org/package/2006/content-types'
def xml(t): return E.tostring(t,encoding='UTF-8',xml_declaration=True,standalone=True)
with ZipFile(work/'reference.pptx') as src,ZipFile(work/'authored.pptx') as art:
    parts={n:src.read(n) for n in src.namelist()}
    newnames=['ppt/slides/slide13.xml','ppt/slides/_rels/slide13.xml.rels','ppt/notesSlides/notesSlide13.xml','ppt/notesSlides/_rels/notesSlide13.xml.rels']
    for name in newnames:
        assert name not in parts
        parts[name]=art.read(name)
    rel=E.fromstring(parts['ppt/_rels/presentation.xml.rels'])
    rid='rIdTokenizerSummary13'
    assert rid not in [r.get('Id') for r in rel]
    E.SubElement(rel,'{'+PKG+'}Relationship',Id=rid,Type=R+'/slide',Target='slides/slide13.xml')
    parts['ppt/_rels/presentation.xml.rels']=xml(rel)
    pres=E.fromstring(parts['ppt/presentation.xml']); ids=pres.find('{'+P+'}sldIdLst')
    assert len(ids)==12
    E.SubElement(ids,'{'+P+'}sldId',{'id':str(max(int(x.get('id')) for x in ids)+1),'{'+R+'}id':rid})
    parts['ppt/presentation.xml']=xml(pres)
    ct=E.fromstring(parts['[Content_Types].xml'])
    authoredct=E.fromstring(art.read('[Content_Types].xml'))
    for name in newnames:
        if name.endswith('.rels'): continue
        entry=next(x for x in authoredct if x.get('PartName')=='/'+name)
        ct.append(entry)
    parts['[Content_Types].xml']=xml(ct)
    app=E.fromstring(parts['docProps/app.xml'])
    slides=app.find('{http://schemas.openxmlformats.org/officeDocument/2006/extended-properties}Slides')
    if slides is not None: slides.text='13'
    parts['docProps/app.xml']=xml(app)
    with ZipFile(work/'candidate.pptx','w') as out:
        for item in src.infolist(): out.writestr(item,parts.pop(item.filename))
        for name,value in parts.items(): out.writestr(name,value)
with ZipFile(work/'reference.pptx') as a,ZipFile(work/'candidate.pptx') as b:
    changed=[n for n in a.namelist() if a.read(n)!=b.read(n)]
    assert set(changed)<=set(['ppt/presentation.xml','ppt/_rels/presentation.xml.rels','[Content_Types].xml','docProps/app.xml'])
    fonts=sorted({el.get('typeface') for n in a.namelist() if n.startswith('ppt/slides/slide') and n.endswith('.xml') for el in E.fromstring(a.read(n)).iter() if el.get('typeface')})
receipt={'existing_slide_parts_byte_identical':True,'changed_package_entries':changed,'font_families':fonts,'reference_sha256':hashlib.sha256((work/'reference.pptx').read_bytes()).hexdigest()}
(work/'preservation.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
