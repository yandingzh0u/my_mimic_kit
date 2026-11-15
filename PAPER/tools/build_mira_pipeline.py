#!/usr/bin/env python3
"""Editable MIRA overview following the supplied reference composition.

All cards, labels, arrows, neural-network glyphs and the reward curve are native
PowerPoint objects. Only the two genuine Unitree G1 renders are raster images.
Robot pictures are schematic icons; the experiment embodiments are unchanged.
"""
from __future__ import annotations
import json,math,subprocess,tempfile
from pathlib import Path
from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR,PP_ALIGN
from pptx.util import Inches,Pt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'PAPER/figures'
ASSET=OUT/'assets/g1'
NAME='mira_pipeline'
FONT='Times New Roman'
MATH='Cambria Math'
NAVY='73879B'; BLUE='96B2C9'; BLUE2='A0B9CE'; PALEBLUE='F1F7FC'
HEADERBLUE='E0EDF7'
GRAY='ABB9C9'; PALEGRAY='F3F5F8'; INK='263442'; WHITE='FFFFFF'
GREEN='E1EFDD'; PALEGREEN='F3F9F0'; CORAL='D9B0AB'; PALECORAL='FFF4F1'
NS='http://schemas.openxmlformats.org/drawingml/2006/main'

def rgb(x):return RGBColor.from_string(x)
def qn(n):return '{'+NS+'}'+n

def clear_style(s):
    for e in s._element.findall('{http://schemas.openxmlformats.org/presentationml/2006/main}style'):s._element.remove(e)

def fill_gradient(s,top,bottom):
    sp=s._element.spPr
    for tag in ['solidFill','noFill','gradFill']:
        old=sp.find(qn(tag))
        if old is not None:sp.remove(old)
    grad=etree.Element(qn('gradFill'),rotWithShape='1')
    stops=etree.SubElement(grad,qn('gsLst'))
    for position,color in [('0',top),('100000',bottom)]:
        gs=etree.SubElement(stops,qn('gs'),pos=position)
        etree.SubElement(gs,qn('srgbClr'),val=color)
    etree.SubElement(grad,qn('lin'),ang='5400000',scaled='1')
    sp.insert(1,grad)

def box(slide,x,y,w,h,fill=WHITE,stroke=GRAY,width=1.2,radius=.08,gradient=None):
    shape_type=MSO_SHAPE.RECTANGLE if radius==0 else MSO_SHAPE.ROUNDED_RECTANGLE
    s=slide.shapes.add_shape(shape_type,Inches(x),Inches(y),Inches(w),Inches(h))
    clear_style(s)
    if radius>0:s.adjustments[0]=radius
    s.fill.solid();s.fill.fore_color.rgb=rgb(fill)
    s.line.color.rgb=rgb(stroke);s.line.width=Pt(width)
    if gradient:fill_gradient(s,*gradient)
    return s

def rect(slide,x,y,w,h,fill):
    s=slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h))
    clear_style(s);s.fill.solid();s.fill.fore_color.rgb=rgb(fill);s.line.fill.background();return s

def label(slide,x,y,w,h,value,size=23,color=INK,bold=True,align=PP_ALIGN.CENTER,font=FONT,italic=False):
    return rich(slide,x,y,w,h,[(value,size,color,bold,italic,0,font)],align)

def rich(slide,x,y,w,h,runs,align=PP_ALIGN.CENTER):
    s=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
    tf=s.text_frame;tf.margin_top=tf.margin_bottom=tf.margin_left=tf.margin_right=0
    tf.vertical_anchor=MSO_ANCHOR.MIDDLE;tf.word_wrap=False
    p=tf.paragraphs[0];p.alignment=align;p.space_before=p.space_after=Pt(0)
    for value,size,color,bold,italic,baseline,font in runs:
        r=p.add_run();r.text=value;r.font.name=font;r.font.size=Pt(size*.88);r.font.bold=bold;r.font.italic=italic;r.font.color.rgb=rgb(color)
        if baseline:r._r.get_or_add_rPr().set('baseline',str(baseline))
    return s

def m(value,size=25,color=INK,bold=False,italic=True,baseline=0):
    return (value,size,color,bold,italic,baseline,MATH)

def t(value,size=24,color=INK,bold=True):
    return (value,size,color,bold,False,0,FONT)

def vector(slide,commands,color=NAVY,width=1.7,arrow=False,name='Vector path'):
    """A native PPT custom path, including editable Bezier control points."""
    # Coordinates are inches in slide space; path coordinates use local EMUs.
    pts=[(v[i],v[i+1]) for cmd,*v in commands for i in range(0,len(v),2)]
    x0=min(p[0] for p in pts);y0=min(p[1] for p in pts)
    x1=max(p[0] for p in pts);y1=max(p[1] for p in pts)
    w=max(x1-x0,.001);h=max(y1-y0,.001)
    s=slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(x0),Inches(y0),Inches(w),Inches(h));s.name=name
    clear_style(s);sp=s._element.spPr
    old=sp.find(qn('prstGeom'));sp.remove(old)
    geom=etree.Element(qn('custGeom'))
    for tag in ['avLst','gdLst','ahLst','cxnLst']:etree.SubElement(geom,qn(tag))
    etree.SubElement(geom,qn('rect'),l='0',t='0',r='r',b='b')
    paths=etree.SubElement(geom,qn('pathLst'))
    path=etree.SubElement(paths,qn('path'),w=str(Inches(w)),h=str(Inches(h)),fill='none',stroke='1',extrusionOk='0')
    def point(parent,x,y):etree.SubElement(parent,qn('pt'),x=str(round(Inches(x-x0))),y=str(round(Inches(y-y0))))
    for cmd,*v in commands:
        if cmd=='M':point(etree.SubElement(path,qn('moveTo')),*v)
        elif cmd=='L':point(etree.SubElement(path,qn('lnTo')),*v)
        elif cmd=='C':
            c=etree.SubElement(path,qn('cubicBezTo'))
            for i in range(0,6,2):point(c,v[i],v[i+1])
    sp.insert(1,geom);s.fill.background();s.line.color.rgb=rgb(color);s.line.width=Pt(width)
    ln=sp.find(qn('ln'))
    etree.SubElement(ln,qn('round'))
    if arrow:etree.SubElement(ln,qn('tailEnd'),type='triangle',w='med',len='med')
    return s

def edge(slide,points,color=NAVY,width=1.7,arrow=True):
    return vector(slide,[('M',*points[0])]+[('L',*p) for p in points[1:]],color,width,arrow,'Connection arrow')

def panel(slide,x,y,w,h,title,header=HEADERBLUE,body=PALEBLUE,title_size=28,header_h=.51,stroke=BLUE,square=False):
    box(slide,x,y,w,h,header,stroke,1.3,0 if square else .075)
    # Square panels use straight corners for both the header and body.
    box(slide,x+.008,y+header_h,w-.016,h-header_h-.008,body,stroke,1.05,0 if square else .085,gradient=(WHITE,body))
    label(slide,x+.10,y+.045,w-.20,header_h-.06,title,title_size,INK)

def neuron(slide,x,y,r=.14):
    s=slide.shapes.add_shape(MSO_SHAPE.OVAL,Inches(x-r),Inches(y-r),Inches(2*r),Inches(2*r))
    clear_style(s);s.fill.solid();s.fill.fore_color.rgb=rgb('F6F9FD');s.line.color.rgb=rgb(BLUE);s.line.width=Pt(1.25)

def make_slide(prs):
    slide=prs.slides.add_slide(prs.slide_layouts[6])
    # Rounded robot input cards, with authentic G1 meshes replacing the reference icon.
    for y,kind,title,s_hat in [(.18,'reference','Reference Motion (',True),(2.19,'simulated','Simulated Motion (',False)]:
        box(slide,.18,y,2.72,1.78,PALEGRAY,GRAY,1.25,.10)
        pic=slide.shapes.add_picture(str(ASSET/f'g1_{kind}.png'),Inches(.205),Inches(y+.37),width=Inches(2.67),height=Inches(1.365))
        pic.name=f'Unitree G1 — {kind} motion schematic'
        image_geom=pic._element.spPr.find(qn('prstGeom'));image_geom.set('prst','roundRect')
        av=image_geom.find(qn('avLst'));etree.SubElement(av,qn('gd'),name='adj',fmla='val 6000')
        # Picture crop is the same in both cards and never crops a body part.
        rich(slide,.25,y+.025,2.58,.31,[t(title,21.5),m('ŝ' if s_hat else 's',21.5),m('t',15,baseline=-23000),t(')',21.5)])

    # Physical representation card (selected examples only, as in supplied reference).
    box(slide,3.43,.54,1.94,2.84,'F7F8FB',GRAY,1.25,.09,gradient=(WHITE,PALEGRAY))
    label(slide,3.48,.64,1.84,.30,'Physical Errors',21.0)
    rich(slide,3.53,.94,1.74,.32,[m('e',25),m('t',17,baseline=-23000)])
    for y,name in [(1.35,'Body pos.'),(1.84,'Root rot.'),(2.80,'Joint vel.')]:
        b=box(slide,3.70,y,1.39,.38,'E4E9F1','E4E9F1',0,.42)
        label(slide,3.72,y+.012,1.35,.35,name,21.5)
    label(slide,3.95,2.28,.89,.36,'⋮',28,bold=False,font=MATH)
    box(slide,5.63,2.03,.87,.81,'ECF3F8',GRAY,1.25,.17,gradient=(WHITE,'E3EDF6'))
    label(slide,5.67,2.075,.79,.32,'Group',21.0)
    label(slide,5.67,2.385,.79,.32,'Norm.',21.0)

    # Equal-width sample cards share a vertical centerline and aligned edges.
    sample_x, sample_w = 6.72, 2.0
    panel(slide,sample_x,.64,sample_w,.91,'Positive Sample',GREEN,PALEGREEN,21,.42,stroke='B5CCAE')
    label(slide,sample_x+.10,1.08,sample_w-.20,.36,'0',29,bold=False,font=MATH)
    panel(slide,sample_x,1.985,sample_w,.91,'Negative Sample','FBE4DF',PALECORAL,20,.42,stroke=CORAL)
    rich(slide,sample_x+.10,2.425,sample_w-.20,.36,[m('z',26),m('t',17,baseline=-23000)])

    # Discriminator: conceptual glyph only, no layer dimensions or hyperparameters.
    panel(slide,9.07,.18,4.02,3.28,'',HEADERBLUE,PALEBLUE,28,.52,square=True)
    rich(slide,9.28,.245,3.60,.37,[t('MIRA Discriminator ',28),m('E',28),m('θ',18,baseline=-23000)])
    layers=[[(9.76,1.16),(9.76,1.62),(9.76,2.29)],[(10.79,1.16),(10.79,1.62),(10.79,2.29)],[(11.81,1.16),(11.81,1.62),(11.81,2.29)]]
    for left,right in zip(layers,layers[1:]):
        for a in left:
            for b in right:edge(slide,[a,b],'89A8C9',1.00,False)
    for a in layers[-1]:edge(slide,[a,(12.62,1.80)],BLUE,1.05,False)
    for layer in layers:
        for x,y in layer:neuron(slide,x,y)
        label(slide,layer[0][0]-.16,1.90,.32,.24,'⋮',20,bold=False,font=MATH)
    neuron(slide,12.62,1.80)
    box(slide,9.37,2.81,1.48,.45,'E6EFF8','E6EFF8',0,.44)
    box(slide,11.05,2.81,1.80,.45,'E6EFF8','E6EFF8',0,.44)
    rich(slide,9.43,2.835,1.36,.38,[m('W',27),m(' ≥ 0',27,italic=False)])
    rich(slide,11.11,2.835,1.68,.38,[m('‖',26,italic=False),m('W',26),m('‖',26,italic=False),m('2',17,italic=False,baseline=-23000),m(' ≤ 1',26,italic=False)])

    panel(slide,14.05,.55,3.08,2.84,'Softplus Reward',HEADERBLUE,PALEBLUE,27,.48)
    box(slide,14.28,1.16,2.62,.39,'F0F6FC','F0F6FC',0,.45)
    rich(slide,14.30,1.17,2.58,.37,[m('r',24),m('t',16,baseline=-23000),m(' = ',24,italic=False),t('softplus(',23,bold=False),m('s',24),m('θ',16,baseline=-23000),m('(',24,italic=False),m('z',24),m('t',16,baseline=-23000),m('))',24,italic=False)])
    # The softplus plot is a smooth native Bezier curve, not a screenshot.
    edge(slide,[(14.72,2.98),(16.83,2.98)],INK,1.75,True)
    edge(slide,[(14.91,3.17),(14.91,1.76)],INK,1.75,True)
    label(slide,14.62,3.005,.24,.27,'0',22,bold=False,font=MATH)
    label(slide,14.55,1.72,.24,.31,'r',25,bold=False,font=MATH,italic=True)
    label(slide,16.64,3.055,.24,.27,'s',25,bold=False,font=MATH,italic=True)
    vector(slide,[('M',14.96,2.96),('C',15.66,2.93,16.25,2.62,16.70,1.89)],'83A8CA',2.2,False,'Softplus reward curve')

    # Main inference and sample routes; all spatial relations follow the reference.
    vector(slide,[('M',2.90,1.13),('L',3.06,1.13),('C',3.17,1.13,3.18,1.20,3.18,1.29),('L',3.18,1.91),('C',3.18,2.05,3.25,2.07,3.43,2.07)],NAVY,1.9,True,'Reference error input')
    vector(slide,[('M',2.90,3.12),('L',3.05,3.12),('C',3.16,3.12,3.18,3.05,3.18,2.94),('L',3.18,2.23),('C',3.18,2.10,3.23,2.07,3.43,2.07)],NAVY,1.9,False,'Simulation error input')
    edge(slide,[(5.37,2.44),(5.63,2.44)],NAVY,1.8)
    edge(slide,[(6.50,2.44),(sample_x,2.44)],NAVY,1.8)
    edge(slide,[(sample_x+sample_w,1.095),(9.07,1.095)],NAVY,1.9)
    edge(slide,[(sample_x+sample_w,2.44),(9.07,2.44)],NAVY,1.9)
    edge(slide,[(13.09,2.02),(14.05,2.02)],NAVY,1.9)
    rich(slide,13.16,1.61,.80,.32,[m('s',25),m('θ',16,baseline=-23000),m('(',25,italic=False),m('z',25),m('t',16,baseline=-23000),m(')',25,italic=False)])

    # Bottom return loop: reward -> PPO -> policy -> simulation -> simulated motion.
    box(slide,3.43,4.32,3.82,.71,PALEGRAY,GRAY,1.25,.15,gradient=(WHITE,'DCE3EC'))
    label(slide,3.50,4.405,3.68,.48,'Physics Simulation',29)
    box(slide,8.40,4.32,2.94,.71,PALEGRAY,GRAY,1.25,.15,gradient=(WHITE,'E3E8F0'))
    rich(slide,8.54,4.405,2.66,.48,[t('Policy ',29),m('π',29)])
    box(slide,12.50,4.32,2.28,.71,'E7F2FC',BLUE2,1.25,.18,gradient=('F1F8FE','DCECF9'))
    label(slide,12.66,4.405,1.96,.48,'PPO',31)
    edge(slide,[(8.40,4.675),(7.25,4.675)],NAVY,1.9)
    edge(slide,[(12.50,4.675),(11.34,4.675)],NAVY,1.9)
    vector(slide,[('M',17.13,2.08),('L',17.30,2.08),('C',17.41,2.08,17.43,2.17,17.43,2.28),('L',17.43,4.48),('C',17.43,4.61,17.36,4.675,17.23,4.675),('L',14.78,4.675)],NAVY,1.9,True,'Reward to PPO')
    vector(slide,[('M',3.43,4.675),('L',1.72,4.675),('C',1.60,4.675,1.55,4.61,1.55,4.49),('L',1.55,3.97)],NAVY,1.9,True,'Simulation feedback')
    slide.notes_slide.notes_text_frame.text=(
      'Composition follows the user-provided Fig. 1 reference. Genuine Unitree G1 29DOF '
      'meshes and a G1 walking pose were rendered locally with MuJoCo OSMesa on CPU; '
      'these are schematic robot illustrations, not evidence of G1 training. Experiment '
      'robot and checkpoints are unchanged. Source and license: figures/assets/g1/provenance.json '
      'and UNITREE_LICENSE.txt. Every other object is native editable PowerPoint geometry '
      'or text; the neural glyph is conceptual, not a declaration of hidden widths. '
      'Actual score is anchored as defined in the method; only its symbol is shown here.')
    return slide

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for kind in ['reference','simulated']:
        assert (ASSET/f'g1_{kind}.png').exists(),'Run render_mira_g1_icons.py using env_isaaclab Python first.'
    prs=Presentation();prs.slide_width=Inches(17.6);prs.slide_height=Inches(5.15)
    prs.core_properties.title='MIRA method overview'
    prs.core_properties.subject='Reference-layout method diagram with genuine Unitree G1 robot illustrations'
    prs.core_properties.author='MIRA'
    slide=make_slide(prs)
    ppt=OUT/f'{NAME}.pptx';prs.save(ppt)
    with tempfile.TemporaryDirectory(prefix='mira_method_reference_') as td:
        td=Path(td)
        r=subprocess.run(['libreoffice','-env:UserInstallation=file://'+str(td/'lo'),'--headless','--convert-to','pdf','--outdir',str(td),str(ppt)],check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        import fitz
        doc=fitz.open(td/f'{NAME}.pdf');assert len(doc)==1
        doc.save(OUT/f'{NAME}.pdf',garbage=4,deflate=True)
        doc[0].get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(OUT/f'{NAME}.png')
    summary={'slides':1,'page_inches':[17.6,5.15],'paper_inches_at_7in_width':[7,7*5.15/17.6],'g1_robot_images':2,'editable_native_elements':len(slide.shapes)-2,'reference_composition_followed':True,'robot_provenance':str(ASSET/'provenance.json'),'files':[str(OUT/f'{NAME}.{ext}') for ext in ['pptx','pdf','png']]}
    (ASSET/'pipeline_validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
