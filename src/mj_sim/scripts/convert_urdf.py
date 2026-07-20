#!/usr/bin/env python3
"""URDF→MJCF converter for cubot robot. Run after xacro generation."""
import xml.etree.ElementTree as ET, os, math

# Paths resolved via ament or relative to this script
try:
    from ament_index_python.packages import get_package_share_directory
    cubot_share = get_package_share_directory("cubot_description")
    mj_share = get_package_share_directory("mj_sim")
except ImportError:
    # Fallback: paths relative to workspace root
    ws = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    cubot_share = os.path.join(ws, "install", "cubot_description", "share", "cubot_description")
    mj_share = os.path.join(ws, "install", "mj_sim", "share", "mj_sim")

urdf = os.path.join(cubot_share, "cubot_urdf", "urdf", "cubot.urdf")
md = os.path.join(cubot_share, "cubot_urdf", "meshes")
out = os.path.join(mj_share, "models", "cubot.xml")
# During development, write to source tree
src_out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "cubot.xml")
if os.path.exists(os.path.dirname(src_out)):
    out = src_out

tree = ET.parse(urdf); root = tree.getroot()
ms=set()
for m in root.iter('mesh'):
    f=os.path.basename(m.get('filename',''))
    if f: ms.add(f)

Lks={}
for l in root.iter('link'): Lks[l.get('name')]={'i':l.find('inertial'),'v':l.findall('visual'),'c':l.findall('collision')}

Js=[]
for j in root.iter('joint'):
    p=j.find('parent').get('link'); c=j.find('child').get('link')
    o=j.find('origin'); x=o.get('xyz','0 0 0') if o is not None else '0 0 0'
    r=o.get('rpy','0 0 0') if o is not None else '0 0 0'
    x=' '.join(str(float(v)) for v in x.split()); r=' '.join(str(float(v)) for v in r.split())
    a=j.find('axis'); ax=a.get('xyz','1 0 0') if a is not None else '1 0 0'
    lm=j.find('limit')
    dyn=j.find('dynamics')
    damp=dyn.get('damping','0') if dyn is not None else '0'
    fric=dyn.get('friction','0') if dyn is not None else '0'
    Js.append({'p':p,'c':c,'x':x,'r':r,'a':ax,'t':j.get('type'),'n':j.get('name',''),
        'lo':lm.get('lower','0') if lm is not None else '0','hi':lm.get('upper','0') if lm is not None else '0',
        'ef':lm.get('effort','0') if lm is not None else '0','damp':damp,'fric':fric})
ch={}
for j in Js: ch.setdefault(j['p'],[]).append(j)
roots=set(j['p'] for j in Js)-set(j['c'] for j in Js)

type_map = {'revolute':'hinge','continuous':'hinge','prismatic':'slide','fixed':None}

def rq(r):
    x,y,z=[float(v) for v in r.split()]; cx,sx=math.cos(x/2),math.sin(x/2); cy,sy=math.cos(y/2),math.sin(y/2); cz,sz=math.cos(z/2),math.sin(z/2)
    return f'{cx*cy*cz+sx*sy*sz:.6g} {sx*cy*cz-cx*sy*sz:.6g} {cx*sy*cz+sx*cy*sz:.6g} {cx*cy*sz-sx*sy*cz:.6g}'
def hs(s): return ' '.join(str(float(v)/2) for v in s.split())

L=['<mujoco model="cubot">','  <compiler angle="radian"/>','','  <asset>']
for m in sorted(ms): n=m.replace('.STL','').replace('.stl',''); L.append(f'    <mesh name="{n}" file="{md}/{m}"/>')
L+=['  </asset>','','  <worldbody>','    <body name="root" pos="0 0 0">','      <freejoint/>']

def wb(name,indent):
    if name not in Lks or name=='base_footprint': return []
    lk=Lks[name]; pfx=' '*indent; out=[]
    ji=next((j for j in Js if j['c']==name),None)
    at=''
    if ji: at=f' pos="{ji["x"]}"'
    if ji and ji['r']!='0.0 0.0 0.0': at+=f' quat="{rq(ji["r"])}"'
    out.append(f'{pfx}<body name="{name}"{at}>')
    if lk['i'] is not None:
        inv=lk['i']; ma=inv.find('mass').get('value')
        ixx,iyy,izz=inv.find('inertia').get('ixx'),inv.find('inertia').get('iyy'),inv.find('inertia').get('izz')
        o=inv.find('origin'); ip=o.get('xyz','0 0 0') if o is not None else '0 0 0'
        out.append(f'{pfx}  <inertial pos="{ip}" mass="{ma}" diaginertia="{ixx} {iyy} {izz}"/>')
    if ji and ji['t']!='fixed':
        jt = type_map.get(ji['t'], 'hinge')
        out.append(f'{pfx}  <joint name="{ji["n"]}" type="{jt}" axis="{ji["a"]}" range="{ji["lo"]} {ji["hi"]}" actuatorfrcrange="-{ji["ef"]} {ji["ef"]}" damping="{ji["damp"]}" frictionloss="{ji["fric"]}"/>')
    for vis in lk['v']:
        mesh=vis.find('.//mesh'); geom=vis.find('.//geometry'); o=vis.find('origin')
        pos=o.get('xyz','0 0 0') if o is not None else '0 0 0'
        rpy=[float(v) for v in (o.get('rpy','0 0 0') if o is not None else '0 0 0').split()]
        if mesh is not None:
            f=os.path.basename(mesh.get('filename','')); mn=f.replace('.STL','').replace('.stl','')
            q=f' quat="{rq(" ".join(str(v) for v in rpy))}"' if rpy!=[0,0,0] else ''
            out.append(f'{pfx}  <geom type="mesh" mesh="{mn}" pos="{pos}"{q} rgba="0.4 0.4 0.4 1"/>')
        elif geom is not None and len(geom)>0:
            g=geom[0]
            if g.tag=='cylinder': r=g.get('radius','0'); l=g.get('length','0'); out.append(f'{pfx}  <geom type="cylinder" size="{r} {float(l)/2}" pos="{pos}" rgba="0.3 0.3 0.3 1"/>')
            elif g.tag=='box': s=hs(g.get('size','0 0 0')); out.append(f'{pfx}  <geom type="box" size="{s}" pos="{pos}" rgba="0.4 0.4 0.4 1"/>')
    for col in lk['c']:
        geom=col.find('.//geometry'); o=col.find('origin')
        pos=o.get('xyz','0 0 0') if o is not None else '0 0 0'
        if geom is not None and len(geom)>0:
            g=geom[0]
            if g.tag=='cylinder': r=g.get('radius','0'); l=g.get('length','0'); out.append(f'{pfx}  <geom type="cylinder" size="{r} {float(l)/2}" pos="{pos}" group="3" rgba="0 0 0 0"/>')
            elif g.tag=='box': s=hs(g.get('size','0 0 0')); out.append(f'{pfx}  <geom type="box" size="{s}" pos="{pos}" group="3" rgba="0 0 0 0"/>')
    if name in ch:
        for cj in ch[name]: out.extend(wb(cj['c'],indent+2))
    out.append(f'{pfx}</body>')
    return out

for rl in roots:
    if rl=='base_footprint' and rl in ch:
        for cj in ch[rl]: L.extend(wb(cj['c'],6))
    elif rl!='base_footprint': L.extend(wb(rl,6))
L+=['    </body>','  </worldbody>']
    
# Generate actuators for all non-fixed joints
all_joints = []
for jn in Js:
    if jn['t'] != 'fixed':
        all_joints.append(jn)
if all_joints:
    L.append('  <actuator>')
    for jn in all_joints:
        jt = jn['t']
        kp = '5.0' if jt in ('revolute','continuous') else '2.0'
        kv = str(round(float(kp) * 0.15, 2))
        ctrl = ' ctrlrange="0 0.08"' if jt in ('slide','prismatic') else ''
        L.append(f'    <position name="a_{jn["n"]}" joint="{jn["n"]}" kp="{kp}" kv="{kv}"{ctrl}/>')
    L.append("  </actuator>")
    
    L.append('</mujoco>')
with open(out,'w') as f: f.write('\n'.join(L))
c='\n'.join(L)
print(f"Converted: {c.count('type=\"mesh\"')} mesh, {c.count('type=\"box\"')} box, {c.count('<mesh name=')} assets → {out}")
