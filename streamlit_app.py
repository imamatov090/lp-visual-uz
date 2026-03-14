import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
from itertools import combinations
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph,
                                 Spacer, Image, Table, TableStyle)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

st.set_page_config(page_title="ЛП — Решатель", page_icon="📊", layout="wide")

st.markdown("""
<style>
.title{text-align:center;font-size:1.8rem;font-weight:800;
       color:#1a3a5c;margin-bottom:1rem}
.step{background:#f0f7ff;border-left:4px solid #2980b9;
      padding:8px 12px;border-radius:6px;margin-bottom:6px;font-size:0.9rem}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="title">📊 Линейное программирование — Решатель</div>',
    unsafe_allow_html=True)

# ── Алгоритм ─────────────────────────────────────────────────

def pf(s):
    return float(str(s).replace(',', '.').strip())

def intersect(a1, b1, c1, a2, b2, c2):
    d = a1*b2 - a2*b1
    if abs(d) < 1e-10:
        return None
    return ((c1*b2 - c2*b1)/d, (a1*c2 - a2*c1)/d)

def feasible(x, y, cons):
    for a, b, s, c in cons:
        v = a*x + b*y
        if s == '<=' and v > c + 1e-8: return False
        if s == '>=' and v < c - 1e-8: return False
        if s == '='  and abs(v-c) > 1e-8: return False
    return True

def get_corners(cons):
    pts = []
    for i, j in combinations(range(len(cons)), 2):
        a1,b1,_,c1 = cons[i]
        a2,b2,_,c2 = cons[j]
        p = intersect(a1,b1,c1,a2,b2,c2)
        if p is None:
            continue
        x, y = p
        if feasible(x, y, cons):
            if not any(abs(cx-x)<1e-6 and abs(cy-y)<1e-6 for cx,cy in pts):
                pts.append((round(x,6), round(y,6)))
    return pts

def get_optimum(pts, c1, c2, ot):
    if not pts:
        return None, None
    vals = [c1*x + c2*y for x,y in pts]
    i = int(np.argmax(vals)) if ot == 'max' else int(np.argmin(vals))
    return pts[i], vals[i]

def sort_poly(pts):
    if len(pts) < 3:
        return pts
    cx = sum(p[0] for p in pts)/len(pts)
    cy = sum(p[1] for p in pts)/len(pts)
    return sorted(pts, key=lambda p: np.arctan2(p[1]-cy, p[0]-cx))

def make_fig(cons, c1, c2, ot, cpts, opt, oval, astep=None, tsteps=40):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor('#FAFAFA')
    COLS = ['#1f77b4','#d62728','#2ca02c','#9467bd',
            '#8c564b','#e377c2','#17becf','#bcbd22']

    if cpts:
        xs = [p[0] for p in cpts]; ys = [p[1] for p in cpts]
        mg = max(5, (max(xs)-min(xs))*0.6+3, (max(ys)-min(ys))*0.6+3)
        mx, my = (max(xs)+min(xs))/2, (max(ys)+min(ys))/2
    else:
        mg, mx, my = 8, 0, 0

    xl, xr = mx-mg, mx+mg
    yl, yr = my-mg, my+mg
    xv = np.linspace(xl, xr, 600)

    for idx, (a, b, s, c) in enumerate(cons):
        col = COLS[idx % len(COLS)]
        lbl = f'{a}*x + {b}*y {s} {c}'
        if abs(b) > 1e-10:
            ax.plot(xv, (c-a*xv)/b, color=col, lw=1.8, label=lbl)
        elif abs(a) > 1e-10:
            ax.axvline(c/a, color=col, lw=1.8, label=lbl)

    if len(cpts) >= 3:
        ax.add_patch(MplPolygon(sort_poly(cpts), closed=True,
                                facecolor='#AED6F1', edgecolor='none',
                                alpha=0.35, zorder=2, label='ОДР'))
    if cpts:
        ax.scatter([p[0] for p in cpts], [p[1] for p in cpts],
                   color='red', s=65, zorder=6,
                   edgecolors='darkred', lw=1, label='Угловые точки')

    if opt:
        ox, oy = opt
        Cf = c1*ox + c2*oy
        if astep is not None and len(cpts) >= 2:
            x3 = (cpts[0][0]+cpts[1][0])/2
            y3 = (cpts[0][1]+cpts[1][1])/2
            Ci = c1*x3 + c2*y3
            t  = min(astep, tsteps)/tsteps
            Cc = Ci + (Cf-Ci)*t
        else:
            Cc = Cf

        if abs(c2) > 1e-10:
            lbl2 = (f'Линия уровня Z={Cc:.2f}' if astep is not None
                    else f'Целевая прямая: {c1}*x+{c2}*y={Cc:.2f}')
            ax.plot(xv, (Cc-c1*xv)/c2, 'k--', lw=1.8,
                    alpha=0.8, label=lbl2, zorder=5)

        if astep is None:
            nm = np.sqrt(c1**2 + c2**2)
            if nm > 1e-10:
                sc = mg*0.18
                dx, dy = c1/nm*sc, c2/nm*sc
                ax.annotate('', xy=(ox+dx, oy+dy), xytext=(ox, oy),
                            arrowprops=dict(arrowstyle='->',
                                            color='darkred', lw=2.2), zorder=9)
                ax.text(ox+dx*1.2, oy+dy*1.2, '∇Z', fontsize=11,
                        color='darkred', fontweight='bold', zorder=9)

        ax.scatter([ox], [oy], color='gold', s=240, zorder=10, marker='*',
                   edgecolors='#8B6914', lw=1.2, label='Оптимум')
        ax.annotate(f'Оптимум ({ox:.2f}; {oy:.2f})', (ox, oy),
                    xytext=(ox+mg*0.18, oy+mg*0.18),
                    fontsize=9, color='#1A5276', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.25',
                              facecolor='#EBF5FB', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='#1A5276', lw=1.2),
                    zorder=11)

    ax.set_xlim(xl, xr); ax.set_ylim(yl, yr)
    ax.axhline(0, color='black', lw=0.8)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('x', fontsize=12); ax.set_ylabel('y', fontsize=12)
    ax.set_title('График решения', fontsize=13, pad=10)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7.5, loc='upper right', framealpha=0.93)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig); buf.seek(0)
    return buf.read()

def make_pdf(cons, c1, c2, ot, cpts, opt, oval, imgb):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    ts = ParagraphStyle('T', parent=styles['Title'], fontSize=14,
                        alignment=TA_CENTER, spaceAfter=8,
                        fontName='Helvetica-Bold')
    hs = ParagraphStyle('H', parent=styles['Heading2'], fontSize=11,
                        spaceBefore=6, spaceAfter=4, fontName='Helvetica-Bold')
    bs = ParagraphStyle('B', parent=styles['Normal'],
                        fontSize=10, spaceAfter=3)
    arr = 'max' if ot == 'max' else 'min'
    story = [Paragraph('Линейное программирование — Решатель', ts),
             Spacer(1, 0.2*cm),
             Paragraph(f'<b>Целевая функция:</b> Z = {c1}*x + ({c2})*y → {arr}', bs),
             Paragraph('<b>Ограничения:</b>', hs)]
    for a, b, s, c in cons:
        story.append(Paragraph(f'&nbsp;&nbsp;{a}*x + ({b})*y {s} {c}', bs))
    ox, oy = opt
    story += [Spacer(1, 0.2*cm),
              Paragraph('<b>Результат:</b>', hs),
              Paragraph(f'x* = {ox:.4f},  y* = {oy:.4f}', bs),
              Paragraph(f'Z* = {oval:.4f}', bs),
              Spacer(1, 0.2*cm),
              Paragraph('<b>Угловые точки:</b>', hs)]
    td = [['№','x','y','Z']]
    for i, (cx, cy) in enumerate(cpts):
        zv = c1*cx + c2*cy
        m = ' ← оптимум' if abs(cx-ox)<1e-4 and abs(cy-oy)<1e-4 else ''
        td.append([str(i+1), f'{cx:.4f}', f'{cy:.4f}', f'{zv:.4f}{m}'])
    t = Table(td, colWidths=[1.2*cm,4*cm,4*cm,6.5*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#2980B9')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),
         [colors.white, colors.HexColor('#EBF5FB')]),
        ('GRID',(0,0),(-1,-1),0.5,colors.grey),
        ('FONTSIZE',(0,0),(-1,-1),9),
    ]))
    story += [t, Spacer(1,0.3*cm),
              Paragraph('<b>График:</b>', hs),
              Image(io.BytesIO(imgb), width=14*cm, height=11*cm)]
    doc.build(story); buf.seek(0)
    return buf.read()

# ── Session state ─────────────────────────────────────────────
if 'n'   not in st.session_state: st.session_state.n   = 5
if 'res' not in st.session_state: st.session_state.res = None

DEFAULTS = [(3.2,-2.0,'=',3.0), (1.6,2.3,'<=',-5.0),
            (3.2,-6.0,'>=',7.0), (7.0,-2.0,'<=',10.0), (-6.5,3.0,'<=',9.0)]

# ── Layout ────────────────────────────────────────────────────
LEFT, RIGHT = st.columns([1, 1.7], gap="large")

with LEFT:
    # Целевая функция
    st.subheader("Целевая функция")
    f1, f2, f3 = st.columns([2, 2, 1])
    with f1: c1s  = st.text_input("c₁  * x  +", value="5,3",  key="c1")
    with f2: c2s  = st.text_input("c₂  * y  →", value="-7,1", key="c2")
    with f3: otype = st.selectbox("Тип", ["max","min"],        key="ot")

    # Ограничения
    st.subheader("Ограничения")
    btn1, btn2 = st.columns(2)
    with btn1:
        if st.button("+ Добавить", use_container_width=True):
            st.session_state.n += 1; st.rerun()
    with btn2:
        if st.button("− Удалить", use_container_width=True,
                     disabled=st.session_state.n <= 1):
            st.session_state.n -= 1; st.rerun()

    cons_raw = []
    for i in range(st.session_state.n):
        da, db, ds, dc = DEFAULTS[i] if i < len(DEFAULTS) else (1.0,1.0,'<=',0.0)
        ca, cb, cs, cc = st.columns([2, 2, 1, 2])
        with ca: a    = st.text_input("a · x  +", value=str(da), key=f"row_a_{i}")
        with cb: b    = st.text_input("b · y",    value=str(db), key=f"row_b_{i}")
        with cs:
            sign_labels = ["≤  не более", "≥  не менее", "=  равно"]
            sign_values = ["<=", ">=", "="]
            chosen = st.selectbox("Знак", sign_labels,
                                  index=sign_values.index(ds), key=f"row_s_{i}")
            sign = sign_values[sign_labels.index(chosen)]
        with cc: c    = st.text_input("Правая часть", value=str(dc), key=f"row_c_{i}")
        cons_raw.append((a, b, sign, c))

    st.caption("Коэффициенты: целые или дробные (запятая/точка).")
    st.divider()

    solve_btn = st.button("✅  Решить",   type="primary", use_container_width=True)
    clear_btn = st.button("🗑  Очистить", use_container_width=True)

    if clear_btn:
        st.session_state.n   = 5
        st.session_state.res = None
        st.rerun()

    # Результат
    r = st.session_state.res
    if r:
        if r['status'] == 'error':
            st.error(r['msg'])
        else:
            ox, oy = r['opt']
            st.success(
                f"**x\\* = {ox:.4f},  y\\* = {oy:.4f}**\n\n"
                f"Z\\* = {r['oval']:.4f}  ({r['otype']})")
            cpts = r['cpts']
            x3 = (cpts[0][0]+cpts[1][0])/2 if len(cpts)>=2 else 0
            y3 = (cpts[0][1]+cpts[1][1])/2 if len(cpts)>=2 else 0
            Ci = r['c1']*x3 + r['c2']*y3
            with st.expander("📋 Алгоритм (шаги)", expanded=True):
                st.markdown(f"""
<div class="step"><b>Шаг 0 — ОДР:</b> {len(cpts)} угловых точек:<br>
{', '.join(f'({x:.3f}; {y:.3f})' for x,y in cpts)}</div>
<div class="step"><b>Шаг 1 — Внутренняя точка:</b>
x₃ = {x3:.4f},  y₃ = {y3:.4f}</div>
<div class="step"><b>Шаг 2 — Линия уровня:</b> C = {Ci:.4f}</div>
<div class="step"><b>Шаг 3 — Оптимум:</b>
x* = {ox:.4f},  y* = {oy:.4f},  Z* = {r['oval']:.4f}</div>
""", unsafe_allow_html=True)
            pdf = make_pdf(r['cons'], r['c1'], r['c2'], r['otype'],
                           r['cpts'], r['opt'], r['oval'], r['img'])
            st.download_button("📄 Скачать PDF", data=pdf,
                               file_name="lp_report.pdf",
                               mime="application/pdf",
                               use_container_width=True)

# ── Решение ──────────────────────────────────────────────────
if solve_btn:
    try:
        c1   = pf(c1s); c2 = pf(c2s)
        cons = [(pf(a), pf(b), s, pf(c)) for a,b,s,c in cons_raw]
        cpts = get_corners(cons)
        if not cpts:
            st.session_state.res = {
                'status': 'error',
                'msg': 'ОДР пустое — система ограничений несовместна.'}
        else:
            opt, oval = get_optimum(cpts, c1, c2, otype)
            img = make_fig(cons, c1, c2, otype, cpts, opt, oval)
            st.session_state.res = {
                'status':'ok', 'c1':c1, 'c2':c2, 'otype':otype,
                'cons':cons, 'cpts':cpts, 'opt':opt, 'oval':oval, 'img':img}
    except Exception as e:
        st.session_state.res = {'status':'error', 'msg':str(e)}
    st.rerun()

# ── График ────────────────────────────────────────────────────
with RIGHT:
    r = st.session_state.res
    if r and r['status'] == 'ok':
        tab1, tab2 = st.tabs(["📈 График решения", "🎬 Анимация"])
        with tab1:
            st.image(r['img'], use_container_width=True)
        with tab2:
            st.markdown("**Шаг 3 — движение линии уровня к оптимуму**")
            step = st.slider("Шаг", 0, 40, 0)
            aimg = make_fig(r['cons'], r['c1'], r['c2'], r['otype'],
                            r['cpts'], r['opt'], r['oval'],
                            astep=step, tsteps=40)
            st.image(aimg, use_container_width=True)
            if step == 0:
                st.info("Бошланғич ҳолат — линия через внутреннюю точку")
            elif step == 40:
                st.success("✅ Оптимум топилди!")
            else:
                x3 = (r['cpts'][0][0]+r['cpts'][1][0])/2
                y3 = (r['cpts'][0][1]+r['cpts'][1][1])/2
                Ci = r['c1']*x3 + r['c2']*y3
                st.caption(f"Z = {Ci + (r['oval']-Ci)*step/40:.2f}")
    else:
        st.info("👈 Маълумот киритиб **«Решить»** тугмасини босинг")
