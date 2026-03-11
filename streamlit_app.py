import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
from itertools import combinations
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
import base64

# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Линейное программирование — Решатель",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 2rem;
    font-weight: 800;
    color: #1a3a5c;
    margin-bottom: 0.5rem;
}
.step-box {
    background: #f0f7ff;
    border-left: 4px solid #2980b9;
    padding: 10px 14px;
    border-radius: 6px;
    margin-bottom: 8px;
    font-size: 0.92rem;
}
.result-box {
    background: #eafaf1;
    border: 2px solid #27ae60;
    border-radius: 10px;
    padding: 14px;
    margin-top: 10px;
}
.error-box {
    background: #fef9f9;
    border: 2px solid #e74c3c;
    border-radius: 10px;
    padding: 14px;
    margin-top: 10px;
    color: #c0392b;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📊 Линейное программирование — Решатель</div>',
            unsafe_allow_html=True)

# ─────────────────────────────────────────────
# АЛГОРИТМ
# ─────────────────────────────────────────────

def parse_float(s):
    return float(str(s).replace(',', '.').strip())

def line_intersection(a1, b1, c1, a2, b2, c2):
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-10:
        return None
    x = (c1 * b2 - c2 * b1) / det
    y = (a1 * c2 - a2 * c1) / det
    return (x, y)

def satisfies(x, y, constraints):
    for a, b, sign, c in constraints:
        val = a * x + b * y
        if sign == '<=' and val > c + 1e-8:
            return False
        if sign == '>=' and val < c - 1e-8:
            return False
        if sign == '='  and abs(val - c) > 1e-8:
            return False
    return True

def find_corner_points(constraints):
    """Шаг 0: перебор всех пар прямых → угловые точки ОДР."""
    corners = []
    for i, j in combinations(range(len(constraints)), 2):
        a1, b1, _, c1 = constraints[i]
        a2, b2, _, c2 = constraints[j]
        pt = line_intersection(a1, b1, c1, a2, b2, c2)
        if pt is None:
            continue
        x, y = pt
        if satisfies(x, y, constraints):
            dup = any(abs(cx - x) < 1e-6 and abs(cy - y) < 1e-6
                      for cx, cy in corners)
            if not dup:
                corners.append((round(x, 6), round(y, 6)))
    return corners

def find_optimum(corners, c1, c2, opt_type):
    if not corners:
        return None, None
    vals = [c1 * x + c2 * y for x, y in corners]
    idx = int(np.argmax(vals)) if opt_type == 'max' else int(np.argmin(vals))
    return corners[idx], vals[idx]

def inner_point(corners):
    """Шаг 1: внутренняя точка — среднее двух угловых точек."""
    x3 = (corners[0][0] + corners[1][0]) / 2
    y3 = (corners[0][1] + corners[1][1]) / 2
    return x3, y3

def level_line_axis_points(c1, c2, x3, y3):
    """Шаг 2: пересечения линии уровня с осями."""
    C = c1 * x3 + c2 * y3
    pts = []
    if abs(c2) > 1e-10:
        pts.append((0.0, round(C / c2, 4)))
    if abs(c1) > 1e-10:
        pts.append((round(C / c1, 4), 0.0))
    return C, pts

def sort_polygon(points):
    if len(points) < 3:
        return points
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return sorted(points, key=lambda p: np.arctan2(p[1]-cy, p[0]-cx))

# ─────────────────────────────────────────────
# ГРАФИК
# ─────────────────────────────────────────────

COLORS = ['#1f77b4','#d62728','#2ca02c','#9467bd',
          '#8c564b','#e377c2','#17becf','#bcbd22']

def build_figure(constraints, c1, c2, opt_type,
                 corners, optimum, opt_val,
                 animate_step=None, total_steps=40):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.set_facecolor('#FAFAFA')

    # Диапазон осей
    if corners:
        all_x = [p[0] for p in corners]
        all_y = [p[1] for p in corners]
        margin = max(4, (max(all_x)-min(all_x))*0.55+3,
                        (max(all_y)-min(all_y))*0.55+3)
        cx_mid = (max(all_x)+min(all_x))/2
        cy_mid = (max(all_y)+min(all_y))/2
    else:
        margin, cx_mid, cy_mid = 8, 0, 0

    x_min, x_max = cx_mid - margin, cx_mid + margin
    y_min, y_max = cy_mid - margin, cy_mid + margin
    xs = np.linspace(x_min, x_max, 700)

    # ── Ограничения ──
    for idx, (a, b, sign, c) in enumerate(constraints):
        col = COLORS[idx % len(COLORS)]
        s = {'<=':'≤','>=':'≥','=':'='}.get(sign, sign)
        lbl = f'{a:.2f}·x + {b:.2f}·y {s} {c:.2f}'
        if abs(b) > 1e-10:
            ax.plot(xs, (c - a*xs)/b, color=col, lw=1.8, label=lbl)
        elif abs(a) > 1e-10:
            ax.axvline(c/a, color=col, lw=1.8, label=lbl)

    # ── ОДР (закрашенный многоугольник) ──
    if len(corners) >= 3:
        poly_pts = sort_polygon(corners)
        patch = MplPolygon(poly_pts, closed=True,
                           facecolor='#AED6F1', edgecolor='none',
                           alpha=0.35, zorder=2, label='ОДР')
        ax.add_patch(patch)

    # ── Угловые точки ──
    if corners:
        ax.scatter([p[0] for p in corners],
                   [p[1] for p in corners],
                   color='red', s=65, zorder=6,
                   edgecolors='darkred', lw=1, label='Угловые точки')

    # ── Линия уровня ──
    if optimum:
        opt_x, opt_y = optimum
        C_final = c1*opt_x + c2*opt_y

        if animate_step is not None and len(corners) >= 2:
            # Анимация: линия движется от C_inner к C_final
            x3, y3 = inner_point(corners)
            C_inner = c1*x3 + c2*y3
            t = min(animate_step, total_steps) / total_steps
            C_cur = C_inner + (C_final - C_inner) * t
        else:
            C_cur = C_final

        if abs(c2) > 1e-10:
            y_lv = (C_cur - c1*xs) / c2
            lbl_lv = (f'Линия уровня Z={C_cur:.2f}'
                      if animate_step is not None
                      else f'Целевая прямая: {c1:.2f}·x+{c2:.2f}·y={C_cur:.2f}')
            ax.plot(xs, y_lv, 'k--', lw=1.8, alpha=0.8,
                    label=lbl_lv, zorder=5)

        # ── Вектор ∇Z ──
        if animate_step is None:
            norm = np.sqrt(c1**2 + c2**2)
            if norm > 1e-10:
                sc = margin * 0.17
                dx, dy = c1/norm*sc, c2/norm*sc
                ax.annotate('', xy=(opt_x+dx, opt_y+dy),
                            xytext=(opt_x, opt_y),
                            arrowprops=dict(arrowstyle='->',
                                            color='darkred', lw=2.2),
                            zorder=9)
                ax.text(opt_x+dx*1.2, opt_y+dy*1.2, '∇Z',
                        fontsize=11, color='darkred',
                        fontweight='bold', zorder=9)

        # ── Оптимум ──
        ax.scatter([opt_x], [opt_y], color='gold', s=240,
                   zorder=10, marker='*',
                   edgecolors='#8B6914', lw=1.2, label='Оптимум')
        ax.annotate(f'Оптимум ({opt_x:.2f}; {opt_y:.2f})',
                    (opt_x, opt_y),
                    xytext=(opt_x + margin*0.18, opt_y + margin*0.18),
                    fontsize=9, color='#1A5276', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.25',
                              facecolor='#EBF5FB', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='#1A5276', lw=1.2),
                    zorder=11)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axhline(0, color='black', lw=0.8)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('График решения', fontsize=13, pad=10)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7.5, loc='upper right', framealpha=0.93)

    fig.tight_layout()
    return fig

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ─────────────────────────────────────────────
# PDF
# ─────────────────────────────────────────────

def generate_pdf(constraints, c1, c2, opt_type,
                 corners, optimum, opt_val, img_bytes):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    title_st = ParagraphStyle('T', parent=styles['Title'],
                               fontSize=15, alignment=TA_CENTER,
                               spaceAfter=10, fontName='Helvetica-Bold')
    h2_st = ParagraphStyle('H2', parent=styles['Heading2'],
                            fontSize=11, spaceBefore=8, spaceAfter=5,
                            fontName='Helvetica-Bold')
    body_st = ParagraphStyle('B', parent=styles['Normal'],
                              fontSize=10, spaceAfter=3)

    story = []
    story.append(Paragraph('Линейное программирование — Решатель', title_st))
    story.append(Paragraph('Отчёт о решении задачи', h2_st))
    story.append(Spacer(1, 0.2*cm))

    arrow = '→ max' if opt_type == 'max' else '→ min'
    story.append(Paragraph(
        f'<b>Целевая функция:</b> Z = {c1}·x + ({c2})·y {arrow}', body_st))
    story.append(Spacer(1, 0.15*cm))

    story.append(Paragraph('<b>Ограничения:</b>', body_st))
    for a, b, sign, c in constraints:
        s = {'<=':'≤','>=':'≥','=':'='}.get(sign, sign)
        story.append(Paragraph(f'&nbsp;&nbsp;{a}·x + ({b})·y {s} {c}', body_st))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph('<b>Результат:</b>', h2_st))
    opt_x, opt_y = optimum
    story.append(Paragraph(
        f'Оптимальная точка: x* = {opt_x:.4f}, y* = {opt_y:.4f}', body_st))
    story.append(Paragraph(
        f'Оптимальное значение: Z* = {opt_val:.4f}', body_st))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph('<b>Угловые точки ОДР:</b>', h2_st))
    tbl_data = [['№', 'x', 'y', 'Z']]
    for i, (cx, cy) in enumerate(corners):
        zv = c1*cx + c2*cy
        mark = ' ← оптимум' if (abs(cx-opt_x)<1e-4 and abs(cy-opt_y)<1e-4) else ''
        tbl_data.append([str(i+1), f'{cx:.4f}', f'{cy:.4f}', f'{zv:.4f}{mark}'])
    tbl = Table(tbl_data, colWidths=[1.2*cm, 4*cm, 4*cm, 6.5*cm])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2980B9')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.white, colors.HexColor('#EBF5FB')]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTSIZE', (0,0), (-1,-1), 9),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph('<b>График решения:</b>', h2_st))
    img_buf = io.BytesIO(img_bytes)
    story.append(Image(img_buf, width=14*cm, height=11*cm))

    doc.build(story)
    buf.seek(0)
    return buf.read()

# ─────────────────────────────────────────────
# ИНТЕРФЕЙС
# ─────────────────────────────────────────────

col_left, col_right = st.columns([1, 1.6], gap="large")

with col_left:
    st.subheader("Целевая функция")
    c1_col, c2_col, opt_col = st.columns([2, 2, 1.5])
    with c1_col:
        c1_str = st.text_input("Коэф. x", value="5,3", label_visibility="collapsed",
                                placeholder="c₁")
        st.caption("* x  +")
    with c2_col:
        c2_str = st.text_input("Коэф. y", value="-7,1", label_visibility="collapsed",
                                placeholder="c₂")
        st.caption("* y  →")
    with opt_col:
        opt_type = st.selectbox("Тип", ["max", "min"],
                                 label_visibility="collapsed")

    st.subheader("Ограничения")

    # Число ограничений
    if 'n_constraints' not in st.session_state:
        st.session_state.n_constraints = 5

    b1, b2 = st.columns(2)
    with b1:
        if st.button("+ Добавить"):
            st.session_state.n_constraints += 1
    with b2:
        if st.button("− Удалить", disabled=st.session_state.n_constraints <= 1):
            st.session_state.n_constraints -= 1

    # Дефолтные значения из скриншота
    defaults = [
        (3.2, -2.0, '=',  3.0),
        (1.6,  2.3, '<=', -5.0),
        (3.2, -6.0, '>=',  7.0),
        (7.0, -2.0, '<=', 10.0),
        (-6.5, 3.0, '<=',  9.0),
    ]

    constraint_inputs = []
    for i in range(st.session_state.n_constraints):
        da, db, ds, dc = defaults[i] if i < len(defaults) else (1.0, 1.0, '<=', 0.0)
        # Строка ввода ограничения: a *x  +  b *y  [≤/≥/=]  c
        row_html_open = "<div style='display:flex;align-items:center;gap:4px;margin-bottom:8px;background:#f7fafd;border:1.5px solid #dce6f0;border-radius:9px;padding:8px 10px;'>"
        st.markdown(row_html_open, unsafe_allow_html=True)

        ca, lbl1, cb, lbl2, cs, lbl3, cc, lbl4 = st.columns([2.2, 0.6, 2.2, 0.5, 1.2, 0.3, 2.2, 0.1])
        with ca:
            a = st.text_input(f"a{i}", value=str(da),
                               label_visibility="collapsed", key=f"a_{i}")
        with lbl1:
            st.markdown("<p style='margin:0;padding-top:6px;font-size:1rem;font-weight:600;color:#2c3e50;text-align:center'>·x</p>", unsafe_allow_html=True)
        with cb:
            b = st.text_input(f"b{i}", value=str(db),
                               label_visibility="collapsed", key=f"b_{i}")
        with lbl2:
            st.markdown("<p style='margin:0;padding-top:6px;font-size:1rem;font-weight:600;color:#2c3e50;text-align:center'>·y</p>", unsafe_allow_html=True)
        with cs:
            sign_map = {
                'kichik (<=)': '<=',
                'katta  (>=)': '>=',
                'teng    (=)': '='
            }
            sign_rev = {v: k for k, v in sign_map.items()}
            sign_options = list(sign_map.keys())
            def_idx = sign_options.index(sign_rev[ds])
            chosen = st.selectbox(f"sign_{i}", sign_options,
                                  index=def_idx,
                                  label_visibility="collapsed",
                                  key=f"s_{i}")
            sign = sign_map[chosen]
        with lbl3:
            st.markdown("<p style='margin:0'></p>", unsafe_allow_html=True)
        with cc:
            c = st.text_input(f"c{i}", value=str(dc),
                               label_visibility="collapsed", key=f"c_{i}")
        with lbl4:
            st.markdown("<p style='margin:0'></p>", unsafe_allow_html=True)
        constraint_inputs.append((a, b, sign, c))

    st.caption("Коэффициенты вводите целыми или дробными (запятая/точка).")

    # Кнопки
    solve_btn = st.button("✅ Решить", type="primary", use_container_width=True)
    clear_btn = st.button("🗑 Очистить", use_container_width=True)

    if clear_btn:
        st.session_state.n_constraints = 5
        if 'result' in st.session_state:
            del st.session_state.result
        st.rerun()

# ─────────────────────────────────────────────
# РЕШЕНИЕ
# ─────────────────────────────────────────────

if solve_btn:
    try:
        c1 = parse_float(c1_str)
        c2 = parse_float(c2_str)

        constraints = []
        for a_s, b_s, sign, c_s in constraint_inputs:
            constraints.append((parse_float(a_s), parse_float(b_s),
                                 sign, parse_float(c_s)))

        # Шаг 0
        corners = find_corner_points(constraints)

        if not corners:
            st.session_state.result = {
                'status': 'error',
                'msg': 'ОДР — пустое множество. Система ограничений несовместна.'
            }
        else:
            optimum, opt_val = find_optimum(corners, c1, c2, opt_type)

            # Шаги 1–2
            x3, y3 = inner_point(corners)
            C_inner, level_pts = level_line_axis_points(c1, c2, x3, y3)

            # Строим финальный график
            fig = build_figure(constraints, c1, c2, opt_type,
                               corners, optimum, opt_val)
            img_bytes = fig_to_bytes(fig)

            st.session_state.result = {
                'status': 'ok',
                'c1': c1, 'c2': c2, 'opt_type': opt_type,
                'constraints': constraints,
                'corners': corners,
                'optimum': optimum,
                'opt_val': opt_val,
                'inner': (x3, y3),
                'C_inner': C_inner,
                'level_pts': level_pts,
                'img_bytes': img_bytes,
            }
    except Exception as e:
        st.session_state.result = {'status': 'error', 'msg': str(e)}

# ─────────────────────────────────────────────
# ВЫВОД РЕЗУЛЬТАТА
# ─────────────────────────────────────────────

with col_left:
    if 'result' in st.session_state:
        r = st.session_state.result
        if r['status'] == 'error':
            st.error(r['msg'])
        else:
            opt_x, opt_y = r['optimum']
            st.success(f"""
**Оптимальная точка:** x\\* = {opt_x:.4f}, y\\* = {opt_y:.4f}

**Значение Z\\*:** {r['opt_val']:.4f} ({r['opt_type']})

**Угловых точек ОДР:** {len(r['corners'])}
""")

            # Алгоритм — шаги
            with st.expander("📋 Алгоритм решения (шаги)", expanded=True):
                corners = r['corners']
                st.markdown(f"""
<div class="step-box">
<b>Шаг 0 — ОДР:</b> Найдено <b>{len(corners)}</b> угловых точек:<br>
{', '.join(f'({x:.3f}; {y:.3f})' for x,y in corners)}
</div>
<div class="step-box">
<b>Шаг 1 — Внутренняя точка:</b><br>
x₃ = ({corners[0][0]:.3f} + {corners[1][0]:.3f}) / 2 = <b>{r['inner'][0]:.4f}</b><br>
y₃ = ({corners[0][1]:.3f} + {corners[1][1]:.3f}) / 2 = <b>{r['inner'][1]:.4f}</b>
</div>
<div class="step-box">
<b>Шаг 2 — Линия уровня:</b><br>
C = {r['c1']}·{r['inner'][0]:.4f} + ({r['c2']})·{r['inner'][1]:.4f} = <b>{r['C_inner']:.4f}</b><br>
Пересечения с осями: {', '.join(f'({p[0]}; {p[1]})' for p in r['level_pts'])}
</div>
<div class="step-box">
<b>Шаг 3 — Оптимум:</b><br>
Линия уровня достигает точки <b>({opt_x:.4f}; {opt_y:.4f})</b>, Z* = <b>{r['opt_val']:.4f}</b>
</div>
""", unsafe_allow_html=True)

            # PDF
            pdf_bytes = generate_pdf(
                r['constraints'], r['c1'], r['c2'], r['opt_type'],
                r['corners'], r['optimum'], r['opt_val'], r['img_bytes']
            )
            st.download_button(
                label="📄 Скачать отчёт (PDF)",
                data=pdf_bytes,
                file_name="lp_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

# ─────────────────────────────────────────────
# ПРАВАЯ КОЛОНКА — ГРАФИК + АНИМАЦИЯ
# ─────────────────────────────────────────────

with col_right:
    if 'result' in st.session_state and st.session_state.result['status'] == 'ok':
        r = st.session_state.result

        tabs = st.tabs(["📈 График решения", "🎬 Анимация (Шаг 3)"])

        # ── Вкладка 1: Финальный график ──
        with tabs[0]:
            st.image(r['img_bytes'], use_container_width=True)

        # ── Вкладка 2: Анимация ──
        with tabs[1]:
            st.markdown("**Шаг 3 — Смещение линии уровня к оптимуму**")
            total_steps = 40
            anim_step = st.slider(
                "Шаг анимации", 0, total_steps, 0,
                help="Двигайте ползунок чтобы увидеть движение линии уровня"
            )
            fig_anim = build_figure(
                r['constraints'], r['c1'], r['c2'], r['opt_type'],
                r['corners'], r['optimum'], r['opt_val'],
                animate_step=anim_step, total_steps=total_steps
            )
            img_anim = fig_to_bytes(fig_anim)
            st.image(img_anim, use_container_width=True)

            if anim_step == 0:
                st.info("⬆️ Ползунок 0 — линия уровня через внутреннюю точку")
            elif anim_step == total_steps:
                st.success("✅ Ползунок максимум — линия достигла оптимума!")
            else:
                t = anim_step / total_steps
                C_cur = r['C_inner'] + (r['opt_val'] - r['C_inner']) * t
                st.caption(f"Текущее значение Z = {C_cur:.2f}")
    else:
        st.info("👈 Введите данные и нажмите **«Решить»**")
