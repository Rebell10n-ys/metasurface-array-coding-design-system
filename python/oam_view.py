"""Pillow/Tk views for the sampled OAM field; no plotting dependency needed."""
import colorsys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size, bold=False):
    candidates = ([r'C:\Windows\Fonts\segoeuib.ttf'] if bold else [r'C:\Windows\Fonts\segoeui.ttf'])
    candidates += ['DejaVuSans.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _heatmap(field, valid, phase=False):
    if phase:
        values = (np.angle(field) + np.pi) / (2*np.pi)
        hsv = np.zeros(field.shape + (3,), dtype=np.uint8)
        hsv[..., 0] = (np.nan_to_num(values) * 255).astype(np.uint8)
        hsv[..., 1:] = 230
        pixels = np.asarray(Image.fromarray(hsv, mode='HSV').convert('RGB')).copy()
        finite_power = np.nan_to_num(np.abs(field) ** 2)
        valid = valid & (finite_power > np.max(finite_power)*1e-3)
    else:
        values = np.nan_to_num(np.abs(field)**2)
        values = np.clip(values / max(np.max(values), 1e-30), 0, 1)
        anchors = np.array([[15, 23, 42], [70, 45, 119], [163, 55, 114],
                            [232, 108, 65], [253, 194, 89], [255, 248, 183]])
        pixels = np.stack([np.interp(values, np.linspace(0, 1, len(anchors)), anchors[:, c])
                           for c in range(3)], axis=-1).astype(np.uint8)
    pixels[~valid] = [232, 237, 244]
    # Field arrays store increasing v down the rows; show +v upward.
    return Image.fromarray(np.flipud(pixels))


def render_oam_panel(B, analysis):
    image = Image.new('RGB', (1240, 850), '#f3f6fb')
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1240, 88), fill='#10243b')
    metrics = analysis['metrics']
    draw.text((28, 13), 'OAM VORTEX / SAMPLED ARRAY-FACTOR MODEL', font=_font(25, True), fill='white')
    winding = metrics['ring_winding']
    winding_text = 'undefined' if winding is None else f'{winding:+.2f}'
    draw.text((29, 52),
              f"Target l={metrics['target_charge']:+d}    Disk mode fraction={metrics['target_mode_fraction']:.1%}"
              f"    Ring winding={winding_text}    Axis/peak={metrics['axis_to_peak_db']:.1f} dB",
              font=_font(17), fill='#c4e3f0')
    panel_x = [28, 432, 836]
    titles = ['Coding states (2-bit)', 'Far-field intensity / peak', 'Far-field phase (masked nulls)']
    palette = np.array([[27,79,114],[46,134,193],[243,156,18],[192,57,43]], dtype=np.uint8)
    pictures = [Image.fromarray(palette[np.asarray(B, dtype=int)]),
                _heatmap(analysis['complex_map'], analysis['valid_map']),
                _heatmap(analysis['complex_map'], analysis['valid_map'], phase=True)]
    for index, (left, title, picture) in enumerate(zip(panel_x, titles, pictures)):
        draw.rounded_rectangle((left-8, 101, left+376, 508), radius=12, fill='white')
        draw.text((left+7, 113), title, font=_font(18, True), fill='#172c45')
        image.paste(picture.resize((336,336), Image.Resampling.NEAREST), (left+15,149))
        if index:
            draw.text((left+16, 488), f"u,v in [-{metrics['uv_radius']:.3f}, +{metrics['uv_radius']:.3f}]",
                      font=_font(13), fill='#52677c')
        else:
            draw.text((left+16, 488), 'States: 0 / 1 / 2 / 3; rows=x, columns=y', font=_font(13), fill='#52677c')
    # Color scales are explicit; phase is undefined at very weak amplitude.
    for i in range(4):
        draw.rectangle((panel_x[0]+18+82*i, 516, panel_x[0]+30+82*i, 528), fill=tuple(palette[i]))
        draw.text((panel_x[0]+35+82*i, 513), str(i), font=_font(14), fill='#334155')
    for i in range(336):
        value = i/335
        intensity = _heatmap(np.array([[np.sqrt(value), 1.0]], dtype=complex), np.ones((1,2), bool))
        draw.line((panel_x[1]+15+i,518,panel_x[1]+15+i,526),fill=intensity.getpixel((0,0)))
        rgb=tuple(round(255*v) for v in colorsys.hsv_to_rgb(value,230/255,230/255))
        draw.line((panel_x[2]+15+i,518,panel_x[2]+15+i,526),fill=rgb)
    draw.text((panel_x[1]+15,530),'0',font=_font(12),fill='#52677c')
    draw.text((panel_x[1]+338,530),'1',font=_font(12),fill='#52677c')
    draw.text((panel_x[2]+15,530),'-pi',font=_font(12),fill='#52677c')
    draw.text((panel_x[2]+330,530),'+pi',font=_font(12),fill='#52677c')
    # Ring phase, unwrapped, plotted relative to its first sample.
    draw.rounded_rectangle((20,560,600,790),radius=12,fill='white')
    draw.text((40,572),'Phase around the brightest sampled ring',font=_font(18,True),fill='#172c45')
    if metrics['ring_phase_valid']:
        # Close the curve at 2*pi; the original angular samples exclude that endpoint.
        ring=np.asarray(analysis['ring_field'])
        phase=np.unwrap(np.angle(np.r_[ring, ring[:1]]))
        phase=(phase-phase[0])/(2*np.pi)
        low=min(-0.1,float(phase.min())-0.1)
        high=max(0.1,float(phase.max())+0.1)
        draw.line((68,613,68,749,568,749),fill='#aab8c6',width=1)
        points=[(68+500*i/(len(phase)-1),749-136*(v-low)/(high-low)) for i,v in enumerate(phase)]
        draw.line(points,fill='#0e9aa7',width=3)
        draw.text((40,613),f'{high:.1f}',font=_font(12),fill='#52677c')
        draw.text((40,735),f'{low:.1f}',font=_font(12),fill='#52677c')
        draw.text((68,760),'0',font=_font(13),fill='#52677c')
        draw.text((260,760),'azimuth (rad)',font=_font(13),fill='#52677c')
        draw.text((535,760),'2*pi',font=_font(13),fill='#52677c')
        draw.text((70,596),'phase / (2*pi)',font=_font(11),fill='#52677c')
    else:
        draw.text((50,650),'Phase winding is undefined on this ring.',font=_font(17),fill='#a64b32')
    draw.rounded_rectangle((620,560,1220,790),radius=12,fill='white')
    draw.text((640,572),'OAM spectrum within the declared uv disk',font=_font(18,True),fill='#172c45')
    orders=np.asarray(analysis['mode_orders'])
    fractions=np.asarray(analysis['mode_fractions'])
    selected=list(range(-6,7))
    values=[float(fractions[orders==m].sum()) for m in selected]
    values.append(max(0.0,1-sum(values)))
    labels=[str(m) for m in selected]+['rest']
    draw.line((650,610,650,749,1200,749),fill='#aab8c6',width=1)
    for i,(label,value) in enumerate(zip(labels,values)):
        left=658+i*38
        fill='#0e9aa7' if label==str(metrics['target_charge']) else '#9cb7ce'
        draw.rectangle((left,749-130*value,left+22,749),fill=fill)
        draw.text((left-1,757),label,font=_font(12),fill='#52677c')
    draw.text((628,608),'1.0',font=_font(11),fill='#52677c')
    wide=metrics.get('wide_angle')
    if wide:
        draw.text((27,799),f"Wide disk R={wide['uv_radius']:.2f}: target fraction {wide['target_mode_fraction']:.1%}. "
                  'Ring winding is enclosed charge; a centered unsplit core is not verified.',
                  font=_font(13),fill='#a64b32')
    draw.text((27,825),'All FFT bins, r*dr measure in declared uv disks. Ideal scalar model; not radiation efficiency or full-wave validation.',
              font=_font(13),fill='#52677c')
    return image


def show_oam_window(parent, B, analysis):
    import tkinter as tk
    from PIL import ImageTk
    window=tk.Toplevel(parent)
    window.title('涡旋波强度、相位与OAM模式谱')
    panel=render_oam_panel(B,analysis)
    scale=min(1.0,(window.winfo_screenwidth()-100)/panel.width,(window.winfo_screenheight()-140)/panel.height)
    if scale<1:
        panel=panel.resize((int(panel.width*scale),int(panel.height*scale)),Image.Resampling.LANCZOS)
    photo=ImageTk.PhotoImage(panel,master=window)
    label=tk.Label(window,image=photo)
    label.image=photo
    label.pack()
    return window
