# -------------------------------
# Base directory and configuration
# -------------------------------
import os
import numpy as np
import pandas as pd

from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource,
    MultiSelect,
    Dropdown,
    Button,
    Range1d,
    LinearAxis,
    Div,
    Spacer,
    DataTable,
    TableColumn,
    NumberFormatter,
    LinearColorMapper,
    ColorBar,
    NumericInput
)
from bokeh.layouts import column, row

SRC_PATH = os.getcwd()
ROOT_PATH = os.path.abspath(os.path.join(SRC_PATH, "../"))
data_dir = f'{ROOT_PATH}/data/MatAnx_Data_all'
colors = ["blue", "green", "red", "orange", "purple", "brown", "pink", "gray", "cyan", "lime", "magenta"]

# -------------------------------
# Gramian Angular Field (GAF) Function
# -------------------------------
def gramian_angular_field(signal):
    if len(signal) < 2:
        return np.array([]), np.array([])
    min_val, max_val = np.min(signal), np.max(signal)
    if max_val == min_val:
        return np.array([]), np.array([])
    signal_scaled = 2 * (signal - min_val) / (max_val - min_val) - 1
    phi = np.arccos(np.clip(signal_scaled, -1, 1))
    gaf_sum = np.cos(phi[:, None] + phi[None, :])
    gaf_diff = np.sin(phi[:, None] - phi[None, :])
    return gaf_sum, gaf_diff

# -------------------------------
# Main time-series plot
# -------------------------------
p = figure(
    title="Vrednosti signala skozi čas",
    x_axis_label='Čas[s]',
    y_axis_label='Vrednost signala',
    width=1200, height=300,
    tools="pan,wheel_zoom,box_zoom,reset,save"
)
p.extra_y_ranges = {"right": Range1d(start=0, end=1)}
p.add_layout(LinearAxis(y_range_name="right", axis_label="Vrednost anksioznosti"), 'right')

# Dictionaries to hold active signal sources
sources = {}
sources_020 = {}

# -------------------------------
# Widgets
# -------------------------------
available_signals = ['021', '022', '023', '024', '025', '026', '027']
multi_select_signals = MultiSelect(title="", options=available_signals)

available_IDs = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011']
multi_select_IDs = MultiSelect(title="", options=available_IDs)

available_nrs = ['001', '002', '003', '004', '005']
dropdown_nrs = Dropdown(label="Izberi št. poskusa", menu=[(v, v) for v in available_nrs])

available_features = [
    ("časovne značilke", "časovne značilke"),
    ("spektralne značilke", "spektralne značilke"),
    ("gramova kotna polja", "gramova kotna polja")
]
feature_dropdown = Dropdown(label="Tip značilk:", menu=available_features)

signals_title = Div(text="<b>Izbira signala</b>")
ids_title     = Div(text="<b>Izbira uID</b>")
noe_title     = Div(text="<b>Izbira št. poskusa</b>")
feature_title = Div(text="<b>Izbira tipa značilk</b>")

signals_column  = column(signals_title, multi_select_signals)
ids_column      = column(ids_title, multi_select_IDs)
controls_column = column(noe_title, dropdown_nrs, feature_title, feature_dropdown)

plot_button         = Button(label="Izriši", width=120)
global_ok_button    = Button(label="OK", width=100)
global_ok_button.visible = False
global_ok_button.on_click(lambda: update_per_uid_table())

drawn_data = False

# -------------------------------
# “Global” features table (time or spectral)
# -------------------------------
time_columns = [
    TableColumn(field="uID",    title="uID"),
    TableColumn(field="Signal", title="Tip signala"),
    TableColumn(field="Average", title="Povprečje", formatter=NumberFormatter(format="0.0000")),
    TableColumn(field="Variance", title="Varianca", formatter=NumberFormatter(format="0.0000")),
    TableColumn(field="StdDev",  title="Std odklon", formatter=NumberFormatter(format="0.0000"))
]
spectral_columns = [
    TableColumn(field="uID",       title="uID"),
    TableColumn(field="Signal",    title="Tip signala"),
    TableColumn(field="Amplitude", title="Amplituda", formatter=NumberFormatter(format="0.0000")),
    TableColumn(field="Phase",     title="Faza",      formatter=NumberFormatter(format="0.0000"))
]
stats_source = ColumnDataSource(data={})
stats_table  = DataTable(source=stats_source, columns=time_columns, width=400, height=200)
stats_table_title = Div(text="<b>Vrednosti izbranega signala</b>", visible=False)

stats_table.visible = False

# -------------------------------
# “Per-UID” inputs + stats table
# -------------------------------
uid_inputs = {}

per_uid_stats_source = ColumnDataSource(data=dict(
    uID=[], Signal=[], Average=[], Variance=[], StdDev=[]
))
per_uid_time_columns     = time_columns.copy()
per_uid_spectral_columns = spectral_columns.copy()
per_uid_stats_table = DataTable(
    source=per_uid_stats_source,
    columns=per_uid_time_columns,
    width=400, height=200,
    index_position=None
)

per_uid_inputs_column = column()
per_uid_title          = Div(text="<b>Nastavitev časovnih območij</b>", visible=False)
start_end              = row(per_uid_stats_table, per_uid_inputs_column)
per_uid_inputs_column.visible = False
per_uid_stats_table.visible  = False
start_end.visible            = False

# -------------------------------
# Containers for GAF and spectral sub-plots
# -------------------------------
gaf_container      = column()
spectral_container = column()

# -------------------------------
# Track selections
# -------------------------------
selected_signals = []
selected_IDs     = []
selected_nr      = None
selected_feature = None

# -------------------------------
# Callbacks for widget updates
# -------------------------------
def update_signals(attr, old, new):
    global selected_signals
    selected_signals = new

def update_IDs(attr, old, new):
    global selected_IDs, uid_inputs
    selected_IDs = new or []

    new_inputs = {}
    for uid in selected_IDs:
        if uid in uid_inputs:
            start_w, end_w = uid_inputs[uid]
        else:
            start_w = NumericInput(value=0, low=0, high=10000, width=100)
            end_w   = NumericInput(value=10000, low=0, high=10000, width=100)
        new_inputs[uid] = (start_w, end_w)
    uid_inputs = new_inputs

def update_nr(event):
    global selected_nr
    selected_nr = event.item
    dropdown_nrs.label = f"Izbran: {selected_nr}"

def update_feature(event):
    global selected_feature, drawn_data
    selected_feature = event.item
    feature_dropdown.label = f"Tip značilk: {selected_feature}"

    has_data = bool(drawn_data and selected_IDs and selected_nr and selected_signals)

    has_data = bool(drawn_data and selected_IDs and selected_nr and selected_signals)

    if has_data and selected_feature == "časovne značilke":
        stats_table.columns         = time_columns
        per_uid_stats_table.columns = per_uid_time_columns
    elif has_data and selected_feature == "spektralne značilke":
        stats_table.columns         = spectral_columns
        per_uid_stats_table.columns = per_uid_spectral_columns
    else:
        stats_source.data = {}
        has_data = False

    stats_table_title.visible    = has_data
    per_uid_title.visible        = has_data
    start_end.visible            = has_data
    global_ok_button.visible     = has_data
    stats_table.visible          = has_data
    per_uid_stats_table.visible  = has_data

    show = selected_feature in ["časovne značilke", "spektralne značilke"]
    per_uid_inputs_column.visible = show
    per_uid_stats_table.visible   = show

    update_statistics(None, None, None)
    update_per_uid_table()
    
multi_select_signals.on_change('value', update_signals)
multi_select_IDs.on_change('value', update_IDs)
dropdown_nrs.on_click(update_nr)
feature_dropdown.on_click(update_feature)

# -------------------------------
# Plot button: load/normalize signals & update both tables
# -------------------------------
def draw_data():
    global sources, sources_020, drawn_data
    sources     = {}
    sources_020 = {}

    if bool(selected_IDs and selected_nr and selected_signals):
        drawn_data  = True

    if not selected_signals or not selected_IDs or selected_nr is None:
        global_ok_button.visible        = False
        per_uid_title.visible          = False
        stats_table_title.visible      = False
        stats_table.visible            = False
        per_uid_stats_table.visible    = False
        per_uid_inputs_column.visible  = False
        start_end.visible              = False
        p.title.text = "Ena ali več možnosti v spustnem seznamu ni izbranih!"
        stats_source.data              = {}
        per_uid_stats_source.data      = {}
        return

    p.renderers = []
    if p.legend:
        p.legend.items = []

    color_index = 0
    for signal in selected_signals:
        for uid in selected_IDs:
            key = f"{signal}-{uid}-{selected_nr}"
            sources[key] = ColumnDataSource(data={'time': [], 'signal': []})
            sources_020[key] = ColumnDataSource(data={'time': [], 'signal': []})
            src = sources[key]
            file_path = os.path.join(data_dir, f"{signal}-{uid}-{selected_nr}-000.csv")
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path, header=None)
                    min_val = df[3].min()
                    max_val = df[3].max()
                    normalized_signal = (df[3] - min_val) / (max_val - min_val)
                    src.data = {'time': df[2], 'signal': normalized_signal}
                    color = colors[color_index % len(colors)]
                    p.line('time', 'signal',
                           legend_label=f"Signal {signal}-{uid}-{selected_nr}",
                           line_width=2, source=src, color=color)
                    color_index += 1
                except Exception as e:
                    print(f"Error loading data for {key}: {e}")
            else:
                print(f"File not found for {key}!")
            
            file_path_020 = os.path.join(data_dir, f"020-{uid}-{selected_nr}-000.csv")
            if os.path.exists(file_path_020):
                try:
                    df_020 = pd.read_csv(file_path_020, header=None)
                    sources_020[key].data = {'time': df_020[2], 'signal': df_020[4]}
                    p.scatter('time', 'signal',
                              legend_label=f"Anx 020-{uid}-{selected_nr}",
                              size=10, source=sources_020[key],
                              y_range_name="right",
                              color=color,
                              marker='square',
                              fill_alpha=1,
                              line_color='black',
                              line_width=1)
                except Exception as e:
                    print(f"Error loading signal 020 for {uid}: {e}")
            else:
                print(f"File for signal 020 not found for {uid}!")

    p.y_range.start = 0
    p.y_range.end   = 1
    p.extra_y_ranges["right"].start = 0
    p.extra_y_ranges["right"].end   = 1
    p.title.text = "Vrednosti signala skozi čas"

    show_stats = selected_feature in ["časovne značilke", "spektralne značilke"]

    stats_table.visible           = show_stats
    per_uid_title.visible         = show_stats
    stats_table_title.visible = show_stats
    start_end.visible             = show_stats
    per_uid_inputs_column.visible = show_stats
    per_uid_stats_table.visible = show_stats
    global_ok_button.visible      = show_stats and len(selected_IDs) > 1

    if show_stats:
        if selected_feature == "časovne značilke":
            stats_table.columns         = time_columns
            per_uid_stats_table.columns = per_uid_time_columns
            global_ok_button.visible = True
        else:
            stats_table.columns         = spectral_columns
            per_uid_stats_table.columns = per_uid_spectral_columns
            global_ok_button.visible = True

    per_uid_inputs_column.children = []
    header = row(
        Div(text="", width=30),                  
        Div(text="<b>t1</b>", width=100),
        Spacer(width=10),
        Div(text="<b>t2</b>",   width=100),
        Spacer(width=10),
        Div(text="<b>Potrdi vse</b>", width=100)
    )
    per_uid_inputs_column.children.append(header)

    for idx, uid in enumerate(selected_IDs):
        start_w, end_w = uid_inputs[uid]
        cell = global_ok_button if idx == 0 else Div(text="", width=100)
        per_uid_inputs_column.children.append(
            row(
                Div(text=f"<b>{uid}</b>", width=30),
                start_w, Spacer(width=10),
                end_w,   Spacer(width=10),
                cell
            )
        )

    show = len(selected_IDs) >= 1
    per_uid_inputs_column.visible = show
    global_ok_button.visible      = show

    update_statistics(None, None, None)
    update_per_uid_table()

plot_button.on_click(draw_data)

# -------------------------------
# Spectral features plotting
# -------------------------------
def plot_spectral_features():
    plot_size = 250
    columns_list = []

    start, end = p.x_range.start, p.x_range.end

    for key in sources:
        data  = sources[key].data
        times = np.array(data.get('time', []))
        sig   = np.array(data.get('signal', []))
        if times.size == 0:
            continue

        mask = (times >= start) & (times <= end)
        filtered_signal = sig[mask]
        if filtered_signal.size == 0:
            continue

        fft_vals = np.fft.fft(filtered_signal)
        half_N   = len(filtered_signal) // 2
        xvals    = np.arange(half_N)

        amp_spec     = np.abs(fft_vals[:half_N])
        amp_spec     = np.where(amp_spec == 0, 1e-12, amp_spec)
        log_amp_spec = np.log10(amp_spec)
        phase_spec   = np.angle(fft_vals[:half_N])

        fig_amp = figure(width=plot_size, height=plot_size,
                         toolbar_location=None,
                         title=f"Log-Amplitude {key}")
        fig_amp.line(xvals, log_amp_spec, line_width=1)

        fig_phase = figure(width=plot_size, height=plot_size,
                           toolbar_location=None,
                           title=f"Phase {key}")
        fig_phase.line(xvals, phase_spec, line_width=1)

        columns_list.append(column(fig_amp, fig_phase))

    if columns_list:
        return row(*columns_list)
    else:
        return Div(text=" ")

# -------------------------------
# Dynamic GAF update
# -------------------------------
def update_gaf_plots(attr, old, new):
    start, end = p.x_range.start, p.x_range.end
    gaf_plots = []
    for key, source in sources.items():
        data  = source.data
        times = np.array(data.get('time', []))
        sig   = np.array(data.get('signal', []))
        if times.size == 0:
            continue
        mask = (times >= start) & (times <= end)
        filtered_signal = sig[mask]
        if filtered_signal.size < 2:
            continue
        max_length = 2500
        if filtered_signal.size > max_length:
            factor = int(filtered_signal.size / max_length)
            filtered_signal = filtered_signal[::factor]
        gaf_sum, _ = gramian_angular_field(filtered_signal)
        if gaf_sum.size == 0:
            continue

        fig = figure(width=400, height=300, tools="pan,wheel_zoom,reset",
                     title=f"GAF {key}")
        mapper    = LinearColorMapper(palette="Viridis256", low=np.min(gaf_sum), high=np.max(gaf_sum))
        fig.image(image=[gaf_sum], x=0, y=0, dw=gaf_sum.shape[1], dh=gaf_sum.shape[0], color_mapper=mapper)
        color_bar = ColorBar(color_mapper=mapper, label_standoff=12, location=(0, 0))
        fig.add_layout(color_bar, 'right')
        gaf_plots.append(fig)

    if gaf_plots:
        gaf_container.children = [row(*gaf_plots)]
    else:
        gaf_container.children = []

# -------------------------------
# Update “global” features table
# -------------------------------
def update_statistics(attr, old, new):
    start, end = p.x_range.start, p.x_range.end
    if start is None or end is None:
        stats_source.data = {}
        return
    
    stats_rows = []
    for key, source in sources.items():
        data  = source.data
        times = np.array(data.get('time', []))
        sig   = np.array(data.get('signal', []))
        if times.size == 0:
            continue
        mask = (times >= start) & (times <= end)
        filtered_signal = sig[mask]
        if filtered_signal.size == 0:
            continue

        parts      = key.split('-')
        uID        = parts[1] if len(parts) >= 2 else ""
        signal_code= parts[0] if len(parts) >= 1 else ""

        if selected_feature == "časovne značilke":
            avg_val = np.mean(filtered_signal)
            var_val = np.var(filtered_signal)
            std_val = np.std(filtered_signal)
            stats_rows.append({
                'uID': uID,
                'Signal': signal_code,
                'Average': avg_val,
                'Variance': var_val,
                'StdDev': std_val
            })
        elif selected_feature == "spektralne značilke":
            fft_vals   = np.fft.fft(filtered_signal)
            half_N     = len(filtered_signal) // 2
            amp_spec   = np.abs(fft_vals[:half_N])
            phase_spec = np.angle(fft_vals[:half_N])
            idx = np.argmax(amp_spec[1:]) + 1 if half_N > 1 else 0
            stats_rows.append({
                'uID': uID,
                'Signal': signal_code,
                'Amplitude': amp_spec[idx],
                'Phase': phase_spec[idx]
            })

    if stats_rows:
        if selected_feature == "časovne značilke":
            stats_source.data = {
                'uID':       [r['uID']       for r in stats_rows],
                'Signal':    [r['Signal']    for r in stats_rows],
                'Average':   [r['Average']   for r in stats_rows],
                'Variance':  [r['Variance']  for r in stats_rows],
                'StdDev':    [r['StdDev']    for r in stats_rows]
            }
        else:
            stats_source.data = {
                'uID':      [r['uID']      for r in stats_rows],
                'Signal':   [r['Signal']   for r in stats_rows],
                'Amplitude':[r['Amplitude']for r in stats_rows],
                'Phase':    [r['Phase']    for r in stats_rows]
            }
    else:
        stats_source.data = {}

    if selected_feature == "spektralne značilke":
        spectral_container.children = [plot_spectral_features()]
        gaf_container.children      = []
    elif selected_feature == "gramova kotna polja":
        update_gaf_plots(None, None, None)
        spectral_container.children = []
    else:
        spectral_container.children = []
        gaf_container.children      = []

p.x_range.on_change('start', update_statistics)
p.x_range.on_change('end',   update_statistics)

# -------------------------------
# Update “per-UID” stats table
# -------------------------------
def update_per_uid_table():
    if selected_feature == "časovne značilke":
        stats = {'uID':[], 'Signal':[], 'Average':[], 'Variance':[], 'StdDev':[]}
    else:
        stats = {'uID':[], 'Signal':[], 'Amplitude':[], 'Phase':[]}

    for signal in selected_signals:
        for uid in selected_IDs:
            key = f"{signal}-{uid}-{selected_nr}"
            if key not in sources:
                continue
            start_w, end_w = uid_inputs.get(uid, (None, None))
            if start_w is None:
                continue

            t = np.array(sources[key].data['time'])
            s = np.array(sources[key].data['signal'])
            seg = s[(t >= start_w.value) & (t <= end_w.value)]
            if seg.size == 0:
                continue

            if selected_feature == "časovne značilke":
                stats['uID'].append(uid)
                stats['Signal'].append(signal)
                stats['Average'].append(np.mean(seg))
                stats['Variance'].append(np.var(seg))
                stats['StdDev'].append(np.std(seg))
            else:
                fftv  = np.fft.fft(seg)
                half  = len(seg) // 2
                amp   = np.abs(fftv[:half])
                phase = np.angle(fftv[:half])
                idx = np.argmax(amp[1:]) + 1 if half > 1 else 0
                stats['uID'].append(uid)
                stats['Signal'].append(signal)
                stats['Amplitude'].append(amp[idx])
                stats['Phase'].append(phase[idx])

    per_uid_stats_source.data = stats

# -------------------------------
# Layout: assemble everything
# -------------------------------
left_col = column(
    row(
        signals_column,
        ids_column,
        column(noe_title, dropdown_nrs, feature_title, feature_dropdown, plot_button)
    ),
    stats_table_title,
    stats_table,
    Spacer(height=10),
    per_uid_title,
    start_end,
    Spacer(height=10),
)

spacer1 = Spacer(width=50, height=10)
spacer2 = Spacer(width=50, height=10)
top_row = row(left_col, spacer1, gaf_container, spacer2, spectral_container)
layout  = column(
    top_row,
    p,
    sizing_mode="stretch_width"
)

curdoc().clear()
curdoc().add_root(layout)
curdoc().title = "Zaslon 2"
