import os
import re
import base64
import io
import pandas as pd

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource, FileInput, MultiSelect,
    Dropdown, Button, Range1d, LinearAxis, Div
)

# -------------------------------
# Configuration
# -------------------------------
SRC_PATH = os.getcwd()
ROOT_PATH = os.path.abspath(os.path.join(SRC_PATH, "../"))
#DATA_FOLDER = r"C:\Users\gaspa\OneDrive\Desktop\masters\data\MatAnx_Data_all"
DATA_FOLDER = os.path.join(os.getcwd(), "..", "data", "MatAnx_Data_all")
#DATA_FOLDER = os.path.join(os.getcwd(), "data", "MatAnx_Data_all")
colors = ["blue", "green", "red", "orange", "purple", "brown", "pink", "gray", "cyan", "lime", "magenta"]
files_uploaded = False

# -------------------------------
# Data store for uploaded CSVs
# -------------------------------
uploaded_dfs = {}

# -------------------------------
# Create the Bokeh plot
# -------------------------------
p = figure(
    title="Vrednosti signala skozi čas",
    x_axis_label='Čas[s]', y_axis_label='Vrednost signala',
    width=1200
)

p.extra_y_ranges = {"right": Range1d(start=0, end=1)}
p.add_layout(LinearAxis(y_range_name="right", axis_label="Vrednost anksioznosti"), 'right')

# -------------------------------
# FileInput widget for CSV upload
# -------------------------------
file_input = FileInput(accept=".csv", multiple=True, width=400)
file_list_div = Div(text="Ni naloženih datotek", width=400)
# file_list_div.visible = False
auto_upload_button = Button(label="Naloži datoteke", width=100)

# -------------------------------
# Selection widgets for plotting
# -------------------------------
available_signals = ['021', '022', '023', '024', '025', '026', '027']
multi_select_signals = MultiSelect(options=available_signals, size=7)

available_IDs = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011']
multi_select_IDs = MultiSelect(options=available_IDs, size=7)

available_nrs = ['001', '002', '003', '004', '005']
menu_nrs = [(noe, noe) for noe in available_nrs]
dropdown_nrs = Dropdown(label="Izberi št. poskusa", menu=menu_nrs)

button = Button(label="Izriši")

# -------------------------------
# State variables for selections
# -------------------------------
selected_signals = []
selected_IDs = []
selected_nr = None

# -------------------------------
# Callbacks for selection widgets
# -------------------------------
def update_signals(attr, old, new):
    global selected_signals
    selected_signals = new

multi_select_signals.on_change('value', update_signals)

def update_IDs(attr, old, new):
    global selected_IDs
    selected_IDs = new

multi_select_IDs.on_change('value', update_IDs)

def update_nr(event):
    global selected_nr
    selected_nr = event.item
    dropdown_nrs.label = f"Izbran: {selected_nr}"

dropdown_nrs.on_click(update_nr)

# -------------------------------
# Callback to handle file uploads via FileInput
# -------------------------------
def upload_files(attr, old, new):
    global files_uploaded
    names = file_input.filename
    values = file_input.value
    if not names or not values:
        file_list_div.text = "Ni naloženih datotek"
        files_uploaded = False
        return
    if isinstance(names, str):
        names = [names]
    if isinstance(values, str):
        values = [values]

    uploaded_dfs.clear()
    items_html = []

    for fname, b64 in zip(names, values):
        try:
            decoded = base64.b64decode(b64)
            bio = io.BytesIO(decoded)
            df = pd.read_csv(bio, header=None)
        except Exception as e:
            print(f"Failed to parse {fname}: {e}")
            continue

        m = re.match(r'^(\d+)-(\d+)-(\d+)-(\d+)\.csv$', fname)
        if m:
            signal_code, id_code, nr_code, suffix = m.groups()
            uploaded_dfs[(signal_code, id_code, nr_code)] = df
            items_html.append(f"<li>{fname}</li>")
            files_uploaded = True
        else:
            print(f"Filename {fname} doesn't match expected pattern.")

    file_list_div.text = "<b>Datoteke so naložene</b>" if files_uploaded else "<b>Ni naloženih datotek</b>"

# -------------------------------
# Callback to handle auto-upload from data folder
# -------------------------------
def auto_upload_from_folder():
    global files_uploaded
    #print(f"Checking folder: {DATA_FOLDER}") 
    uploaded_dfs.clear()
    valid_upload = False
    loaded_files = []

    if not os.path.exists(DATA_FOLDER):
        file_list_div.text = f"Mapa '{DATA_FOLDER}' ne obstaja!"
        print(f"Error: Folder '{DATA_FOLDER}' does not exist")
        files_uploaded = False
        return

    for fname in os.listdir(DATA_FOLDER):
        if not fname.endswith('.csv'):
            continue
        file_path = os.path.join(DATA_FOLDER, fname)
        try:
            df = pd.read_csv(file_path, header=None)
            m = re.match(r'^(\d+)-(\d+)-(\d+)-(\d+)\.csv$', fname)
            if m:
                signal_code, id_code, nr_code, suffix = m.groups() 
                uploaded_dfs[(signal_code, id_code, nr_code)] = df
                loaded_files.append(fname)
                valid_upload = True
            else:
                print(f"Filename {fname} doesn't match expected pattern.")
        except Exception as e:
            print(f"Failed to parse {fname}: {e}")
            continue

    file_list_div.text = "<b>Datoteke so naložene</b>" if valid_upload else "<b>Ni veljavnih datotek v mapi!</b>"
    files_uploaded = valid_upload

file_input.on_change('filename', upload_files)
auto_upload_button.on_click(auto_upload_from_folder)

# -------------------------------
# Draw data based on uploads and selections
# -------------------------------
def draw_data():
    global files_uploaded
    if not selected_signals or not selected_IDs or selected_nr is None or not files_uploaded:
        p.title.text = "Ena ali več možnosti v spustnem seznamu ni izbranih ali datoteke niso naložene!"
        return

    p.renderers = []
    if p.legend:
        p.legend.items = []

    any_data_plotted = False
    color_index = 0
    for signal in selected_signals:
        for sel_id in selected_IDs:
            key = (signal, sel_id, selected_nr)
            if key in uploaded_dfs:
                df_main = uploaded_dfs[key]
                min_val, max_val = df_main[3].min(), df_main[3].max()
                norm = (df_main[3] - min_val) / (max_val - min_val)
                cds_main = ColumnDataSource({'time': df_main[2], 'signal': norm})
                color = colors[color_index % len(colors)]
                p.line(
                    'time', 'signal', source=cds_main,
                    legend_label=f"Signal {signal}-{sel_id}-{selected_nr}",
                    line_width=2, color=color
                )
                color_index += 1
                any_data_plotted = True
            else:
                print(f"No upload for signal file {signal}-{sel_id}-{selected_nr}-000.csv")

            key020 = ('020', sel_id, selected_nr)
            if key020 in uploaded_dfs:
                df_020 = uploaded_dfs[key020]
                cds_020 = ColumnDataSource({'time': df_020[2], 'signal': df_020[4]})
                p.scatter(
                    'time', 'signal', source=cds_020,
                    legend_label=f"Anks 020-{sel_id}-{selected_nr}",
                    marker='square', size=8,
                    y_range_name='right',
                    fill_alpha=1, line_color='black', line_width=1,
                    color=colors[(color_index-1) % len(colors)]
                )
                any_data_plotted = True
            else:
                print(f"No upload for anks file 020-{sel_id}-{selected_nr}-000.csv")

    if not any_data_plotted:
        p.title.text = "Ni podatkov za izris: preverite, ali so izbrane datoteke pravilne!"
        return

    p.y_range.start, p.y_range.end = 0, 1
    p.extra_y_ranges['right'].start, p.extra_y_ranges['right'].end = 0, 1
    p.title.text = "Vrednosti signala skozi čas"

button.on_click(draw_data)

# -------------------------------
# Layout and add to document
# -------------------------------
#file_column = column(Div(text="<b>Izberi datoteke</b>"), file_input, file_list_div, auto_upload_button)
file_column = column(Div(text="<b>Naloži datoteke</b>"), auto_upload_button, file_list_div)
signals_column = column(Div(text="<b>Izbira signala</b>"), multi_select_signals)
ids_column = column(Div(text="<b>Izbira uID</b>"), multi_select_IDs)
controls_column = column(Div(text="<b>Izbira št. poskusa</b>"), dropdown_nrs, button)
controls = row(signals_column, ids_column, controls_column, file_column)
layout = column(controls, p)

curdoc().add_root(layout)
curdoc().title = "Zaslon 1"