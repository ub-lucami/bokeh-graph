# -------------------------------
# Base directory and configuration
# -------------------------------
import os
from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, MultiSelect, Dropdown, Button, Range1d, LinearAxis, Div
from bokeh.layouts import column, row
import pandas as pd

SRC_PATH = os.getcwd()
ROOT_PATH = os.path.abspath(os.path.join(SRC_PATH, "../"))
data_dir = f'{ROOT_PATH}/data/MatAnx_Data_all'
colors = ["blue", "green", "red", "orange", "purple", "brown", "pink", "gray", "cyan", "lime", "magenta"]

# -------------------------------
# Create the Bokeh plot
# -------------------------------
p = figure(title="Vrednosti signala skozi čas", x_axis_label='Čas[s]', y_axis_label='Vrednost signala', width=1200)

# -------------------------------
# Dictionaries to hold sources for multiple IDs and Signals
# -------------------------------
sources = {}
sources_020 = {}

# -------------------------------
# Configure second y-range (right side axis)
# -------------------------------
p.extra_y_ranges = {"right": Range1d(start=0, end=1)}
p.add_layout(LinearAxis(y_range_name="right", axis_label="Vrednost anksioznosti"), 'right')

# -------------------------------
# MultiSelect widget for signals and IDs
# -------------------------------
available_signals = ['021', '022', '023', '024', '025', '026', '027']
multi_select_signals = MultiSelect(options=available_signals)

available_IDs = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011']
multi_select_IDs = MultiSelect(options=available_IDs)

# -------------------------------
# Dropdown menu for numbers (only single selection allowed)
# -------------------------------
available_nrs = ['001', '002', '003', '004', '005']
menu_nrs = [(noe, noe) for noe in available_nrs]
dropdown_nrs = Dropdown(label="Izberi št. poskusa", menu=menu_nrs)

# -------------------------------
# Add titles using Div elements
# -------------------------------
signals_title = Div(text="<b>Izbira signala</b>")
ids_title = Div(text="<b>Izbira uID</b>")
noe_title = Div(text="<b>Izbira št. poskusa</b>")

# -------------------------------
# Organize widgets in layout columns
# -------------------------------
signals_column = column(signals_title, multi_select_signals)
ids_column = column(ids_title, multi_select_IDs)
noe_column = column(noe_title, dropdown_nrs)

# -------------------------------
# Button to trigger action (Plot)
# -------------------------------
button = Button(label="Izriši")

# -------------------------------
# Variables to store selected data
# -------------------------------
selected_signals = []
selected_IDs = []
selected_nr = None

# -------------------------------
# Define callbacks for widget selections
# -------------------------------
def update_signals(attr, old, new):
    global selected_signals
    selected_signals = new

def update_IDs(attr, old, new):
    global selected_IDs
    selected_IDs = new

def update_nr(event):
    global selected_nr
    selected_nr = event.item
    dropdown_nrs.label = f"Izbran: {selected_nr}"

def draw_data():
    global sources, sources_020
    if not selected_signals or not selected_IDs or selected_nr is None:
        p.title.text = "Ena ali več možnosti v spustnem seznamu ni izbranih!"
        return

    # -------------------------------
    # Clear previous plots and legend
    # -------------------------------
    p.renderers = []
    if p.legend:
        p.legend.items = []

    color_index = 0
    for signal in selected_signals:
        for selected_ID in selected_IDs:
            source_key = f"{signal}-{selected_ID}-{selected_nr}"
            if source_key not in sources:
                sources[source_key] = ColumnDataSource(data={'time': [], 'signal': []})
                sources_020[source_key] = ColumnDataSource(data={'time': [], 'signal': []})

            source = sources[source_key]
            source_020 = sources_020[source_key]

            # -------------------------------
            # Load the main signal
            # -------------------------------
            file_path = os.path.join(data_dir, f"{signal}-{selected_ID}-{selected_nr}-000.csv")
            if os.path.exists(file_path):
                try:
                    new_df = pd.read_csv(file_path, header=None)
                    min_val = new_df[3].min()
                    max_val = new_df[3].max()
                    normalized_signal = (new_df[3] - min_val) / (max_val - min_val)

                    source.data = {'time': new_df[2], 'signal': normalized_signal}

                    color = colors[color_index % len(colors)]
                    p.line('time', 'signal', legend_label=f"Signal {signal}-{selected_ID}-{selected_nr}",
                           line_width=2, source=source, color=color)
                    color_index += 1

                except Exception as e:
                    print(f"Error loading data for {source_key}: {e}")
            else:
                print(f"File not found for {source_key}!")

            # -------------------------------
            # Load signal 020
            # -------------------------------
            file_path_020 = os.path.join(data_dir, f"020-{selected_ID}-{selected_nr}-000.csv")
            if os.path.exists(file_path_020):
                try:
                    new_df_020 = pd.read_csv(file_path_020, header=None)
                    source_020.data = {'time': new_df_020[2], 'signal': new_df_020[4]}

                    p.scatter(
                        'time', 'signal',
                        legend_label=f"Anx 020-{selected_ID}-{selected_nr}",
                        size=10, source=source_020,
                        y_range_name="right",
                        color=color,
                        marker='square',
                        fill_alpha=1,
                        line_color='black',
                        line_width=1 
                    )
                except Exception as e:
                    print(f"Error loading signal 020 for {selected_ID}: {e}")
            else:
                print(f"File for signal 020 not found for {selected_ID}!")

    # -------------------------------
    # Set axis ranges and update title
    # -------------------------------
    p.y_range.start = 0
    p.y_range.end = 1
    p.extra_y_ranges["right"].start = 0
    p.extra_y_ranges["right"].end = 1
    p.title.text = "Vrednosti signala skozi čas"

# -------------------------------
# Attach callbacks to widgets
# -------------------------------
multi_select_signals.on_change('value', update_signals)
multi_select_IDs.on_change('value', update_IDs)
dropdown_nrs.on_click(update_nr)
button.on_click(draw_data)

# -------------------------------
# Arrange layout: controls and plot
# -------------------------------
noe_column = column(noe_title, dropdown_nrs, button)
signals_column = column(signals_title, multi_select_signals)
ids_column = column(ids_title, multi_select_IDs)
controls_row = row(signals_column, ids_column, noe_column)
layout = column(controls_row, p)

# -------------------------------
# Add layout to the current document
# -------------------------------
curdoc().add_root(layout)
curdoc().title = "Zaslon 1"
