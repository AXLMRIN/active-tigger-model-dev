from plotly.graph_objs._figure import Figure
from jinja2 import Template
from . import ROOT
# === === === === === === === === === === === === === === === === === === === ==
# LOCAL VARIABLES
# === === === === === === === === === === === === === === === === === === === ==

saving_kwargs = {
    "full_html" : False,
    "auto_play" : False,
    "include_plotlyjs" : False, 
    "include_mathjax" : False,
    "config" : {"responsive" : True} # Not sure it works nor it's useful
}

# === === === === === === === === === === === === === === === === === === === ==
# FUNCTIONS
# === === === === === === === === === === === === === === === === === === === ==

def export(all_figures : dict[str:Figure], N_best : int = None ):
    plotly_jinja_data = {
        figure_name : all_figures[figure_name].to_html(**saving_kwargs)
        for figure_name in all_figures 
    }
    plotly_jinja_data["N_best"] = N_best
    with open(f"{ROOT}/index.html", "w", encoding="utf-8") as output_file:
        with open(f"{ROOT}/template.html") as template_file:
            j2_template = Template(template_file.read())
            output_file.write(j2_template.render(plotly_jinja_data))