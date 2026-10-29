"""Keep figure labels readable at the actual MDPI text width, in points."""
from matplotlib.text import Text

def journal_layout(fig,target_width_points=392.88,minimum_print_points=8):
    # A figure made at ten inches is reduced when printed at 5.46 inches.
    # Account for that reduction before layout rather than only increasing DPI.
    floor=minimum_print_points*fig.get_figwidth()*72/target_width_points
    for artist in fig.findobj(match=Text):
        if artist.get_text():artist.set_fontsize(max(artist.get_fontsize(),floor))
