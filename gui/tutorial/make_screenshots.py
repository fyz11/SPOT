""" open the SPOT GUI on a project and save screenshots of its tabs (optionally in several themes)

python make_screenshots.py <gui dir> <spot_project.json> <out dir> <prefix> <theme>[,<theme>...] [tabs e.g. 0,1,2,3,4] [qc video] [result image substring]
"""
import os
import sys
import ctypes

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

gui_dir, project, out, prefix, themes = sys.argv[1:6]
tabs = [int(t) for t in (sys.argv[6] if len(sys.argv) > 6 else '0,1,2,3,4').split(',')]
qc_video = sys.argv[7] if len(sys.argv) > 7 else ''
result_key = sys.argv[8] if len(sys.argv) > 8 else ''
sys.path.insert(0, gui_dir)
import spot_gui  # noqa: E402

# never touch the user's GUI settings or the project files
spot_gui.SETTINGS_FILE = os.path.join(out, 'gui_settings_tmp.json')
spot_gui.save_settings = lambda d: None
from PIL import ImageGrab  # noqa: E402

os.makedirs(out, exist_ok=True)
app = spot_gui.SpotApp(None)
app.save_project = lambda quiet=False: True
app.geometry('1400x900+20+20')
app.open_project(project)


def settle(ms=700):
    end = [False]
    app.after(ms, lambda: end.__setitem__(0, True))
    while not end[0]:
        app.update()


def grab(name):
    settle()
    x, y = app.winfo_rootx(), app.winfo_rooty()
    w, h = app.winfo_width(), app.winfo_height()
    ImageGrab.grab(bbox=(x, y, x + w, y + h), all_screens=True).save(os.path.join(out, name + '.png'))
    print('saved', name)


app.deiconify()
app.lift()
app.attributes('-topmost', True)
settle(1500)
for theme in themes.split(','):
    app._apply_theme(theme)
    app.theme_var.set(theme)
    tag = theme.replace(' ', '').lower()
    for t in tabs:
        app.nb.select(t)
        settle(500)
        if t == 3 and qc_video:
            app.qc_video.set(qc_video)
            app.qc_load()
            app.qc_frame.set(20)
            app.qc_scale.set(20)
            app.qc_draw()
        if t == 4 and result_key:
            for item in app.result_tree.get_children():
                for sub in app.result_tree.get_children(item) or [item]:
                    if result_key in sub:
                        app.result_tree.selection_set(sub)
                        app.result_tree.see(sub)
            settle(300)
            app._show_result_image()
        grab('%s_%s_tab%d' % (prefix, tag, t + 1))
app.destroy()
