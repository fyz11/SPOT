# GUI tutorial sources

`SAM-SPOT_GUI_tutorial.html` + `images/` build `../SAM-SPOT_GUI_tutorial.pdf`.

1. Screenshots: run the pipeline on a project, then (Windows; opens the GUI briefly, keep other windows away from it)

   ```shell
   python gui/tutorial/make_screenshots.py gui <results folder>/spot_project.json gui/tutorial/images org "Standard,Pink Panther,Forest Green" 0,1,2,3,4 "<video for tab 4>" <result figure name>
   ```

   The tutorial uses `org_*` (the organoid video `E5 5FU Hoechst 1`) and `ctc_*` (the Cell Tracking Challenge dataset PhC-C2DH-U373, 0.65 µm / pixel, 0.25 h / frame).
2. PDF: print the HTML with Edge or Chrome, e.g.

   ```shell
   msedge --headless=new --no-pdf-header-footer --print-to-pdf=gui/SAM-SPOT_GUI_tutorial.pdf gui/tutorial/SAM-SPOT_GUI_tutorial.html
   ```
