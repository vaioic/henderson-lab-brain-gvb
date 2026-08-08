# GVB particle analysis (Henderson Lab)

This project is to build an image analysis pipeline to analyze images of granulovacuolar
degeneration bodies (GVB), acquired using serial staining and imaging.

## Usage

To process the images, this project utilizes several different tools:

* Fiji/ImageJ with the **Image Stitching** package (Version 3.1.9)
* Python (Version 3.14)

### Image acquisition

Sequentially stained samples must typically be registered (aligned) to account for
translational shifts. To do this, each image round must have a DAPI channel that can be
used for registration.

The original dataset was acquired using an ImageXpress Confocal HT.ai (Molecular
Devices). For this project, the individual image for each tile is required - do not tile
the final image.

It is also expected that the filename for each tile has the following pattern:

``<filename>_s<tile>_w<channel>.TIF``

For example: ``AM1c-s11-r002_A01_s1_w2.TIF``

### Setup and installation

#### Using uv (Recommended)

This project uses [uv](https://docs.astral.sh/uv/) to manage virtual environments and dependencies. 

1. Install ``uv``
    * **macOS or Linux:** ``curl -LsSf https://astral.sh/uv/install.sh | sh``
    * **Windows:** ``powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"``
    
    To check if you have ``uv`` installed, open a terminal and run ``uv --version``.

2. Clone the repository
   ```bash
   git clone git@github.com:vaioic/henderson-lab-brain-gvb.git
   cd henderson-lab-brain-gvb
   ```

3. Sync the environment (this will setup the correct virtual environment and dependencies)
   ```bash
   uv sync
   ```

4. Run the illumination correction (Needs to be run on every image channel separately -
   see section below):
   ```bash
   uv run python -m shared.gvb_analyzer -i "..\data\Dataset 1\raw\AM1c-s11-r002_Plate_4555\TimePoint_1" -o "../processed/20260808 Dev" -p "*_w1.TIF"
   ```

5. Stitch the image in Fiji (see below)

6. Run the analysis
   ```bash
   uv run analysis/analysis_script.py
   ```

#### Using venv and pip

1. Clone the repository
   ```bash
   git clone git@github.com:vaioic/henderson-lab-brain-gvb.git
   cd henderson-lab-brain-gvb
   ```

2. Create a virtual environment
   ```bash
   python -m venv venv
   ```

3. Activate the environment
   ```bash
   # macOS/Linux
   source ./venv/bin/activate

   # Windows (PowerShell)
   .\venv\Scripts\Activate.ps1
   ```

4. Install the repository as an editable module
   ```bash
   python -m pip install -e .
   ```

5. Run the illumination correction (Needs to be run on every image channel separately -
   see below):

   ```bash
   python -m shared.gvb_analyzer -i "..\data\Dataset 1\raw\AM1c-s11-r002_Plate_4555\TimePoint_1" -o "../processed/20260808 Dev" -p "*_w1.TIF"
   ```

6. Stitch the images using Fiji (see below)

7. Run the analysis script
   ```bash
   python -m analysis.analysis_script

   # or
   python analysis/analysis_script.py
   ```

### Shading correction

The original dataset exhibited strong vignetting along
the edge of each tile. The pipeline attempts to correct for this by stacking the tiles to
estimate the shading, then applying a correction to each tile. The shading appears to be
different for each channel.

Shading correction can be applied using the CLI, for example:

```bash
# Using uv (see below)
uv run python -m shared.gvb_analyzer -i "..\data\Dataset 1\raw\AM1c-s11-r002_Plate_4555\TimePoint_1" -o "../processed/20260808 Dev" -p "*_w1.TIF"

# Alternatively if using pip/venv:
python -m shared.gvb_analyzer -i "..\data\Dataset 1\raw\AM1c-s11-r002_Plate_4555\TimePoint_1" -o "../processed/20260808 Dev" -p "*_w1.TIF"
```

For help:

```bash
python -m shared.gvb_analyzer --help
```

### Stitching the large image

To obtain the final stitched large image, we stitch on the DAPI image. The resulting
stitching coordinates must be saved, then applied to the remaining channels of the
dataset.

For the DAPI channel:

1. In Fiji, open the Stitching plugin: **Plugins** > **Stitching** > **Grid/Collection
   stitching**

2. In the dialog box, set the following:
   - Type: **Grid: row-by-row**
   - Order: **Right & Down**
   - Click **Ok**

3. In the next dialog box, set the following:
  - Grid size x: **7** (See note below)
  - Grid size y: **9**
  - Tile overlap: **20%**
  - Directory: Click on **Browse** to select the directory with the shading corrected
    DAPI tiles
  - File names for files: **img_{ii}.tif**
  - Check **Compute overlap**
  - Image output: **Write to disk**
  - Leave all other settings as default/unchecked
  - Click **OK**

4. In the next dialog box, click **Browse** and select the directory to save the
   stitched large image. Click **OK** to start the stitching.

5. Open the stitched file and check the final result looks good without significant cell
   shearing.

6. Copy the ``TileConfiguration.txt`` and ``TileConfiguration.registered.txt`` to the
   other folders.

Note: The number of X and Y tiles can be obtained from the HTD-file supplied with the
images. HTD-file can be opened using a text editor. Search for XSites and YSites.

For the remaining channels:

1. In Fiji, open the Stitching plugin: **Plugins** > **Stitching** > **Grid/Collection
   stitching**

2. In the dialog box, set the following:
   - Type: **Positions from file**
   - Order: **Defined by TileConfiguration**
   - Click **Ok**

3. In the next dialog box, set the following:
  - Directory: Click on **Browse** to select the directory with the shading corrected tiles
  - Layout file: **TileConfiguration.txt**
  - Check **Compute overlap**
  - Image output: **Write to disk**
  - Leave all other settings as default/unchecked
  - Click **OK**. 

4. In the next dialog box, click **Browse** and select the directory to save the
   stitched large image. Click **OK** to start the stitching.


## Issues

If you encounter any issues with running the code or have any questions, please create an [Issue](https://github.com/vaioic/henderson-lab-brain-gvb/issues) or send an email to opticalimaging@vai.org. If you are reporting a bug, please include any error messages to aid with troubleshooting.

## License

This project is licensed under the GPLv3 License. See the [LICENSE](LICENSE) file for details.

## Citing & Acknowledgements

This repository is publicly available for open-source use, but it is developed and maintained by the Optical Imaging Core at the Van Andel Institute. If code from this repository contributed to data used in a publication, abstract, or presentation, please cite and acknowledge our work based on your affiliation:

### For External Users
Please cite this repository and acknowledge the author(s) in your publication's materials, methods, or acknowledgements section:
> "Image analysis pipelines were adapted from open-source tools developed by the Optical Imaging Core at the Van Andel Institute (GitHub:[henderson-lab-brain-gvb](https://github.com/vaioic/henderson-lab-brain-gvb))."

If you require custom adjustments or advanced analysis support, please contact us at opticalimaging@vai.org.

### For Internal Users & Close Collaborators
If you are an internal researcher or an external collaborator working directly with our staff, please include our Research Resource Identifier (RRID) in your materials and methods section:
> "Image analysis and data processing were performed in collaboration with the Optical Imaging Core at the Van Andel Institute (RRID:SCR_021968)."

Please review the Acknowledgement and Authorship Guidelines on [VAI's Core Technology and Services website](https://vanandelinstitute.sharepoint.com/sites/Cores/SitePages/Acknowledgements-and-Authorship.aspx)

### Contributors
<a href="https://github.com/vaioic/henderson-lab-brain-gvb/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=vaioic/henderson-lab-brain-gvb" />
</a>

## Changelog

### v0.0.1 (2026-06-15)
* Initial commit with preliminary code ([OIC-320](https://varioic.atlassian.net/browse/OIC-320))