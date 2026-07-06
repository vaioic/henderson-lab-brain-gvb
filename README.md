# GVB particle analysis (Henderson Lab)

This project is to build an image analysis pipeline to analyze images of granulovacuolar
degeneration bodies (GVB), acquired using serial staining and imaging.


> [!IMPORTANT]
> **Template Setup Checklist**
> Before committing this repository, please complete the following steps:
> 
> 3. **Configuration Metadata:** Update the project metadata in `pyproject.toml`. Specifically, change the `name = "henderson-lab-brain-gvb"` line to your new project name, and update the `description` line.

> 5. **Project Tracking:** Update the `Changelog` section at the bottom with the correct Jira ticket number for this project.
> 
> *Once these steps are complete, delete this notice block entirely!*

This is the template repository for image analysis scripts. 

* Analysis scripts should be placed in the ``analysis`` folder and named using the format ``YYYYMMDD_description`` (e.g., ``20260630_batch01.py``)
* Shared scripts should be places in the ``shared`` folder
* Shared scripts can be imported into analysis scripts using:
  ```python
  from shared import core_logic

  core_logic.main()
  ```

## Usage

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

4. Run the analysis
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

5. Run the analysis script
   ```bash
   python -m analysis.analysis_script

   # or
   python analysis/analysis_script.py
   ```

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