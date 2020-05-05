Setting up Lattice's LabRAD + ARTIQ locally on Windows
==============
Before you start, if you have anything left over from a previous setup, go ahead and delete it. For example, you should delete:
  - `C:\LabRAD`, if it exists
  - Anything LabRAD or ARTIQ related under `C:\Repos` or `C:\Users\<username>`
  - Any shortcuts you may have for starting LabRAD or ARTIQ
  
Now follow these steps carefully to get the necessary Lattice LabRAD and ARTIQ components running on your Windows machine.

1. Install Anaconda, if you don't have it already, from:
https://repo.anaconda.com/archive/Anaconda3-2020.02-Windows-x86_64.exe.
Keep all the installation defaults, as some later parts of the setup depend on this.

1. Install Visual Studio Build Tools from:
https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16.
During the setup, go to the `Individual Components` tab and choose to install the following components:
    - `MSVC v140 - VS 2015 C++ build tools (v14.00)`
    - `Windows 10 SDK (10.0.18362.0)`

1. Install VC++ Compiler for Python 2.7 from:
https://www.microsoft.com/en-us/download/confirmation.aspx?id=44266.
This is also required while setting up the conda environments.

1. Run the following command exactly to copy required files to their correct location:
    - `copy "c:\Program Files (x86)\Windows Kits\10\bin\10.0.18362.0\x64\rc*" "c:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin"`

1. Download this complete package of all the LabRAD and ARTIQ stuff you will need: 

    **[Lattice-LabRAD-ARTIQ-Windows-package](https://www.dropbox.com/sh/qe9fhxtldfolaqv/AADFkTKwL77O3U7sHsn5t9nIa?dl=1)**

    (It's about 900 MB, may take a little while.) Once this finishes downloading, unzip it somewhere convenient, maybe `C:\Artiq-Windows`.

1. After the Anaconda installation has finished, open an Anaconda Prompt
and run `conda-recreate-envs.bat` from the `C:\Artiq-Windows` folder.
This will take a while - it will delete any `artiq` or `labrad` conda environments you might have,
and recreate them from scratch. You can continue with the other steps while this is going.

1. Unzip the `LabRAD (unzip to C drive root).zip` file to `C:\`. This should create a `C:\LabRAD` folder.

1. Unzip the `Repos (unzip to user home folder).zip` to `C:\Users\<your_user_name>`.  This should create a number
of folders directly in your `C:\Users\<username>` folder:
    - `artiq`
    - `artiq-master`
    - `artiq-work`
    - `labrad`
    - `.labrad` (with a period)
      - This folder is important; it contains the full parameter vault. If you don't see it, ensure that "show hidden files" is enabled
      in your File Explorer settings.
      
    Also note that there are clones of several GitHub repos here, and if you install Git/GitHub tools on your machine you'll be able
    to sync these to pull any changes that have been pushed since you first set this up.

1. Open a command prompt and run `set-labrad-env-variables.bat` from the `C:\Artiq-Windows` folder.

1. Install Java Runtime if you don't have it already by running `Java Runtime 8u241 x86 (required for LabRAD).exe`
from the `C:\Artiq-Windows` folder. LabRAD requires this.

1. Double-click the `lab-artiq-putty.reg` file from the `C:\Artiq-Windows` folder.
This adds a PuTTy configuration file with tunnels that allow communication with the ARTIQ hardware in the lab.

1. Make sure that the Anaconda environment creation completed successfully by typing `conda env list`. You should see both `artiq` and `labrad` in the list.

Setting up Julia to run simulations with IonSim
==============
If you want to run local simulations from the Lattice ARTIQ Dashboard, you'll need to set up Julia, IonSim, and PyJulia:

1. Install Julia from https://julialang.org/downloads/.
1. From a Julia prompt, use `Sys.BINDIR` to see where Julia's `bin` folder is:
    ```
    julia> Sys.BINDIR
    "C:\\Path\\To\\Julia\\Julia-1.4.1\\bin"
    ```
    Then add that folder to your system `PATH` environment variable. 
    
    > 🛈 To add a folder to your system `PATH` in Windows: Type `sysdm.cpl` in the Start menu,
    > go to the "Advanced" tab, click "Environment Variables", find "Path", click "Edit...",
    > and add the folder (with double-slashes replaced by single-slashes) to the list.
    > You must restart all open command prompts for the change to take effect.
    
1. From a Julia prompt, run the following commands to install the required packages:
    ```julia
    using Pkg
    Pkg.add(PackageSpec(url="https://github.com/HaeffnerLab/IonSim.jl.git"))
    Pkg.add("QuantumOptics")
    Pkg.add("PyCall")
    ```
1. From an Anaconda prompt, run the following commands to ensure that PyJulia is installed in your `artiq` environment:
    ```
    conda activate artiq
    pip install julia==0.5.3
    python -c "import julia; julia.install()"
    ```

Running Lattice's LabRAD + ARTIQ locally on Windows
==============
After you've completed the above installation steps, here are the steps to get ARTIQ Dashboard running:
1. _(Optional, only if you want to run things on the real ARTIQ hardware)_ Connect to the lab tunnel: Open `PuTTy.exe` (you should have a copy in `C:\Artiq-Windows`),
load the "Haeffner Lab with Tunnels" profile, and click Open.
You'll have to login with a valid lab username and password, either yours or lab-user. Just minimize the shell window after you have logged in.
1. To start LabRAD and all necessary servers, run `C:\LabRAD\start_labrad.bat`. Wait a few seconds for this to complete.
    - _Note:_ here and elsewhere below, if you get a Microsoft SmartScreen warning about the file being unsafe, click `More info` and then click `Run anyway`.
1. To start the ARTIQ Master, run `C:\Users\<username>\artiq\artiq_master_start.bat`.
1. To start the ARTIQ Dashboard, run `C:\Users\<username>\artiq\artiq_dashboard_start.bat`.
1. To start the Real Complicated Grapher, run `C:\Users\<username>\artiq\artiq_grapher_start.bat`.

After several seconds, the ARTIQ dashboard and grapher should load successfully. If you didn't connect to the lab tunnel, you'll see a bunch of error messages about being unable to communicate with the hardware, but these can be ignored.

Troubleshooting Common Issues
==============
**Issue:** If your computer unexpectedly shuts down or crashes while ARTIQ is running, you may see a failure
the next time you try to start ARTIQ Master, due to a corrupted `dataset_db.pyon` file.  
**Fix:** Delete the file at `C:\Users\<username>\artiq-master\dataset_db.pyon`. It will be automatically recreated on the next run.

**Issue:** Running ARTIQ Master and/or Dashboard gives a matplotlib error like:
````
  File "C:\Users\ryan\anaconda3\envs\artiq\lib\site-packages\matplotlib\font_manager.py", line 264, in findSystemFonts
    fontfiles.update(win32InstalledFonts(fontext=fontext))
TypeError: 'NoneType' object is not iterable
````
**Fix:** Open the file `%USERPROFILE%\anaconda3\envs\artiq\lib\site-packages\matplotlib\font_manager.py` and change line 210 from `return None` to `return []`. (Credit [here](https://github.com/matplotlib/matplotlib/issues/12439#issuecomment-427743646) for this hacky fix.)
