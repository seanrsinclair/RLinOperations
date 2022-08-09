# RLinOperations

Materials for the first two days code demonstration at the [RL in Operations Workshop](https://www.kellogg.northwestern.edu/news-events/conference/bootcamp-reinforcement-learning.aspx) at [Northwestern University](https://www.northwestern.edu/).

The subfolders should contain the appropriate code demos for the different demonstrations.  Note that each of them have their own readme files with installation instructions to install the prerequisite python packages.  We also included PDF files of the slide presentations (note that not all of the material will be covered, but we included some extra supplementary slides for more information).

### Running a Code Demo

You will need the following software already installed on your machine:
- Visual Studio Code: https://code.visualstudio.com/Download
- Anaconda: https://www.anaconda.com/products/distribution

In order to run the first code demonstration for example, you will do the following:

1. Open Anaconda Prompt via search toolbar
2. git clone https://github.com/seanrsinclair/RLinOperations

(Note: If git is not installed then can download from: https://github.com/git-guides/install-git, can also skip this step if this code is already downloaded)

3. cd RLinOperations

(Or navigate to where your code is saved)

4. code .

This will open the visual studio code window. There will be five folders, one for the slides, and one for each of the code demos.

5. cd custom_simulator\ORSuite (or custom_simulator/ORSuite depending on platform)
6. conda env create --name custom_simulator --file environment.yml
7. conda activate custom_simulator
8. pip install -e .
9. conda install jupyter
10. jupyter notebook


You can now navigate to examples folder and open the demo.

Questions: srs429@cornell.edu
