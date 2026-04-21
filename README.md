Steps to autogenerate plasmids for PichAI Project

1) Download Git Repository.
2) Make sure to have python3 and pip3 installed.
3) In the folder, call python3 -m venv .venv
4) Activate virtualenvironment via source .venv/bin/activate (Linux, MacOS) or .venv\Scripts\activate (Windows in CMD Shell).
5) Install required packages via pip install -r requirements.txt
6) Copy an xlsx file (see template) into your folder, e.g. create a folder called input with targets.xlsx in it
7) Run python generate.py -i input/targets.xlsx.
8) Find your plasmid maps, plasmid and primer list in the folder output (date and time stamp).
9) Use python generate.py --help to read about further arguments (such as start number of primers and plasmids).
