#!/bin/bash

cd tex 
echo "Tangling...."
notangle  -Rtspnng.py mainlitprog.nw > ../src/tspnng.py
echo "Staging tspnng.py...."
git add ../src/tspnng.py
echo "Done"
cd ..