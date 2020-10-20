#!/bin/bash

cd tex
noweave  -n -index -latex mainlitprog.nw > mainlitprog.tex
git add mainlitprog.tex
pdflatex -shell-escape -interaction=nonstopmode main.tex
bibtex main.aux
pdflatex -shell-escape -interaction=nonstopmode main.tex
pdflatex -shell-escape -interaction=nonstopmode main.tex
./clean.sh
mv main.pdf ../README.pdf
cd ..
git add README.pdf
