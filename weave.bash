#!/bin/bash

cd tex
noweave  -n -index -latex mainlitprog.nw > mainlitprog.tex
pdflatex -shell-escape -interaction=nonstopmode main.tex
bibtex main.aux
pdflatex -shell-escape -interaction=nonstopmode main.tex
pdflatex -shell-escape -interaction=nonstopmode main.tex
mv main.pdf ../
cd ..
