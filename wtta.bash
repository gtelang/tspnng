#!/usr/bin/env bash
#----------------------------------------
# [W]eave,[T]angle,[T]est,[A]ttach
# A Harness script for the entire
# literate code base, for performing
# each of the above stapes.
# Programs needed for running THIS script
# - PANDOC 
# - PDFTK
# - NOWEB
#----------------------------------------
main_tex_file="main.tex"
main_aux_file="main.aux"
latex_compiler="pdflatex"
progname="tspnng"
# Exit when any command fails. A Tip from 
# https://intoli.com/blog/exit-on-errors-in-bash-scripts/ 
set -e
if [ $# -eq 0 ]; then
    echo "......................................................................... " >  wtta.log
    printf "Compilation of project started on:  "                                     >> wtta.log
    date                                                                              >> wtta.log 
    echo "...................System Info (Output of uname -a)...................... " >> wtta.log
    uname -a                                                                          >> wtta.log
    echo "....................Processor Details (Output of lscpu)..................." >> wtta.log
    lscpu                                                                             >> wtta.log
    #-------------------------------------------------------------------------------------
    # Weave the documentation into a LaTeX file. 
    # TODO: The tangle for each source file should happen to /tmp and brught in only if 
    # the orginal file and the new tangled file are not byte equal. byte equality of two 
    # files can be found with the `cmp` tool https://stackoverflow.com/a/12900693. 
    # Basically notangle should be wrapped around with cmp inside a function that 
    # passes the same flags down to notangle. Call this function notangle-smart. Also the 
    # same can be used for asymptote sripts so that asymptote does not have to do heavy 
    # processing for rendering images
    #--------------------------------------------------------------------------------------
    echo "......................................................."                        >> wtta.log
    printf "Weaving ..."
    cd tex
    noweave  -n -index -latex mainlitprog.nw > mainlitprog.tex
    $latex_compiler -interaction=nonstopmode -halt-on-error  -shell-escape $main_tex_file >> ../wtta.log 2>&1
    #asy asy2d/fig1.asy
    bibtex $main_aux_file                                                                 >> ../wtta.log 2>&1
    $latex_compiler -interaction=nonstopmode -halt-on-error  -shell-escape $main_tex_file >> ../wtta.log 2>&1
    $latex_compiler -interaction=nonstopmode -halt-on-error  -shell-escape $main_tex_file >> ../wtta.log 2>&1
    rm --force -f *-blx.bib *.aux *.bbl *.bcf *.blg *.ind *.idx *.ilg \
                  *.log *.out *.pbsdat *.prc *.pre *.run.xml *.tdo *.toc  *~
    printf " ( done )\n"
    #-------------------------------------------------------------------------
    # Tangle the source code corresponding to each chunk into appropriate file
    #-------------------------------------------------------------------------
    printf "Tangling ..."
    notangle  -Rtspnng.py mainlitprog.nw > ../src/tspnng.py # the <<tspnng.py>> chunk
    printf " ( done )\n"
    #------------------------------------------------------------------------
    # Run all tests
    #------------------------------------------------------------------------
    cd ../tests
    printf "Running Tests ..."
    printf " ( done )\n"
    #----------------------------------------------------------------------------------------
    # Move the main pdf file (and executable/main.py files) to the home directory of program
    # Also convert manpage section inside the TeX file into a README file formatted as markdown. 
    # This README file is then typically displayed on your projects github's homepage. 
    #----------------------------------------------------------------------------------------
    printf "Finishing up ..."
    cd .. # Move back to home directory. 
    mv tex/main.pdf .                      # weaved document moved to home folder
    cat tex/manpage.tex | tail -n +2 | pandoc -f latex -t markdown -s - -o README.md # manpage at the top of the file converted into README file. 
                                           # Helps keep README file in sync with the first page. Use awk to strip away
					   # any remaining bits of irritating code that stays in even after conversion
    printf " ( done )\n"
    #----------------------------------------------------------------------------------------------------------------------
    #  Attach all source and test files tangled to the pdf file using pdftk
    #  There are 2 ways to do this
    #  1. With your pdf reader like Adobe Acrobat Reader(Windows/Mac)(should be in menu somewhere, just explore) or Zathura(GNU/Linux) 
    #     (use :export attachment-code.zip ~/Desktop/code.zip) or Evince(GNU/Linux) (right click and click save attachment)
    #  2. Or with the extremely powerful PDFTK commandline tool  as `pdftk main-attachment.zip unpack_files`
    # Everytime the pdf file is distributed, the source code goes with it. And so no-one has to worry about
    # misplacing tex files and wondering what the TeX file used to generate a particular pdf file and the results
    # looked like. This is kind of an inversion, now the pdf file is the main point of reference for the whole code 
    # base rather the the code tree. Makes it easier to understand the whole system.
    #-----------------------------------------------------------------------------------------------------------------------
    mkdir ${progname}
    cp -r src tests lib bin makefile README.md .git .gitignore ${progname} # needs to be lightly edited depending on the project. Keep the .git folder!
                                                                    # It helps to keep track of changes to all files in src as if everything was written 
								    # in a non-literate way. The appearance of stuff in the `code` should be as if 
								    # it were written in a non-literate way. The only evidence it was written literately
								    # would be the pdf file I will be  attachoing the code.tar.gz file to. 
    tar -cvzf ${progname}.tar.gz ${progname} > /dev/null 2>&1 # compress everything in `code` directory into a .tar.gz file. And silence the damn thing. 
    pdftk main.pdf attach_files ${progname}.tar.gz output main-attach.pdf  # Attach files via pdftk. Using *original* filename 
                                                                    # as destination gives rise to problems as mentioned on 
		                           			    # https://frommindtotype.wordpress.com/2018/09/17/how-to-attach-a-file-to-a-pdf/
    rm -r main.pdf ${progname} ${progname}.tar.gz
    mv main-attach.pdf main.pdf
    echo "...... FINSHED! ..... " 
fi
