
all:
	# if [ ! -d "target" ]; then
	# 	mkdir target
	# fi
	latexmk -outdir=target -bibtex -pdf report.tex && mupdf rapport.pdf
	# if [ ! -L "rapport.pdf" ]; then
	# 	ln -s target/report.pdf rapport.pdf
	# fi

plan:
	latexmk -outdir=target -pdf -bibtex main.tex
	mupdf target/main.pdf

clean:
	latexmk -c report.tex
	cd target && latexmk -c -f report.tex

cleanall:
	latexmk -c

