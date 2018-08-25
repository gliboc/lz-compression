
all:
	# if [ ! -d "target" ]; then
	# 	mkdir target
	# fi
	latexmk -outdir=target -pdf -bibtex report.tex
	# if [ ! -L "rapport.pdf" ]; then
	# 	ln -s target/report.pdf rapport.pdf
	# fi
	mupdf rapport.pdf

plan:
	if [ ! -d "target" ]; then
		mkdir target
	fi
	latexmk -outdir=target -pdf -bibtex main.tex
	mupdf target/main.pdf

clean:
	latexmk -c report.tex
	cd target && latexmk -c -f report.tex

cleanall:
	latexmk -c

