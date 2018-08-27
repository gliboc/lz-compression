
all:
	# if [ ! -d "target" ]; then
	# 	mkdir target
	# fi
	latexmk -outdir=target -bibtex -pdf main.tex && mupdf rapport.pdf
	# if [ ! -L "rapport.pdf" ]; then
	# 	ln -s target/main.pdf rapport.pdf
	# fi

debug:
	latexmk -Werror -diagnostics -outdir=target -bibtex -pdf main.tex && mupdf rapport.pdf

clean:
	latexmk -c main.tex
	cd target && latexmk -c -f main.tex

cleanall:
	latexmk -c

