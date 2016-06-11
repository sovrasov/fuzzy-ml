.PHONY: all clean texfirst texsecond builddir refs

builddir:
	mkdir -p build

texfirst: ${IMAGES} | builddir
	xelatex -output-directory=build report.tex

refs: texfirst
	bibtex build/report.aux

texsecond: refs
	xelatex -output-directory=build report.tex

clean:
	rm -r build/
	rm -r images/*.pdf

all: texsecond
