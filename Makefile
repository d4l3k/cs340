MDS = $(shell find -name "*.md")

PDFS = $(patsubst %.md,%.pdf,$(MDS))

all: $(PDFS)

%.pdf: %.md
	- echo "Processing $<"
	- cd `dirname $<` && pandoc -V geometry:margin=0.4in -V fontfamily=sans -o `basename $@` `basename $<`
	- cd `dirname $<` && share `basename $@`

clean:
	- rm $(PDFS)

.PHONY: clean all

