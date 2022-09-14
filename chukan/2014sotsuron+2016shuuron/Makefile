TARGET = rinko
TEX = platex
BIB = jbibtex
PDF = dvipdfmx

.PHONY: bbl clean cleanup

$(TARGET).pdf: $(TARGET).dvi
	$(PDF) $(TARGET)

$(TARGET).dvi: $(TARGET).aux bbl
	@echo "---------- 2nd level ----------"
	$(TEX) $(TARGET).tex
	@echo "---------- 3rd level ----------"
	$(TEX) $(TARGET).tex

bbl: $(TARGET).aux
	@echo "---------- BibTeX ----------"
	-egrep '^[^%]*\\bibliography{.*$$' $(TARGET).tex > /dev/null
	-if [ $$? -eq 0 ]; then \
		$(BIB) $(TARGET); \
	fi

$(TARGET).aux: $(TARGET).tex
	@echo "---------- 1st level ----------"
	$(TEX) $(TARGET).tex

clean:
	rm -f *.aux *.blg *.bbl \
		$(TARGET).lst $(TARGET).log \
		$(TARGET).dvi $(TARGET).out

cleanup: clean
	rm -f $(TARGET).lof $(TARGET).lot $(TARGET).toc \
		$(TARGET).pdf
