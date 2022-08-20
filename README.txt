Project Dependencies:
    nltk, pandas

=============================================================================================================

How to install dependencies: use command
    $ make install

How to run the query documents to retieve the documents:
    $ make run ARGV=<query_set_file>

Note: The query file must be of type(TAB Seperated Values) .tsv
Make sure the query_set_file is present in 21111020-ir-models folder. 
==============================================================================================================

Directory structure:
    21111020-assignment1
        |--- 21111020-ir-models
            |--- All_Code.ipynb
            |--- runner.py
            |--- run.sh
            |--- Makefile
        |--- 21111020-qrels
        |--- README.txt

==============================================================================================================

Output files: QRels_boolean.csv, QRels_tfidf.csv, QRels_bm25.csv

