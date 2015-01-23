Name: Nirmesh Khandelwal
Unity Id: nbkhande
Project 4
Assigned paper: Paper 1


Language used to implement: python version 3 
Development OS: Ubuntu 14.04.1

-------------------------------------------------------------------------
Dependencies needed to be installed:

networkx==1.9.1
matplotlib==1.3.1

Installation steps:
1) install networkx using python 3: 
$sudo pip3 install networkx

2) install matplotlib
$sudo pip3 install matplotlib

-------------------------------------------------------------------------
Running Step:

Script File Name: anomaly_detection.py

Run the main anamoly detection script:

$python3 anomaly_detection.py <data_directory_path> <outputFile_name> [<MR multiplier>]
Here the MR multiplier (in formula for threshold 'median - N * MR' ) is optional. By default it takes as 3

e.g:

Example Without providing MR multiple in threshold formula
$python3 anomaly_detection.py /home/nirmesh/GDM/data/as-733 as733-out

Example with Providing the value of multiple of MR
$python3 anomaly_detection.py /home/nirmesh/GDM/data/enron_noempty/ enron_noempty-out 2


-------------------------------------------------------------------------

Data set used to test the program: as-733, enron_noempty, p2p-Gnutella

Output file: For each graph, it produces 2 output files: 1 text file containing anomalies detected. Format of this file is same as specified in the project requirement. Second the PNG file of the similarity plot.

as733-out, as733-out.png
enron_noempty-out, enron_noempty-out.png
p2p-Gnutella-out, p2p-Gnutella-out.png


Output file format: Text files are in same format as specified in the project requirement.

---------------------------------------------------------------------------

Reference Citations:

* The implementation is based on paper “Web Graph Similarity for Anomaly Detection by Panagiotis Papadimitriou, Ali Dasdan and Hector Garcia-Molina.”
* I have used networkx library in python in order to calculate the Pagerank.
* Matplotlib library in Python for ploting.
* For testing, I have used the datasets provided by course website only.





