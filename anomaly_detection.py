from os import listdir
from statistics import median
import matplotlib.pyplot as plt
import networkx as nx
import hashlib
import random
import sys

def main():
    #Get List of files in the input directory mentioned
    file_list = get_file_list()

    #Calculate Signature for each file
    simHash_signature_list = get_simhash_signature_list( file_list )
    
    #The number of days for which we have graph data is equal to number of files 
    number_of_days = len(file_list)
    
    #List of similarities between consicutive graphs
    sim_list = []
    for x,y in zip( simHash_signature_list, simHash_signature_list[1:] ):
        sim = calculate_similarity(x,y)
        sim_list.append(sim)

    #Optionally, provide Multiplier of MR as command line input.
    mr_multiplier = 3
    if len(sys.argv) == 4:
        mr_multiplier = float(sys.argv[3])

    #Calculate the threshold
    threshold = calculateThreshold(sim_list, mr_multiplier)
    out_put_file = sys.argv[2]

    #Plot the results
    plot_result(sim_list, threshold, out_put_file+".png" ,mr_multiplier)

    #Calculate Anamolous Points using similarities and threshold
    anomalous_point = find_anomalous_points(sim_list, threshold)

    write_output(anomalous_point, out_put_file)
    
def get_file_list():
    """Method to retrive list of all graph files in the directory mentioned in command line argument"""
    if len(sys.argv) < 3:
        print("Please run program with following parameters:")
        print("python3 anomaly_detection.py <data_directory_path> <outputFile_name> [<MR multiplier>]")
        exit()

    data_directory = sys.argv[1] + "/"
    #Sort the file list based on integer before underscore
    file_list = sorted( listdir(data_directory), key=lambda item: (int(item.partition('_')[0])) )
    return [ data_directory+x for x in file_list ]

def get_simhash_signature_list(file_list):
    """ A function that accepts a list of files, and calculates simHash vector for each graph. It returns the list of simHash vectors"""
    simhash_list = []
    G = nx.DiGraph()
    
    for file_path in file_list:
        graph_file = open(file_path,'r')
        
        #Create a networkx graph by reading the file
        G.clear()
        for line in graph_file:
            n1,n2 = [int(x) for x in line.split(' ')]
            G.add_edge(n1,n2)
        graph_file.close()
        
        #Calculating Feature Set
        feature_set = create_graph_feature_set(G)

        #Calculate Simhash signatures based on feature set
        sim_hash_vector = sim_hash( feature_set )
        simhash_list.append( sim_hash_vector )
    return simhash_list

def create_graph_feature_set(G):  
    """This function creates a feature set of graph. Feature set will include Nodes and Edges along with their associated weights. Returns a dictionary with nodes and edges as Keys and Weights as values"""
    #Calculating the weights of all vertex. 
    page_rank =  nx.pagerank(G) 

    #The weight of each vertex is its pagerank values
    feature_set = [ (str(ti),wi) for ti,wi in page_rank.items() ]

    #Calculating the weights of edges.
    #Weight of each 'directed' edge is pagerank(from_node)/number of outlinks
    for edge in G.edges():
        ti = str(edge[0]) + ' ' + str(edge[1])
        out_degree = G.out_degree(edge[0])       
        wi = page_rank[ edge[0] ] /out_degree 
        feature_set.append( (ti,wi) )
    return feature_set

def sim_hash( feature_set ):
    """The function calculates the simhash of each graph. It receives feature set as input"""
    #Initialize a vector of b bits. here I am using md5 128 bit hashing. Hence b = 128
    V = [0]*128
    for ti, wi in feature_set:
        #Create a hash on ti
        hex_dig = hashlib.md5( ti.encode('utf-8') ).hexdigest()
        bin_repr = bin( int( hex_dig, 16) )[2:].zfill(128)       

        #+wi for each 1 in hash, -wi for each 0
        for i in range(128):
            if bin_repr[i] == '1':
                V[i] = V[i] + wi  
            else: 
                V[i] = V[i] - wi

    #Replcaing each positive number by 1 and each negative number by 0 in the simHash
    for i in range(128):
        if V[i] > 0: V[i] = 1  
        else: V[i] = 0
    return(V)

def calculate_similarity(x,y):
    """Calculates similarity between two SimHash Vectors. The formula for similarity is (1 - hamming_dist/b)"""
    #Both similarity vectors should be of equal length in order to find Hamming Distance
    assert len(x) == len(y)
    b = len(x)
    
    #Calculate Hamming Distance
    hamming_dist = 0
    for i in range(b):
        if x[i] != y[i]: hamming_dist += 1
    return(1 - hamming_dist/b) 

def calculateThreshold(similarity_list, multiple_of_mr):
    m = median(similarity_list) #Calculating Median
    n = len(similarity_list) 
    if n < 2: return(m) #To prevent divide by zero error
    
    #Calculating MR which is moving average of similarities
    mr_sum = 0
    for i in range(1,n):
        mr_i = abs(similarity_list[i] - similarity_list[i-1]) # mr(i) = | x(i+1) - x(i) |
        mr_sum += mr_i  
    mr = mr_sum/(n-1)

    #Return a dictionary with lower threshold and median value
    return( {"lower": m - multiple_of_mr * mr, "median": m, "mr" : mr} )

def plot_result(sim_list, threshold, filename, mr_multiplier):
    """A simple function to plot the the points and the threshold lines"""
    plt.plot(range( len(sim_list) ), sim_list, 'ro', color = '0.75' )
    plt.title('Threshold: (Median - ' + str(mr_multiplier) + ' * MR)' )
    plt.xlabel('Index')
    plt.ylabel('Similarity')
    plt.grid(True)   
    #Draw horizontal line for threshold.
    line1, = plt.plot([0, len(sim_list)], [threshold["lower"], threshold["lower"]], 'k--', lw=1, alpha=0.75)
    plt.legend([line1], ['Threshold'])
    plt.savefig(filename)

def find_anomalous_points(sim_list, threshold):
    """Finds the anamalous points based on threshold"""
    anomalous_point = []
    #Ignoring last point 
    for index, similarity in enumerate( sim_list[:-1] ): 
        next_similarity = sim_list[index+1]
        if ( (similarity < threshold["lower"] and next_similarity < threshold["lower"]) ):
            #Finding the disance of anamolous point from median in order to perform sorting
            distance_from_median = abs( similarity - threshold["median"] )
            anomalous_point.append( (index+1, distance_from_median) )
    return(anomalous_point)

def write_output(anomalous_points, out_put_file):
    """A simple function to write output in the desired format."""
    number_of_anomalies = len(anomalous_points)
    out_file = open(out_put_file , 'w+') 
    out_file.write(str(number_of_anomalies))

    #Sorting anomalous points based on distance from median
    anomalous_points.sort(key= lambda item: item[1], reverse=True)

    if (number_of_anomalies < 10):
        number_of_prints = number_of_anomalies
    elif (number_of_anomalies>=10 and number_of_anomalies<100):
        number_of_prints = 10
    else:
        number_of_prints = number_of_anomalies * 0.1

    for i in range(number_of_prints):
        out_file.write( '\n' + str( anomalous_points[i][0] ) )

    out_file.close()

if __name__ == '__main__':
    main()