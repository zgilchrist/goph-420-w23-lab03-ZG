from regression import multi_regress

import matplotlib.pyplot as plt
import numpy as np

def main():
    """Main function"""
    
    #tests the regression tool based on an in-class example
    testing_data()

    #Parses data from raw data
    data = np.loadtxt("M_data.txt")
    hours = data[:,0]
    magnitudes = data[:,1]
    
    #Determined visually from graphical plot that there are four distinct events (measured in hours)
    events = [34,46,71,96]
    
    plt.figure()
    plt.xlabel("time, t[hrs]")
    plt.ylabel("event magnitude, M")
    plt.title("Raw Earthquake Event Data: Red Lines Indicate an Event")
    plt.plot(hours,magnitudes,"b.")
    plt.vlines(events,ymin=-2,ymax=2,colors = 'red')

    #Create intervals of magnitudes to create N counts with
    M_thresh = np.array([-0.5,-0.25,0,0.25,0.5,0.75,1.0])
    #M_thresh = np.array([-1.25,-1.0,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1.0, 1.25, 1.5, 1.75,2.])
    
    
    
    #Parses the events to find index where that event stops, then creates array of indices to be used later
    indices = np.zeros(0)
    j=0
    for event in events:
        while event > hours[j]:
            j+=1
            #if j == 13390:
                 #break
        indices = np.append(indices,j)
   
   #Plot setup
    plt.figure()
    plt.xlabel("Magnitude")
    plt.ylabel("log(N) number of events")
    plt.title("log(N) frequency of events at M magnitude")
    
    
    #Each curve plots the magnitude interval against the log_10 of the frequency of events at each interval by calling get_counts
    n_1 = np.log10(get_counts(M_thresh,magnitudes[:int(indices[0])]))
    plt.plot(M_thresh,n_1,"r.-", label = "Event 1")
    
    n_2 = np.log10(get_counts(M_thresh,magnitudes[int(indices[0]):int(indices[1])]))
    plt.plot(M_thresh,n_2,"b.-", label = "Event 2")

    n_3 = np.log10(get_counts(M_thresh,magnitudes[int(indices[1]):int(indices[2])]))
    plt.plot(M_thresh,n_3,"g.-", label = "Event 3")

    n_4 = np.log10(get_counts(M_thresh,magnitudes[int(indices[2]):int(indices[3])]))
    plt.plot(M_thresh,n_4,"y.-", label = "Event 4")

    plt.legend()
    plt.grid()

    #subplots of each curve against its linear regression

    plt.figure()
    plt.subplot(411)
    plt.plot(M_thresh,n_1,"ro", label = "Event 1")
    a,e,rsq,model = get_model(n_1,M_thresh)
    plt.plot(M_thresh,model,"k--")
    plt.title(f"Event 1: params: {a}, R^2: {rsq}", loc='left')
    plt.xlabel("Magnitude")
    plt.ylabel("log(N) number of events")

    plt.subplot(412)
    plt.plot(M_thresh,n_2,"bo", label = "Event 2")
    a,e,rsq,model = get_model(n_2,M_thresh)
    plt.plot(M_thresh,model,"k--")
    plt.title(f"Event 2: params: {a}, R^2: {rsq}", loc='left')
    plt.xlabel("Magnitude")
    plt.ylabel("log(N) number of events")
    
    plt.subplot(413)
    plt.plot(M_thresh,n_3,"go", label = "Event 3")
    a,e,rsq,model = get_model(n_3,M_thresh)
    plt.plot(M_thresh,model,"k--")
    plt.title(f"Event 3: params: {a}, R^2: {rsq}", loc='left')
    plt.xlabel("Magnitude")
    plt.ylabel("log(N) number of events")
    
    plt.subplot(414)
    plt.plot(M_thresh,n_4,"yo", label = "Event 4")
    a,e,rsq,model = get_model(n_4,M_thresh)
    plt.plot(M_thresh,model,"k--")
    plt.title(f"Event 4: params: {a}, R^2: {rsq}", loc='left')
    plt.xlabel("Magnitude")
    plt.ylabel("log(N) number of events")

    plt.show()

def get_model(N,M):
     
    y = np.transpose(N)
    Z = np.transpose(np.array([np.ones(M.shape),M])) 
    a,e,rsq,model = multi_regress(y,Z)

    return a,e,rsq,model

def testing_data():
    #sample data from in-class problem for ease of verification
    #the output is correct
    y = np.transpose(np.array([22.8,22.8,22.8,20.6,13.9,11.7,11.1,11.1]))
    Z = np.transpose(np.array([[1,1,1,1,1,1,1,1],[0,2.3,4.9,9.1,13.7,18.3,22.9,27.2]]))
    
    a,e,rsq,model = multi_regress(y,Z)
    
    print(f"##########\nFor the test data from the example in class\nThe vector of model coefficients is {a}\nThe R^2 value is: {rsq}\n##########")
    
    return None

def get_counts(threshold,magnitude_slice):
    
    M_counts = np.zeros(0)
    
    for index,M_thresh in enumerate(threshold):
        M_counts = np.append(M_counts,np.sum(np.where(magnitude_slice>=threshold[index],1,0)))

    return M_counts


if __name__ == "__main__":
        main()