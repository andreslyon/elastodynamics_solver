import pickle
count = 0
fileObject = open("experiment_counter",'wb') 
pickle.dump(count, fileObject)   
fileObject.close()

