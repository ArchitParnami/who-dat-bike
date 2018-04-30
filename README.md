# who-dat-bike
Is it a mountain bike or a road bike? Find out using this bike classifier implemented using a CNN in keras.

### Folder Structure For Training and Testing
	/bikes  
		/training_set  
			/mountain_bikes  
			/road_bikes    
		/test_set  
			/mountain_bikes  
			/road_bikes  

### Training your model
#### train.py  
	1. Executing this file will start the traning process on /bikes/training_set  
	2. After training is finished the model will be saved as a file 'bike_classsifier.h5'  
	3. A Graph folder should also be created, which is used by Tensorboard for visualization.  
		To View the graph, execute the following command from the terminal (from the current directory)  
		 tensorboard --logdir ./Graph 

### Testing
#### test.py  
	1. Executing this will load the model 'bike_classifier.h5' generated from training.  
	2. It will start testing each file in bike/test_set one by one and will output the results.  


	
