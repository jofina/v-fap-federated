# v-fap-federated

![image](https://user-images.githubusercontent.com/47222463/136626991-4aa0315f-6be3-42dc-be2d-c3b552a0f62e.png) 
![image](https://user-images.githubusercontent.com/47222463/136627277-95a45557-2a17-46fb-91a5-0c4e71ebef5b.png)






data_prepro_split.py
This code runs in the seed node, in our scenario this node is a laptop
This script runs to obtain initial parameters of the dnn model, and also split the training data to be send to the raspberry pis. The training data and nitial parametes are converted into a json format.

singlet.py
This script coordinates the sending and receiving between seed node and service nodes using parallel SSH.
The parameter and the data obtained by running "data_prepro_split.py" is then send to 4 rapaberry pi's.

dnn_global.py
This python script gets executed on each of the raspberry pi's using the received raining data  and the inital parameters.

The trained final parameters are receved by the seed node where the aggreation technique is used to obtain the final aggregated weights.  this aggregation part has been done in a jupyter notebook.
