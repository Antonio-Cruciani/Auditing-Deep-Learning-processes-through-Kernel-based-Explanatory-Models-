README.

This report illustrates the main functionalities implemented in the code.
For more details and other classes/modules, comments are found in codelines or you can write to:

		antonio.cruciani at alumni.uniroma2.eu
	
You should read this papers to fully understand the system:

Introduction to KDA:

	https://www.aclweb.org/anthology/P17-1032

Introduction to Automatic Generation of Explanations:
	
	https://www.aclweb.org/anthology/W18-5403

In this readme file we illustrate the architecture and the functionalities, the application file is in the main folder and the other methods\classes\functions are situated in the following path:

		/src/main/python_code/< dedicated\_folder>
		
Files are situated in the following path:

		\src\main\resources

For running the application go to the main folder and digit
	
		./flask run
NOTE: to run the application you need flask.

You can find a folder with pretrained models and dataset.

If you want to use the pretrained model you have to copy the content of the folder "PretrainedModels" in 
		
		/models/
		
and copy the content of Datasets in
		
		src/main/resources/

# TABLE OF CONTENTS:

	0 - General Infos
	1 - Flask app
		1.1- do_admin_login 
		1.2 - logout
		1.3 - home
		1.4 - init
		1.5 - result
	2 - Neural Net
		2.0 - Result Class 
		2.1 - get_ny_proj
		2.2 - get_arrays_from
		2.3 - classify
		2.4 - create_model
		2.5 - train_new_model
		2.6 - get_flags
	3 - Explanatory Model
		3.0 - Explanation abstract Class
			3.0.1- Methods
		3.1 - build_explanations Class
			3.1.1 - Methods
	4 - Other Classes and Functions
		4.1 - labels class
		4.2 - landmarks class
		4.3 - read_files functions
			4.3.1 - get_landmarks_from_file
			4.3.2 - get_labels_from_file
	5 - How to launch the system
		5.1 - Start Nystrom Module
		5.2 - Start Flask

# 0 - General infos
	
The folder 'KDA_readability_projec' contains all codes and data to perform classification (via Kernel-based Deep Architecture, KDA) and prediction explanation (applying Layerwise Relevance Propagation, LRP) on one task: Question Classificiation.

For this task the (high level explanation) workflow is the following:

	1 - The user write a question on the web page and 
		send it to the flask orchestrator (post or get).
	2 - The orchestrator, given the question, send a 
		request to a nystrom web server
	3 - The nystrom web server, given the question, 
		project it to a nystrom space and return to the 
		orchestrator an embedded vector of the question 
		in that low rank space.
	4 - The orchestrator, given the embedded vector, 
		feed the Neural Network with it.
	5 -  The NN classify the input instance, returning 
		the: Predicted Label (eg: HUM), the ordered list 
		of landmarks activated by the input instance.
	6 - The orchestrator, given the returned items at 
		point 5, use the explanatory model to build the 
		explanations.
	7 - Given the explanations, the orchestrator build a 
		json and return it to the user (by an HTML page)
	
# 1 - Flask app

In this section we will describe the flask application.
## 1.1 - do\_admin\_login

This function show to the user a login html form, and after the send operation perfomed by the user, it checks if he inserted the correct combination of username and password, if so, start a new session for the correcly logged user.

## 1.2 - logut

This function terminate the current session.

## 1.3 - home

This function first checks if the session is correcly setted. If not, call the login function. Else, call the init function that load the neural network and set the mlp and session global variables of the NN and show to the user the index.html page.

## 1.4 - init

This function call the create_model().py function that will be illustrated in Section 2. 

The create model load the neural network and returns the mlp and the session variables

## 1.5 - result

This function is called when the user write a question in the input form in the index.html web page and press "Classify" button.

This function first checks if the session is correcly setted. If not, call the login function. 
Else:

	1) Build a json with the question string 
	2) Do a request (by post) to the nystrom web server 
		and obtain the embedded vector
	3) Transoform the c-vector in a valid input instance  
		for the neural network
	4) Call the get_ny_proj.py function with input 
		parameters:
			- mlp
			- session
			- c-vector instance for the NN
			- question string
	5) Given the classification object returned by get_ny_proj.py build a new explanation object explenation_qc and builds the explanations setting:
		- build_explanation_positive_singleton(prediction)
		- build_explanation_negative_singleton(prediction)
		- build_explanation_positive_conjunctive(prediction,Hyper Parameter K)		
		- build_explanation_negative_conjunctive(prediction,Hyper Parameter K)
		- build_explanation_positive_contrastive(prediction)
		- build_explanation_negative_contrastive(prediction)

After this, given the explanation object, populate a json with the following parameters:

	- input Question
	- PositiveSingleton expl.
	- NegativeSingleton expl.
	- PositiveContrastive expl.
	- NegativeContrastive expl.
	- PositiveConjunctive expl.
	- NegativeConjunctive expl.

And finally, return it to the user using render_template to send the json to the index.html web page.

# - Neural Net

In this section we will illustrate the neural networks functions and objects.

## 2.0 - Result object
This object contains :

	- information , that is the input question
	- predicted_class
	- k_positive_landmarks , that is the list of the 
	  first k-positively activated landmarks by 
	  the prediction. 
	- k_complement_landmaks , that is a dictionary 
	 <key,value> where we have
	  <j-th class, negative-consistent landmarks actived 
	  by the prediciton>
	- k_parameter , that is the K hyper parameter

Clearly the object has set and get methods for each attribute.

## 2.1 - get\_ny\_proj

This function call the prediction function for the classification tast giving to it the mlp,session,c_vector and the input question.

## 2.2 - get\_arrays\_from

This function transform the c-vector in a instance object that is the input "format" for the neural network.

## 2.3 - classify

Function that call the Neural Network for the classification task.
Given the classification we obtains the prediction and the LRP scores.
As last task, after the prediction, the function takes the first k-consistent- positively activated landmarks and the k-cons-neg-activated and populate the Result object with the prediction, the parameter the input question, the positive_consistent list of landmarks and the negative_consistent dictionary. 

## 2.4 create\_model

Function that loads a pretrained model, as a first step load all the needed flags by the neural model. Than define the empty graph (NN) and finally loads the "saved_model" instace of the neural network.

The function returns the restored mlp,session.

## 2.5 train\_new\_model

This function call the training function, called: train_LRP_mlp_qc passing to in the landmarks directory and the size of the landmarks.

## 2.6 get\_flags

This function first clear all flags (deleting them) and after this 'cleaning step' sets all the needed flags for the neural architecture and then return them.

 
## 3 - Explanatory Model
In this section we will explain the explanatory model and the classes that composes it.

## 3.0 Explanation abstract class

This is an abstract class composed by the following methods:

	- build_explanation_positive_singleton
	- build_explanation_negative_singleton
	- build_explanation_positive_conjunctive
	- build_explanation_negative_conjunctive
	- build_explanation_positive_contrastive
	- build_explanation_negative_contrastive

All this methods must be implemented when someone decides to extend this class (NotImplementedError).

## 3.1 build_explanations Class

This class extends the previous one and implements all the methods described before.

	
	- build_explanation_positive_singleton(prediction)
	- build_explanation_negative_singleton(prediction)
	- build_explanation_positive_conjunctive(prediction,positive_number)
	- build_explanation_negative_conjunctive(prediction, negative_number)
	- build_explanation_positive_contrastive(prediction)
	- build_explanation_negative_contrastive(prediction)

Let's explain these methods.

### build\_explanation\_positive_singleton

Given the prediction object we construct the explanation for the positive case. Specifically we construct an explanation using only one landmark, the most relevant consistent positively activate landmark, if it exists.

So, given the prediction object we do a get_k_positive_landmarks for getting the list of all consistent positively activated landmarks (ordered in descending order of relevance) ad if this list is not empty we construct an explanation of the form:

" I think < input question> refers to a < predicted\_class>  since it recalls me of < positive\_landmark> that also refers to a < class\_of\_landrmak>" 

NOTE: < class\_of\_landrmak> = < predicted\_class>

If le list of positive landmarks is empty, so we don't have positive examples to explain the classification an so we construct the following explanation:

" I think < input question> refers to a < predicted\_class>  but I don't know why"
 
### build\_explanation\_negative_singleton

Is the same process of the previous one explanation, but it differs in one thing: we have to explain why the prediction is not of oneother class.

Given the prediction object we get the complement classes dictionary using get\_k\_complement\_landmarks method.
After this, if exists, we sample U.A.R. one element from the dictionary and then we construct a similar explanation of the positive singleton:

" I think < input question> does not refer to a < sampled\_class>  since it does not recall me of < sampled\_landmark> that also refers to a < class\_of\_sampled\_landrmak>" 

Clearly, if we don't have negative explamples the explanation will be:

"I think < input question> does not refer to a < sampled\_class>  but I don't know why"

###build\_explanation\_positive\_conjunctive

This method gets in input one more parameter, that is the number of the positive landmark to use to build the explanation. 
So we get the list of the landmarks as before and then check if we got the required number of landmarks, if so we use, let's say k, the first k-positive landmarks to buil a new explanation that is composed by a conjunction of multiple positive explanations.

" I think [..] because it recalls me of [..] that is [..] and also because it recalls me [..] "

Clearly if we have 0<l<k landmarks we use l landmarks to buil the conjunction.

And if we have 0 landmarks we construct the classical explanation " I think [..] but I don't know why".

### build_explanation_negative_conjunctive

As the negative singleton we construct an explanation of why the input question is not of oneother class.
In this case we perform k-samplings without replacement from the dictionary and use these k-samples to construct the negative explanation.


### build_explanation_positive_contrastive

The contrastive explanation method is a little bit different from the previous.
In this scenario we get the positively and the negatively activated landmarks to construct an explanation of the form:

"I think < original\_question> refers to a < predicted\_class> since it recalls me of < positive\_landmark> that also refers to a < positive\_landmark\_class>                                        and because it does not recall me of < negative\_landmark> which refers to a < negative\_landmark\_class> instead.

### build_explanation_negative_contrastive

The negative contrastive follow the same structure of the previous one, but in this case we're constructing an explanation to explain the following type of sentece:
" The input question is not of the class [..] but is of the class [..] "


# 4 - Other classes and functions

## Label class

This class is composed by the following attributes:

	- label map
	- label inverted map

and the following methods:

	- get_inverted_label_map
	- get_label_map
	- get_labels

## Landmark class

This class is composed by the following attributes:
	
	- landmarks, that is a list
and the following methods:

	- get_landmarks  
	- get_size_of_landmarks

## Read\_files functions

In this subsection we illustrate the functions used to read hypermarameters and landmarks from files.

### get\_landmarks\_from\_file
This function take in input the landmarks file path, and then load the file in a list of landmarks returning it.

### get\_labels\_from\_file
This function take in input the labels path file, and then construct two dictionary: the map and the inverted map, returning them.

# 5 - How to launch the system
In this section we explain how to launch the system.	
##	5.1 - Start Nystrom Module
As a first step you need to stat the Nystrom Module. So you need to create a war file from the NystromProjector and use apache tomcat or a server to host it.

Given the url of the Nystrom Web Server, you have to change the url inside the "app.py" application at line 13.

##	5.2 - Start Flask
After 5.1 you have to start flask.

And that's it.


