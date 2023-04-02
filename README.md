# Question Generation-Question Answering Demo

This is a demo project that utilizes natural language processing models to generate questions based on given inputs and then answer those questions. This project uses argparse to parse command line arguments. The following are the descriptions of the arguments:

- --seed: This argument is used to set the random seed. The default value is 42.

- --qg_model_name: This argument is used to specify the name of the pre-trained model to be used for question generation. The default value is 'mrm8488/t5-base-finetuned-question-generation-ap'.

- --qg_model_type: This argument is used to specify the type of the pre-trained model to be used for question generation. The default value is 't5'.

- --qa_model_name: This argument is used to specify the name of the pre-trained model to be used for question answering. The default value is 'distilbert-base-cased-distilled-squad'.

- --qa_model_type: This argument is used to specify the type of the pre-trained model to be used for question answering. The default value is 'bert'.

- --qg_model_path: This argument is used to specify the path of the saved question generation model.

- --qa_model_path: This argument is used to specify the path of the saved question answering model.

- --input: This argument is used to specify the name of the input file containing the text for which questions are to be generated.

- --output: This argument is used to specify the name of the output file where the generated questions and their answers will be stored.

- --num_wrong_answers: This argument is used to specify the number of wrong answers to include with the correct answer. The default value is 2.

To run this demo project, you can modify the above arguments as per your requirement and then run the main.py file. The generated questions and their answers will be stored in the output file specified by the --output argument.