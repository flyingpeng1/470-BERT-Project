# WARPed QuizBERT - leveraging BERT and WARP loss to answer Quizbowl questions
By Jack, Charlie, Will, and Xavier


## Introduction
This project is based on the Codalab reference system - refer to the reference system documentation for more information on the structure of the project.
https://github.com/Pinafore/qanta-codalab


This is a system designed to answer Quizbowl questions using transformers BERT in a network of embedding layers to produce answers to natural questions.
Refer to the diagram for more details.

![Guesser diagram](BERT_diagram.png)


## Instructions
If you don't want to run the system and just want to see the resulting performance, you can view test_evaluation_output.json, dev_evaluation_output.json, and their associated logs test_eval.txt and dev_eval.txt

In order to use the system, you must make sure that you have a few things set up first. Docker will take care of most dependencies, but you must make sure that you have installed:

- Nvidia drivers  (VRAM >= 4GB)
- docker          (latest version)
- nvidia-docker   (latest version)
- docker-compose  (version > 1.28.0 - current default version does NOT support GPU and will crash)


To run in CPU only mode, rename 'docker-compose - CPU_ONLY - RENAME' to 'docker-compose' and continue with all of the following steps. You will only need:

- docker          (latest version)
- docker-compose  (latest version)


Next, you must download the model file and place it in the data folder:
https://drive.google.com/u/0/uc?export=download&confirm=D8Pf&id=1XDTvJyHEozSXlZAAnJHR1FXbFlacgWsj

Alternatively, run the command to download the model automatically:
`docker-compose run bert_qb ./cli download_model`


Finally, you may start the system using the command:

`docker-compose up bert_qb`

or launch both the answer server and the evaluator
`docker-compose up`


This will launch a web server accessible through the same means as the reference system.


## Running commands through docker-compose

If you want to run individual commands or train the system yourself, you should use
`docker-compose run bert_qb ./cli `
followed by the command that you desire to run:


run the model to answer questions and buzz
`web`
  `--vocab_file "path to vocab file to use"`
  `--model_file "path to model file to use"`
  `--buzzer_file "path to buzzer file to use"`
  `--link_file "path to link file csv to use"`
  `--host`
  `--port`

run the model with a built-in fast evaluator that provides diagnostics
`playquizbowl`
  `--vocab_file`
  `--model_file`
  `--buzzer_file`
  `--link_file`
  `--data_file`
  `--save_loc`

download the qanta data
`download` 

download the model
`download_model`

trains model and saves it in specified loaction
`train`  
  `--vocab_file "path to vocab file"`
  `--train_file "path to data file containing train set"`
  `--data_limit [number of questions to load (defaults to all)]`
  `--epochs [number of epochs to run]`
  `--resume *flag telling system to resume from previous model file`
  `--resume_file "path to model file to resume training from"`
  `--preloaded_manager *flag telling system to load data manager from file`
  `--manager_file "path to data manager file"`
  `--save_regularity [% of progress through an epoch to save (make over 100% for saving every epoch or more)]`
  `--category_only * flag telling system to train on category intead of answer page`
  `--eval_freq [how many epochs between every evaluation cycle]`
  `--unfreeze_layers "what BERT layers should be unfrozen when training in format like: 10+11+12 for freezing top 3 layers"`


`evaluate`
  `--vocab_file`
  `--model_file`
  `--split_sentences`
  `--dobuzztrain`
  `--buzztrainfile`
  `--preloaded_manager`
  `--manager_file`
  `--data_file`
  `--top_k`
  `--category_only`
  `--data_limit`


generate the answer vocab file
`generate_vocab`
  `--save_location "path to save file to"`
  `--data_file "path to data file to use"`
  `--category_only *flag telling system to make a vocab for question categorization istead of answer page`

generate pre-loaded data manager file
`makemanager`
  `--vocab_location`
  `--save_location`
  `--data_file`
  `--limit`
  `--category_only`
  `--cache_pool (unless you know what you are doing, don't use this!)`
  `--split_sentences`

launches interactive command line for exploring manager files like database tables
`managerdatabase`
  `--vocab_file`
  `--manager_file`

training the buzzer
`buzztrain`
  `--vocab_file`
  `--vocab_file`
  `--buzzer_file`
  `--data_file`
  `--data_limit`
  `--num_epochs`
  `--link_file`
  `--batch_size`

evaluating the buzzer
`buzzeval`
  `--vocab_file`
  `--buzzer_file`
  `--data_file`
  `--data_limit`
  `--num_epochs`
  `--link_file`
  `--batch_size`

check if CUDA is available
`cudatest`

## Run project modules in local environment (must manually set up environment - NOT reccomended)

Run project model only
`470-BERT-Project\src>python -m qanta.ProjectModel` 

Run project data loader only
`470-BERT-Project\src>python -m qanta.ProjectDataLoader`

Run project server only - use any of the commands that you would have used in docker.
`470-BERT-Project\src>python -m qanta.ProjectServer`




Guesser model download link:
https://drive.google.com/u/0/uc?export=download&confirm=D8Pf&id=1XDTvJyHEozSXlZAAnJHR1FXbFlacgWsj
