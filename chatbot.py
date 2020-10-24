from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer

# Creating ChatBot Instance
chatbot = ChatBot(
    'DoctorBot',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    logic_adapters=[
        'chatterbot.logic.MathematicalEvaluation',
        'chatterbot.logic.TimeLogicAdapter',
        'chatterbot.logic.BestMatch',
        {
            'import_path': 'chatterbot.logic.BestMatch',
            'default_response': 'I am sorry, but I do not understand. I am still learning.',
            'maximum_similarity_threshold': 0.80
        }
    ],
    database_uri='sqlite:///database.sqlite3'
)

 # Training with Personal Ques & Ans 

training_data_personal = open('training_data/personal_ques.yml').read().splitlines()
training_data_bot1 = open('training_data/bot1_ques.yml').read().splitlines()
training_data_coughcold = open('training_data/coughcold_ques.yml').read().splitlines()
training_data_fever = open('training_data/fever_ques.yml').read().splitlines()
training_data_fracture = open('training_data/fracture_ques.yml').read().splitlines()
training_data_generalhealth = open('training_data/generalhealth_ques.yml').read().splitlines()
training_data_greetings = open('training_data/greetings_ques.yml').read().splitlines()
training_data_headache = open('training_data/headache_ques.yml').read().splitlines()
training_data_personal1=open('training_data/dialogs.yml').read().splitlines()

training_data =  training_data_personal + training_data_bot1 + training_data_coughcold + training_data_fever + training_data_fracture + training_data_generalhealth +  training_data_greetings + training_data_headache + training_data_personal1


trainer = ListTrainer(chatbot)
trainer.train(training_data)
# Training with English Corpus Data 
trainer_corpus = ChatterBotCorpusTrainer(chatbot)
trainer_corpus.train(
    'chatterbot.corpus.english'
) 