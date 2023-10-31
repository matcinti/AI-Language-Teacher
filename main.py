import streamlit as st
from langchain import PromptTemplate
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory

#libraries to parse strings into lists
import pandas as pd
import re
import ast
import os


load_dotenv()


## Template for everything but works only with gpt4 :(

template = """
You are a {language_learn} language teacher and you are chatting with an {language_clarification} user. You will be having conversations with the {language_clarification} user so that he learns how to engage in a conversation in {language_learn}. The tone should be {option_tone}. Some topics you could talk about could be: {option_topics}, however the conversation can also deviate from these topics.


The user will be providing three different inputs in the same prompt: input_conversation, input_question, input_vocabulary. Below a description of the inputs:

input_conversation: The normal input of the conversation that usually is a response of the user to what the AI (teacher) has previously messaged. This will of course be in German.

input_question: the user may have some questions regarding the previous message sent from the AI. e.g. "Why did you use the verb 'watch' instead of the verb 'see'?", "Why did you put the verb in the final position?" This will be of course asked from the user in their native language, so in italian.

input_vocabulary: The user may also have questions regarding the vocabulary used from the AI (teacher) in the previous message. He may not know some words. The user will therefore input the words that he did not understand.


At this point you will need to respond to the user by providing the following four outputs: output_correction, output_conversation, output_question, output_vocabulary. Below a description of the outputs:

output_correction: If the input_conversation is correct return None. Otherwise if the user made a mistake, return the corrected sentence.

output_conversation: The normal output of the conversation. You will need to answer to input_conversation and continue the conversation with the user. Of course this is done in german.

output_question: Here you will answer to the input_question of the user and you will clarify using the user's native language: italian.

output_vocabulary: You will provide the translation of the words that the user requested in input_vocabulary. You will provide this as bullet points.


Every time you answer to the user input you will need to clearly provide the output in the following structure:

output_correction:...

output_conversation:...

output_question:...

output_vocabulary:...


Note that it may happen that the user does not make a mistake and does not provide certain inputs (e.g. output question). Your answer should still have the following structure but with None where no output is needed:

output_correction: None

output_conversation: ...

output_question: None

output_vocabulary: ...

Attention! You must answer only with output_correction, output_conversation, output_question, and output_vocabulary!! No other information neither at beginning nor at the end.
"""

## Conversation TEMPLATE PROMPT AND MODEL

template = """
You are a {language_learn} language teacher and you are chatting with an {language_clarification} user. You will be having conversations with the {language_clarification} user so that he learns how to engage in a conversation in {language_learn}. The tone should be {option_tone}. Some topics you could talk about could be: {option_topics}, however the conversation can also deviate from these topics.
"""

prompt=PromptTemplate(
    template=template,
    input_variables=["language_learn", "language_clarification", "option_tone", "option_topics"]
)

system_message_template = SystemMessagePromptTemplate(prompt=prompt)

def load_chat_model(system_prompt):
    # LLM
    llm = ChatOpenAI(temperature=0.7)

    # Prompt 
    prompt = ChatPromptTemplate(
        messages=[
            system_prompt,
            # The `variable_name` here is what must align with memory
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )

    # Notice that we `return_messages=True` to fit into the MessagesPlaceholder
    # Notice that `"chat_history"` aligns with the MessagesPlaceholder name
    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    conversation = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory
    )
    return conversation


## Correction TEMPLATE and PROMPT

template_correction = """
sentence: {user_input_conversation}
Correct the previous {language_learn} sentence and provide me with an explanation in {language_clarification} to why I made a mistake. 
If the sentence is correct then answer ONLY with: "None"
"""

template_correction = """
sentence: {user_input_conversation}
Correct the {language_learn} sentence if the sentence is wrong.
"""

prompt_correction = PromptTemplate(
    input_variables=["user_input_conversation","language_learn"],
    template=template_correction
)

## Clarification TEMPLATE and PROMPT

template_clarification = """
AI teacher's sentence: "{ai_answer}"

I have some questions regarding the AI teacher's {language_learn} sentence:
{ask_clarification}

Please answer to my questions in {language_clarification}. It is very important that you answer in {language_clarification}.
"""

prompt_clarification = PromptTemplate(
    input_variables=["ai_answer","ask_clarification","language_learn", "language_clarification"],
    template=template_clarification
)

## Vocab TEMPLATE and PROMPT

##removed from prompt:# If the word is a noun add to the variable word_to_translate also the noun's article and its plural. So for example if one of the words to translate is 'Worte' I want word_to_translate to be: Die Wort(-en)
template_vocab = """

words to translate: {words_to_translate}

Translate the words from {language_learn} to {language_clarification}. 

I want the output as an array of three elements for each word: 

[[word1_to_translate, translated_word1, example_of_sentence1, translation_of_example_of_sentence1], [word2_to_translate, translated_word2, example_of_sentence2, translation_of_example_of_sentence2], ...]

In the example_of_sentence you must invent a sentence that contains the word_to_translate.

THE OUPUT MUST STRICTLY BE AN ARRAY, NOTHING MORE! 
ALL of the variables in the array must be contained in double  quotation mark! So below an example of how it should look like:

{example}
As you can see, quotation marks are around ALL elements of array.
In the example words are translated from {language_learn} to italian. Of course you will translate instead from {language_learn} to {language_clarification}
"""

vocab_examples = {
    'German': '[["Die Urlaube", "Vacanza", "Wir planen unsere Urlaube im Sommer", "Pianifichiamo le nostre vacanze in estate"], ["sehen", "vedere", "Er kann sie sehen", "Lui può vederla"], ["gehen", "andare", "Ich will heute gehen", "Voglio andare oggi"], ["Das Wetter", "il tempo", "Das Wetter heute ist warm", "Il tempo oggi è caldo"]]',
    'English': '[["Holiday", "Vacanza", "We plan our holidays in summer", "Pianifichiamo le nostre vacanze in estate"], ["see", "vedere", "He can see her", "Lui può vederla"], ["go", "andare", "I want to go today", "Voglio andare oggi"], ["Weather", "il tempo", "The weather today is warm", "Il tempo oggi è caldo"]]',
    'French': '[["Vacances", "Vacanza", "Nous planifions nos vacances en été", "Pianifichiamo le nostre vacanze in estate"], ["voir", "vedere", "Il peut la voir", "Lui può vederla"], ["aller", "andare", "Je veux aller aujourd\'hui", "Voglio andare oggi"], ["Le temps", "il tempo", "Le temps aujourd\'hui est chaud", "Il tempo oggi è caldo"]]',
    'Spanish': '[["Vacaciones", "Vacanza", "Planeamos nuestras vacaciones en verano", "Pianifichiamo le nostre vacanze in estate"], ["ver", "vedere", "Él puede verla", "Lui può vederla"], ["ir", "andare", "Quiero ir hoy", "Voglio andare oggi"], ["El clima", "il tempo", "El clima de hoy es cálido", "Il tempo oggi è caldo"]]'
}


prompt_vocab = PromptTemplate(
    input_variables=["words_to_translate","language_learn","language_clarification","example"],
    template=template_vocab
)


## Model we are going to use later for Corrections, Vocab
def load_LLM():
    """Logic for loading the chain you want to use should go here."""
    llm = OpenAI(temperature=.7)
    return llm


def string_to_dataframe(variable_string, language_learn, language_clarification):        
    # Convert the string to a Python list
    variable_array = ast.literal_eval(variable_string)
    # Check if it's a single list (by counting the number of opening brackets)
    if variable_string.count('[') == 1:
        variable_array = [variable_array]  # Make it a list of lists
    # Convert list of lists to a dataframe
    df = pd.DataFrame(variable_array, columns=[language_learn, language_clarification, 'Sentence', 'Translated Sentence'])
    return df

def update_csv(variable_string, language_learn, language_clarification,file_name='vocabulary.csv'):
    # Convert string to dataframe
    df_new = string_to_dataframe(variable_string, language_learn, language_clarification)
    
    # If CSV doesn't exist, create one
    if not os.path.exists(file_name):
        df_new.to_csv(file_name, index=False)
    else:
        # If CSV exists, read the existing CSV
        df_existing = pd.read_csv(file_name)
        
        # Filter out rows from the new dataframe that already exist in the CSV
        unique_rows = df_new[~df_new[language_learn].isin(df_existing[language_learn])]
        
        # Append the unique rows to the existing CSV
        unique_rows.to_csv(file_name, mode='a', header=False, index=False)



#------------------------BEGIN STREAMLIT--------------------------------------------------------------------------------

st.set_page_config(page_title="AI teacher", page_icon="Robot")
st.markdown("<h1 style='text-align: center;'>AI Language Teacher</h1>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

#with used for: Setting Up and Tearing Down: Establishing a certain state before executing a block of code and restoring the original state afterward, which is what is happening in your Streamlit example.

# with col1:
#     st.markdown("Often professionals would like to improve their emails, but don't have the skills to do so. \n\n This tool \
#                 will help you improve your email skills by converting your emails into a more professional format. This tool \
#                 is powered by [LangChain](https://langchain.com/) and [OpenAI](https://openai.com) and made by \
#                 [@GregKamradt](https://twitter.com/GregKamradt). \n\n View Source Code on [Github](https://github.com/gkamradt/globalize-text-streamlit/blob/main/main.py)")

# with col2:
#     st.image(image='TweetScreenshot.png', width=500, caption='https://twitter.com/DannyRichman/status/1598254671591723008')


st.markdown("## Settings")

col1, col2 = st.columns(2)

with col1:
    language_learn = st.selectbox(
        'Which language would you like to learn?',
        ('Italian', 'English', 'French', 'Spanish', 'German'))

with col2:
    language_clarification = st.selectbox(
        'Which language would like to choose for clarifications',
        ('Italian', 'English', 'French', 'Spanish', 'German'))
    
st.markdown("### Additional (Optional) Settings")

col1, col2 = st.columns(2)

with col1:
    option_tone = st.selectbox(
        'Which tone would you like your AI teacher to have?',
        ('Formal', 'Informal'))
    
with col2:
    option_topics = st.text_area(
        label="Which topics would you like to discuss?", 
        placeholder="e.g. work, holidays, favourite movies", 
        key="topics_input")

st.markdown("## Begin the Conversation")


# This is your text area for "Answer to Teacher"
user_input_conversation = st.text_area(label="Chat to Teacher", placeholder="your answer", key="user_input_conversation")

col1, col2 = st.columns(2)

# This is your text area for "What did you not understand from the teacher's answer"
with col1:
    ask_clarification = st.text_area(
        label="What did you not understand from the teacher's answer", 
        placeholder="e.g.\nWhy did you put the verb in the final position?\nWhat is the subject of the sentence?\nWhy did you use the verb 'watch' instead of the verb 'see'?", 
        key="clarification")

# This is your text area for "What words did you not understand: list them"
with col2:
    ask_vocabulary = st.text_area(
        label="What words did you not understand: list them", 
        placeholder="e.g. seldom, blew, akin", 
        key="vocabulary")


if st.button('Submit'):

    if user_input_conversation:

        ## Conversation
        system_message_final = system_message_template.format(language_learn=language_learn, language_clarification=language_clarification, option_tone=option_tone, option_topics=option_topics)

        # Check if 'conversation_chain' exists in session state
        if 'conversation_chain' not in st.session_state:
            # If not, create it
            st.session_state.conversation_chain = load_chat_model(system_message_final)

        # Use the existing 'conversation_chain' from session state
        answer = st.session_state.conversation_chain({"question": user_input_conversation})['text']

        with st.container():
            st.markdown("### Conversation")
            st.text_area("", value=answer, height=100, disabled=True)
        

        ## Corrections
        with st.container():
            st.markdown("### Corrections")
            prompt_with_correction = prompt_correction.format(language_learn=language_learn, user_input_conversation=user_input_conversation)
            correction = load_LLM()(prompt_with_correction)
            st.text_area("", value=correction, height=100, disabled=True)

        ## Clarification
    if ask_clarification:
        with st.container():
            st.markdown("### Clarifications")
            prompt_with_clarification = prompt_clarification.format(language_learn=language_learn, language_clarification=language_clarification, ask_clarification=ask_clarification, ai_answer=answer)
            llm = OpenAI(temperature=.7)
            clar = llm(prompt_with_clarification)
            st.text_area("", value=clar, height=100, disabled=True)

        ## Vocab
    if ask_vocabulary:
        with st.container():
            st.markdown("### Vocabulary")
            prompt_with_vocab = prompt_vocab.format(words_to_translate=ask_vocabulary, language_learn=language_learn, language_clarification=language_clarification, example=vocab_examples[language_learn])
            llm = OpenAI(temperature=0)
            vocab = llm(prompt_with_vocab)
            st.text_area("", value=vocab, height=100, disabled=True)
            update_csv(variable_string=vocab, language_learn=language_learn, language_clarification=language_clarification)
        
