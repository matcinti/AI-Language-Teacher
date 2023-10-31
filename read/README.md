# AI Language Teacher

This application uses OpenAI to simulate a language learning experience between the user and an AI language teacher. Users can converse with the AI in the language they're learning, ask for clarifications on grammar/sentence structure, and request vocabulary translations.

## Overview

### Setting up:

The application uses the following libraries:
- **streamlit**: For creating the web application.
- **OpenAI**: To access the language model for language teaching.
- **langchain**: A library for interacting with OpenAI's models in a conversational format.
- **pandas**: For data manipulations.
- **re & ast**: For parsing and manipulating strings.
- **dotenv**: For loading environment variables.
- **os**: To interact with the operating system for file-related tasks.

### How it works:

#### Language Teaching Template:

A predefined template is used to instruct the AI on how to function as a language teacher. The template sets the tone of conversation, the languages being taught and clarified in, and optional topics to discuss.

#### Data Inputs:

Users can provide three types of inputs:

- **input_conversation**: User's response or continuation of a conversation.
- **input_question**: Any question or clarification the user has regarding the AI's previous message.
- **input_vocabulary**: Words from the AI's message that the user did not understand.

#### Data Outputs:

The AI provides four types of outputs in response:

- **output_correction**: Correction of the user's input sentence if it was incorrect.
- **output_conversation**: The AI's response or continuation of the conversation.
- **output_question**: The AI's answer to the user's question, in the user's native language.
- **output_vocabulary**: Translation of words the user didn't understand.

### Models & Templates:

Several models and templates are used to guide the AI's responses:

- **LLMChain**: A conversational model chain from the `langchain` library.
- **ChatOpenAI**: A chat model for general conversation.
- **OpenAI**: The standard OpenAI model for specific tasks like correction and vocabulary translation.

### CSV Vocabulary Update:

The application also has functionality to update a CSV file with vocabulary translations. This can serve as an ongoing glossary for the user.

## Using the App:

1. **Choose Languages**: Users first select the language they want to learn and the language in which they want clarifications.
2. **Additional Settings**: Users can also choose the tone (formal/informal) of the AI teacher and specify particular topics they'd like to discuss.
3. **Begin the Conversation**: Users can start or continue a conversation with the AI teacher, ask questions, and list out words they didn't understand for translations.
4. **Output**: The AI responds in the specified format, correcting sentences if needed, answering questions, continuing the conversation, and providing vocabulary translations.

## Conclusion:

The AI Language Teacher is an innovative way to practice a new language, get real-time feedback, and enhance understanding with the help of artificial intelligence.
