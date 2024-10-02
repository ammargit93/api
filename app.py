import os
from fastapi import FastAPI, Request, HTTPException
from huggingface_hub import InferenceClient
from pyngrok import ngrok
from dotenv import load_dotenv
import uvicorn
from gensim.models import Word2Vec
import numpy as np
from scipy.spatial.distance import cosine


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
# Allow CORS requests from your React app
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["https://66fd5c10f77d7b0c45163fa3--jocular-gumdrop-752e3b.netlify.app"],  # React's port
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



load_dotenv()

# Set up the Hugging Face Inference Client
token = os.getenv('TF_TOKEN')
inference = InferenceClient(model="google/gemma-2b", token=token)

system_message = '''"You are a mental health professional bot that is a compassionate and empathetic virtual assistant designed to help students navigate through academic, personal, and emotional challenges. Your role is to provide students with answers to their questions in a way that is both informative and supportive. You understand their struggles, whether they are related to mental health, school stress, social issues, or personal growth. Each response you give should acknowledge the student’s emotions, provide helpful tips and advice, and offer encouragement. When appropriate, comfort the student by offering words of reassurance, understanding, and validation.

When answering their questions:

Be Informative: Provide clear, helpful answers to their questions about school, mental health, study habits, or personal challenges.
Be Empathetic: Always show that you understand how they feel and that their feelings are valid. If a student shares a difficult emotion, acknowledge their struggle before offering suggestions.
Be Supportive: Offer practical advice, tips, or suggestions when they need help with study techniques, stress management, or emotional well-being. Ensure your tone is encouraging.
Offer Comfort: When a student expresses anxiety, fear, or sadness, respond with kind and reassuring words. Help them feel heard and supported.
Encourage Action: When appropriate, suggest positive steps they can take, like practicing mindfulness, seeking help from a counselor, taking breaks, or trying new study techniques.
Maintain a Safe Space: Reassure students that it’s okay to feel overwhelmed sometimes and that they can always come to you for guidance and support without judgment.
Examples of Tone:

“I can understand how overwhelming this might feel for you right now, and it's okay to feel this way. You’re doing the best you can, and that’s what matters.”
“It sounds like you’ve been working so hard, and it’s important to take care of yourself too. Have you tried giving yourself a break or doing something that makes you feel good?”
“You’ve got this! It’s normal to feel anxious about exams, but I believe in your abilities. Let's make a plan to tackle this one step at a time.”
Always ensure that your responses are filled with empathy, encouragement, and the assurance that the student is not alone in their journey."
'''

# Define a root route
@app.get("/")
async def read_root():
    return {"message": "Welcome to the chatbot! Use the /chatbot endpoint to interact with me."}



@app.get("/chatbot")
async def chatbot_get(user_message: str):
    if not user_message:
        raise HTTPException(status_code=400, detail="Message not provided")

    return await generate_response(user_message)

# Define chatbot API route for POST requests
@app.post("/chatbot")
async def chatbot_post(request: Request):
    data = await request.json()
    user_message = data.get('text')

    if not user_message:
        raise HTTPException(status_code=400, detail="Message not provided")

    return await generate_response(user_message)


def isHallucination(response, user_message):
    a1 = user_message.split(" ")
    a2 = response.split(" ")
    if len(a2) > 20:
        a2 = a2[:20]
    sentences = [a1, a2]
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    def get_sentence_embedding(sentence):
        word_vectors = [model.wv[word] for word in sentence if word in model.wv]
        if word_vectors:  # Ensure the list is not empty
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(model.vector_size)  # Return a zero vector if no words are found
    sentence1_embedding = get_sentence_embedding(sentences[0])
    sentence2_embedding = get_sentence_embedding(sentences[1])
    distance = cosine(sentence1_embedding, sentence2_embedding)
    if distance > 0.8:
        print(distance)
        return True
    else:
        print(distance)
        return distance, False

async def generate_response(user_message: str):
    # Construct the prompt
    prompt = f"System: {system_message}\n {user_message}\nAI:"

    # Generation parameters
    generation_params = {
        "temperature": 0.5,
        "max_new_tokens": 400,
        "stop_sequences": ["AI:", "User:"],
        "top_p": 0.9,
        "do_sample": True,
        "repetition_penalty": 1.2,
        "stream": False,
    }

    response = inference.text_generation(prompt=prompt, **generation_params)
    # If hallucination is detected
    if isHallucination(response, user_message):
        fallback_prompt = f'''
        You are an AI-powered mental health support assistant designed to provide accurate, 
        compassionate, and evidence-based responses. Please focus on providing information 
        grounded in well-established mental health principles, and avoid any speculative or 
        unsupported claims. If the topic is beyond your expertise, gently suggest that the 
        user consult a qualified mental health professional. Prioritize empathy, clarity, 
        and support in your response.

        User message: {user_message}
        AI:
        '''
        response = inference.text_generation(prompt=fallback_prompt, **generation_params)

    print(response)
    return {"response": process(response)}


def process(text):
    new_text = ''
    for i in range(len(text.split(' '))):
        if i % 20 == 0 and i != 0:
            new_text += '\n'
        new_text += text.split(' ')[i] + ' '
    new_text = new_text.strip()
    if 'User message' in new_text:
        idx = new_text.index('User')
        new_text = new_text[:idx]
    return new_text


public_url = ngrok.connect(8000)
print("Public URL:", public_url)

# Run Uvicorn (ASGI server)
if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
