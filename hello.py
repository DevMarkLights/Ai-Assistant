import speech_recognition as sr
from llama_cpp import Llama
import pyttsx3
path_to_model="C:/Users/markl/.lmstudio/models/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
# path_to_model="C:/Users/markl/.lmstudio/models/lmstudio-community/DeepSeek-R1-Distill-Qwen-7B-GGUF/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"

def formatPrompt(userInput):
    formattedPrompt='<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>'+userInput+'<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
    return formattedPrompt

def recognize_speech():
    recognizer = sr.Recognizer()
    llm = Llama(model_path=path_to_model, verbose=False, n_ctx=4096, n_gpu_layers=-1,chat_format="llama-3")
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 150)
    tts_engine.setProperty('volume', 1.0)
    recognizer.pause_threshold = 1.0
    
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        while True:
            recognizer.adjust_for_ambient_noise(source)
            recognizer.pause_threshold = 1.0
            llm.reset()
            # recognizer.dynamic_energy_threshold = False
            audio = recognizer.listen(source)
            print("Say something...")
            text=""
            keyword=""
            bye_detected=False
            try:
                text = recognizer.recognize_google(audio)
                # text='hey jeff'
                print("You said: ",text)
                if text.lower() == "hey jeff":
                    keyword=True
                    text='hey'
                if text.lower() == "bye" or bye_detected:
                    break
                while keyword: 
                    '''
                        - Creative Writing & Brainstorming → Higher temperature (~0.8-1.2)
                        - Technical & Factual Responses → Lower temperature (~0.2-0.5)
                        - Code Generation (Stable) → Very low (~0.1-0.2)
                        - Conversational AI (Balanced Response) → Medium (~0.5-0.7)
                    '''
                    response = llm(formatPrompt(text), max_tokens=256,temperature=.5)  
                    res = response['choices'][0]['text']
                    print(res)
                    tts_engine.say(res)
                    tts_engine.runAndWait()
                    
                    print("Now it's your turn to speak...")
                    audio = recognizer.listen(source, timeout=20, phrase_time_limit=10)  # Capture new speech
                    try:
                        text = recognizer.recognize_google(audio)
                        print("You said:", text)
                    except sr.UnknownValueError:
                        print("unknown value, try again")      
                                          
                    if text.lower() == "bye":
                        keyword = False
                        bye_detected = True

                    
            except sr.UnknownValueError:
                print("unknown value, try again")
            except sr.RequestError:
                print("request error")
                return
                
try:            
    recognize_speech()
except:
    print()