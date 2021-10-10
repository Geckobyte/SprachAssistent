import speech_recognition as sr
import datetime
import pyttsx3
from datetime import date
import requests
import neuralnetwork

speech_engine = sr.Recognizer()
voice_engine = pyttsx3.init()
raw_command = ""

name = "hans"

def get_time():
    time = datetime.datetime.now().strftime("%H:%M")
    today = date.today()
    date_full = today.strftime("%d %B %Y")
    return time, date_full


def say(text):
    voice_engine.say(text)
    voice_engine.runAndWait()


def startup():
    voices = voice_engine.getProperty('voices')
    voice_engine.setProperty('voice', voices[0].id)
    time = get_time()[0]

def from_microphone():
    with sr.Microphone() as micro:
        print('listening...')
        audio = speech_engine.listen(micro)
        try:
            command = speech_engine.recognize_google(audio, language="de-DE").lower()
        except:  
            return ""

    return command


def run_assistant():
    command = from_microphone()
    print(command)
    if name in command:
        raw_command = command.replace(name, "")
        if len(raw_command) > 0:
            action_id = neuralnetwork.getAction(raw_command)
            print(action_id[0], " mit ", action_id[1])
            action(action_id[0])
    else:
        pass


cases = {-1: lambda: idk_c(), 0: lambda: time_c(), 1: lambda: date_c(), 2: lambda: inzidenz_c(),  3: lambda: gethomework_c()}


def action(case_id):
    cases[case_id]()


def idk_c():
    say("Ich habe das leider nicht verstanden.")


def time_c():
    time = get_time()[0]
    say(f"Es ist gerade {time}")


def date_c():
    currentDate = get_time()[1]
    say(f"Heute ist {currentDate}")


def inzidenz_c():
    r = requests.get("https://api.corona-zahlen.org/germany")
    re = r.json()
    weekIncidence = re["weekIncidence"]
    incidence_round = str(round(weekIncidence, 1))
    say(f"Die 7-Tage Inzidenz in Deutschland beträgt: {incidence_round}")

def gethomework_c():
    say("Du hast noch folgendes zu machen")


startup()
while True:
    run_assistant()
