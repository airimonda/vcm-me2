"""
Synthetic audio with Piper: command clips for the classifier (--templates base/round2) and the
"Watson" wake word set (--templates wake). Copied from the earlier project; the wake templates add
"Watson, <benchmark command>" positives and more sound-alike negatives.

Covers all 19 benchmark labels + OUT_OF_SCOPE (not just the thin ones), so that
"sounds synthetic" never predicts a label. Diversity comes from:
  - ~1,060 voices: libritts_r (904 speakers), vctk (109), l2arctic (24 L2
    speakers), arctic (18), plus 4 single-speaker models
  - many phrasings per label with random slot values and politeness fillers
  - random speaking rate / prosody noise per clip
  - a random resample warp (x0.9-1.1) that shifts pitch + formants, i.e. voices
    Piper doesn't have
  - Filipino-English accent simulation on a share of clips: Piper's phonemes are
    rewritten before synthesis (f->p, v->b, th->t/d, z->s, tapped r, full vowels
    instead of reduced ones), each rule applied with some probability

Splits are voice-disjoint: every 10th speaker of each multi-speaker model and
the whole amy model go to dev; everything else to train. Your own recordings
are the test set and never mix with this.

Run with the Piper venv:
  .venv-piper/bin/python scripts/generate_piper.py            # full set
  .venv-piper/bin/python scripts/generate_piper.py --scale 0.02 --out data/raw/Piper_smoke
Writes data/raw/Piper/<label>/<n>.wav (16 kHz PCM16) + manifests/piper_synthetic.csv.
"""
import argparse
import csv
import glob
import json
import os
import random
import time
from multiprocessing import Pool

import numpy as np
import soundfile as sf
import soxr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICE_DIR = os.path.join(REPO_ROOT, "data/piper_voices")
SR = 16000

# model -> sampling weight; multi-speaker models get most of the clips
VOICE_WEIGHTS = {
    "en_US-libritts_r-medium": 0.42, "en_GB-vctk-medium": 0.18,
    "en_US-l2arctic-medium": 0.16, "en_US-arctic-medium": 0.08,
    "en_US-lessac-medium": 0.04, "en_US-ryan-medium": 0.04,
    "en_US-amy-medium": 0.04, "en_GB-alba-medium": 0.04,
}
DEV_SINGLE = {"en_US-amy-medium"}

ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen").split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def num(n):
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    return "one hundred"


def duration(rng):
    unit, vals = rng.choice([("second", [5, 10, 15, 20, 30, 45, 90]),
                             ("minute", [1, 2, 3, 5, 10, 15, 20, 25, 30, 45]),
                             ("hour", [1, 2, 3])])
    v = rng.choice(vals)
    return f"{'one' if v == 1 else num(v)} {unit}{'' if v == 1 else 's'}"


def clock(rng):
    h = rng.choice(range(1, 13))
    m = rng.choice(["", "", " thirty", " fifteen", " forty five", " o'clock"])
    tail = rng.choice([" a.m.", " p.m.", " in the morning", " tonight", " in the evening"])
    return f"{num(h)}{m}{tail}"


SLOTS = {
    "dur": duration,
    "time": clock,
    "deg": lambda r: num(r.choice(range(16, 31))),
    "pct": lambda r: num(r.choice(range(10, 101, 10))),
    "color": lambda r: r.choice("red blue green yellow purple pink orange white violet".split()),
    "rname": lambda r: r.choice(["bedroom", "kitchen", "living room", "office", "hallway", "bathroom", "garage"]),
    "room": lambda r: r.choice(["", "", " in the living room", " in the bedroom", " in the kitchen",
                                " in the bathroom", " in here"]),
    "todo": lambda r: r.choice(["drink water", "study", "exercise", "call mom", "take my medicine",
                                "buy groceries", "submit my homework", "feed the dog", "pay the bills",
                                "water the plants", "charge my phone", "go to the bank"]),
    "who": lambda r: r.choice(["mom", "dad", "Anna", "Miguel", "my sister", "my brother",
                               "Jose", "Maria", "the office", "Carlo", "Grace", "Tita Lorna"]),
    "msg": lambda r: r.choice(["I'm on my way", "I'll be late", "see you later",
                               "call me back", "I'm home", "happy birthday"]),
    "song": lambda r: r.choice(["", " by Ben and Ben", " by Taylor Swift", " by the Beatles",
                                " by Moira", " by SB19", " by Coldplay"]),
}

T = {  # label -> templates; {pre}/{post} are politeness fillers
    "PLAY_MUSIC": ["{pre}play music{post}", "{pre}play a song{post}", "{pre}start the music{post}",
                   "{pre}play some music{song}{post}", "{pre}put on some music", "play something",
                   "I want to listen to music", "{pre}play my playlist{post}", "music please",
                   "{pre}play a song{song}"],
    "WEATHER": ["weather", "what's the weather", "tell me the weather", "what's the weather like today",
                "is it going to rain", "{pre}tell me the forecast", "how's the weather outside",
                "will it be sunny tomorrow", "what's the temperature outside", "do I need an umbrella"],
    "TIME": ["time", "what time is it", "tell me the time", "{pre}tell me what time it is",
             "what's the time now", "what is the date today", "what day is it", "what time is it now"],
    "LIGHT_ON": ["lights on", "power on the lights", "turn on the lights{room}", "{pre}turn the lights on{room}",
                 "switch on the lights{room}", "{pre}turn on the light{room}", "lights on please",
                 "I need some light", "open the lights{room}"],
    "LIGHT_OFF": ["lights off", "kill the lights", "turn off the lights{room}", "{pre}turn the lights off{room}",
                  "switch off the lights{room}", "{pre}shut off the light{room}", "lights out",
                  "close the lights{room}"],
    "BRIGHTNESS": ["brightness {pct} percent", "set the brightness to {pct} percent",
                   "change the brightness to {pct} percent", "{pre}dim the lights{room}",
                   "make the lights brighter{room}", "{pre}set the lights to {pct} percent",
                   "brighter please", "dim it down a little", "make it dimmer{room}"],
    "COLOR": ["color {color}", "change the lights to {color}", "set the lights to {color}",
              "{pre}make the lights {color}{room}", "turn the lights {color}", "I want {color} lights",
              "change the color to {color}", "{pre}switch the light color to {color}"],
    "TIMER": ["timer {dur}", "countdown for {dur}", "start a timer for {dur}", "{pre}set a timer for {dur}",
              "set a {dur} timer", "time me for {dur}", "timer for {dur} please"],
    "ALARM": ["alarm {time}", "wake me up at {time}", "set an alarm for {time}", "{pre}set my alarm for {time}",
              "I need an alarm at {time}", "alarm at {time} please", "{pre}wake me up at {time}"],
    "TEMPERATURE": ["temperature {deg} degrees", "change the temperature to {deg} degrees",
                    "set the temperature to {deg} degrees", "{pre}set the aircon to {deg}",
                    "make it {deg} degrees", "set the thermostat to {deg} degrees", "aircon {deg} degrees",
                    "{pre}change the aircon to {deg} degrees", "I want it at {deg} degrees",
                    "turn the temperature to {deg}"],
    "PAUSE": ["pause", "pause the music", "pause this song", "{pre}pause{post}", "hold the music",
              "{pre}pause the song", "pause it", "pause playback"],
    "STOP": ["stop song", "stop music", "stop playing music", "{pre}stop the music{post}", "stop the song",
             "stop playing", "{pre}turn off the music", "stop it", "end the music"],
    "NEXT": ["skip song", "next song", "play next song", "{pre}skip this song{post}", "next track",
             "skip", "{pre}play the next track", "go to the next song", "skip to the next one", "next"],
    "VOLUME_UP": ["volume up", "increase the volume", "turn the volume up", "{pre}turn it up{post}",
                  "louder", "make it louder", "{pre}raise the volume", "I can't hear it"],
    "VOLUME_DOWN": ["volume down", "lower the volume", "turn the volume down", "{pre}turn it down{post}",
                    "quieter", "make it quieter", "{pre}decrease the volume", "too loud"],
    "CREATE_REMINDER": ["reminder {todo}", "remind me to {todo}", "create a reminder to {todo}",
                        "{pre}remind me to {todo} at {time}", "set a reminder to {todo}",
                        "don't let me forget to {todo}", "add a reminder to {todo}"],
    "LIST_REMINDERS": ["reminders", "show my reminders", "list my reminders", "{pre}show me my reminders",
                       "what are my reminders", "read my reminders", "do I have any reminders",
                       "what reminders do I have today"],
    "CALL": ["call", "make a call", "make a phone call", "{pre}call {who}{post}", "phone {who}",
             "dial {who}", "call {who} for me", "{pre}make a call to {who}", "ring {who}",
             "I want to call {who}", "give {who} a call"],
    "MESSAGE": ["message", "send a message", "send my message", "{pre}text {who}{post}",
                "send a message to {who}", "message {who} {msg}", "text {who} that {msg}",
                "{pre}send {who} a text"],
    "OUT_OF_SCOPE": [
        "what are we having for dinner", "I think it went pretty well", "can you pass me that",
        "the traffic was terrible this morning", "let's meet after class", "how was your weekend",
        "I forgot my charger", "please submit your reports by Friday", "turn on the TV",
        "open the window", "lock the front door", "what's the capital of Japan", "order some pizza",
        "delete my alarm", "how far is Manila from here", "read my emails", "tell me a joke",
        "start the washing machine", "who won the game last night", "thank you so much",
        "never mind", "hello can you hear me", "wait a second", "that's a great idea",
        "I'll be there in five minutes", "close the door please", "where did I put my keys",
        "the meeting is at three", "do you want some coffee", "it's so hot today",
        "can I borrow your notes", "we should take a break", "I'm not sure about that",
        "the music at the party was loud", "my phone is almost dead", "what's on the menu",
        "let's call it a day", "she sent me the file already", "turn left at the corner",
        "how much is this", "the lights in the mall were beautiful", "I set it on the table",
    ],
}
# Round 2 (2026-09-24): targets the failures seen on the own-voice test set --
# short benchmark forms, one-word contrasts (skip/stop, up/down), relative
# temperature, date questions, and everyday questions that must stay OOS.
# Exact copies of test sentences are removed later by build_classifier_manifest.py.
SLOTS["genre"] = lambda r: r.choice(["jazz", "rock", "lofi", "classical music", "acoustic songs", "OPM",
                                     "k-pop", "ocean sounds", "study music", "love songs", "my playlist",
                                     "the top hits", "some chill music"])
SLOTS["city"] = lambda r: r.choice(["Cebu", "Davao", "Iloilo", "Tokyo", "Singapore", "London",
                                    "New York", "Makati", "Pasig", "Bacolod"])
T2 = {
    "TEMPERATURE": ["make it warmer", "make it colder", "make it warmer in the room", "make it cooler in here",
                    "it's too hot", "it's too cold in here", "I'm feeling cold", "I'm so hot",
                    "turn up the aircon", "turn down the aircon", "lower the temperature", "raise the temperature",
                    "set thermostat to {deg} degrees", "set the temperature to {deg}", "{pre}set the aircon to {deg} degrees",
                    "thermostat {deg}", "temperature {deg}", "cool it down a bit", "warm it up please",
                    "can you make the room colder", "can you make it a little warmer"],
    "NEXT": ["skip this music", "skip the track", "skip that song", "skip ahead", "skip to the next track",
             "skip this one", "skip that", "next one please", "skip please", "can you skip this song",
             "play the next one", "next track please", "change the song"],
    "STOP": ["stop the music", "stop that song", "stop this", "stop playing that", "stop everything",
             "stop the track", "stop the playlist"],
    "COLOR": ["color {color}", "color {color}", "{color}", "{color} lights", "make it {color}",
              "lights {color}", "{color} please", "go {color}", "turn it {color}"],
    "CALL": ["call", "call", "call now", "call please", "make a call", "start a call", "place a call",
             "call {who}", "phone call", "dial", "call someone"],
    "MESSAGE": ["text {who} {msg}", "text {who} that {msg}", "send {who} a message saying {msg}",
                "message {who}", "send a text", "text someone", "send a message to {who} saying {msg}",
                "tell {who} {msg}"],
    "TIME": ["what's the date", "what's the date today", "what's today's date", "what day is it today",
             "what date is it", "what's the date in {city}", "what time is it in {city}", "time please",
             "what's the time", "tell me the date", "do you know what time it is", "the time please"],
    "VOLUME_UP": ["turn the volume up to {pct} percent", "turn it up to {pct} percent", "turn it louder",
                  "louder please", "volume up please", "a bit louder", "turn up the music", "raise it"],
    "VOLUME_DOWN": ["turn the volume down to {pct} percent", "turn it down to {pct} percent", "softer please",
                    "volume down please", "a bit quieter", "turn down the music", "lower it"],
    "PLAY_MUSIC": ["play {genre}", "play some {genre}", "put on {genre}", "I want to listen to {genre}",
                   "can you play {genre}", "play {genre} for me", "start {genre}", "turn on some {genre}"],
    "BRIGHTNESS": ["make the {rname} light dimmer", "make the {rname} lights brighter", "dim the {rname} light",
                   "turn the {rname} light dimmer", "brighten the {rname}", "brightness {pct}",
                   "set the {rname} light to {pct} percent"],
    "WEATHER": ["what's the weather", "weather please", "how's the weather", "what's the weather in {city}",
                "is it raining", "will it rain later", "weather today"],
    "OUT_OF_SCOPE": [
        "how's your day", "what's your name", "where are you from", "what's up", "how was class",
        "what are you doing", "are you okay", "who are you", "what do you think", "can you help me",
        "what's the answer to number five", "what's new with you", "how's the family", "how's work",
        "what's the plan for later", "what's for lunch", "what are we eating tonight", "what's on TV",
        "I like this song", "this song is so good", "who sings this", "I love music", "let's go to a concert",
        "can you dance", "do you know any songs", "my favorite song is on the radio", "music is life",
        "sing along with me", "what's your favorite movie", "what's your favorite food",
        "how tall are you", "when is your birthday", "are you a robot", "do you have feelings",
        "can you speak Tagalog", "is the store open", "when does the mall close", "what time is the flight",
        "the class starts at nine", "I woke up late today", "the weather in the movie was dramatic",
        "call me later", "text me when you get home", "I'll call you back", "stop talking",
        "the lights at the concert were amazing", "turn around", "next week is the exam",
        "pause for a second, let me think", "that's cold", "it's hot coffee", "color me surprised",
        "timer is a funny word", "the alarm clock is broken", "remind them tomorrow", "play the game with us",
        "okay", "hmm", "uh huh", "yes please", "no thanks", "maybe later", "good morning", "good night",
        "salamat", "ano ba yan", "ingat ka", "sige na", "bahala na", "ayos lang",
    ],
}
# Wake word "Watson": positives said many ways (alone and running into a command),
# negatives that sound close. Everything else the wake model must ignore comes
# from the command data (build_wake_manifest.py).
BENCH_PROMPTS = os.path.join(os.path.expanduser("~"), "ai231-me2-voice-data/schema/prompts.csv")


@__import__("functools").lru_cache(None)
def _bench_cmds():
    try:
        with open(BENCH_PROMPTS) as f:
            return [speakable(r["text"]) for r in csv.DictReader(f) if r["label"] != "OUT_OF_SCOPE"]
    except FileNotFoundError:
        return ["turn the lights on"]


SLOTS["cmd"] = lambda r: r.choice(_bench_cmds())

T_WAKE = {
    # "[[...]]" = raw phonemes: Filipino-English "Watson" with a full, unreduced
    # second vowel and a short first one (the default is American wˈɑːtsən)
    "WATSON": ["Watson", "Watson", "Watson.", "Watson?", "Watson!", "Watson,", "Watson...",
               "Watson, turn the lights on", "Watson, what time is it", "Watson, play music", "Watson, pause",
               "Watson, turn it up", "Watson, what's the weather", "Watson, set a timer",
               "hey, Watson", "Watson, please",
               "Watson, {cmd}", "Watson, {cmd}", "Watson, {cmd}", "Watson {cmd}", "Watson. {cmd}",
               "okay Watson", "Watson, Watson",
               "[[wˈatson]]", "[[wˈatson]]", "[[wˈɔtson]]", "[[wˈatsɔn]]", "[[wˈotson]]", "[[wˈɑtson]]",
               "[[wˈatson]]", "[[wˈɔtsɔn]]", "[[watsˈon]]", "[[wˈatsən]]"],
    "OTHER": ["what's on", "what's on TV", "watch this", "watch out", "washing", "wash them", "water",
              "watts", "sixty watts", "what's up", "what son", "Wednesday", "Wilson", "Watkins", "Dawson",
              "Hudson", "Jackson", "Madison", "Thompson", "Weston", "Walter", "what's that",
              "want some", "wasn't it", "waiting", "wet sand", "hot son", "what's done", "what's wrong",
              "Watt's son", "watching", "whatever", "Boston", "Austin", "washroom", "watermelon",
              "wattage", "hot sun", "what's the song",
              # round 2: the words behind round 1's false wakes (close to, not copies of, test lines)
              "what's the weather today", "what's the weather like", "what's the date", "what's the plan",
              "what's new", "what's for dinner", "what's happening", "what's that sound", "what's your name",
              "walking home", "walking around", "we're walking", "what song was that", "what song is playing",
              "what's on the menu", "watch the time", "what's so funny", "what's today", "watts per hour",
              "what's left", "was that you", "what's going on",
              # round 3: more sound-alikes and bare commands (said without the wake word)
              "watch it", "Wesson", "Wasson", "Lawson", "Watts on", "wash on", "lots on", "was it",
              "not some", "what's on now", "Robertson", "Johnson", "Atkinson", "Hutson",
              "Watanabe", "what's under", "what time", "what's the time",
              "{cmd}", "{cmd}", "{cmd}", "{cmd}", "{pre}{cmd}"],
}


def speakable(text):
    """Benchmark prompt text -> what Piper should read ('6:00 AM' -> 'six a.m.')."""
    import re
    text = re.sub(r"(\d+):00\s*([AP])M", lambda m: f"{num(int(m.group(1)))} {m.group(2).lower()}.m.", text)
    return re.sub(r"\d+", lambda m: num(int(m.group(0))), text)


PRE = ["", "", "", "please ", "hey, ", "can you ", "could you please ", "can you please "]
POST = ["", "", "", " please", " now"]

# Filipino-English phoneme rules: (from, to, prob). Applied per token.
ACCENT_RULES = [
    ("f", "p", 0.7), ("v", "b", 0.7), ("θ", "t", 0.8), ("ð", "d", 0.8), ("z", "s", 0.6),
    ("ʒ", "ʃ", 0.5), ("ɹ", "ɾ", 0.6), ("ɚ", "ɛɾ", 0.6), ("ə", "a", 0.5), ("ɐ", "a", 0.5),
    ("ɪ", "i", 0.5), ("ᵻ", "i", 0.5), ("ʊ", "u", 0.5), ("æ", "a", 0.6), ("ʌ", "a", 0.6),
    ("ɑ", "a", 0.4), ("ɔ", "o", 0.6), ("ɜ", "ɛ", 0.5), ("ː", "", 0.6),
]

_voices = {}


def _single_thread_onnx():
    """Each worker process is one synthesis stream. onnxruntime's default is one
    thread per core per session, so N workers would start N x cores threads and
    thrash (load ~330 on the 28-thread LOQ). One thread per worker instead."""
    import onnxruntime
    if getattr(onnxruntime, "_vcm_patched", False):
        return
    base = onnxruntime.SessionOptions

    def one_thread():
        o = base()
        o.intra_op_num_threads = 1
        o.inter_op_num_threads = 1
        return o
    onnxruntime.SessionOptions = one_thread
    onnxruntime._vcm_patched = True


def get_voice(name):
    if name not in _voices:
        _single_thread_onnx()
        from piper import PiperVoice
        _voices[name] = PiperVoice.load(os.path.join(VOICE_DIR, name + ".onnx"))
    return _voices[name]


def accentize(phonemes, id_map, rng):
    out = []
    for p in phonemes:
        for src, dst, prob in ACCENT_RULES:
            if p == src and rng.random() < prob and all(c in id_map for c in dst):
                p = dst
                break
        out.extend(list(p) if p else [])
    return out


def render(label, rng, templates=None, fixed=None):
    if fixed:
        return fixed
    text = rng.choice((templates or T)[label])
    fill = {"pre": rng.choice(PRE), "post": rng.choice(POST)}
    for k, fn in SLOTS.items():
        if "{" + k + "}" in text:
            fill[k] = fn(rng)
    return " ".join(text.format(**fill).split())


def synth(job):
    idx, label, split, seed, out_dir, p_accent, tset, fixed, extra = job
    rng = random.Random(seed)
    names = [n for n in VOICE_WEIGHTS if split == "dev" or n not in DEV_SINGLE]
    if split == "dev":
        names = [n for n in names if n in DEV_SINGLE or n in MULTI]
    name = rng.choices(names, weights=[VOICE_WEIGHTS[n] for n in names])[0]
    voice = get_voice(name)
    n_spk = voice.config.num_speakers
    spk = None
    if n_spk > 1:
        pool = [s for s in range(n_spk) if (s % 10 == 0) == (split == "dev")]
        spk = rng.choice(pool)

    from piper import SynthesisConfig
    cfg = SynthesisConfig(speaker_id=spk, length_scale=rng.uniform(0.9, 1.45),
                          noise_scale=rng.uniform(0.4, 0.9), noise_w_scale=rng.uniform(0.5, 1.1))
    text = render(label, rng, {"round2": T2, "wake": T_WAKE}.get(tset, T), fixed)
    accented = rng.random() < p_accent
    id_map = voice.config.phoneme_id_map
    audio = []
    if text.startswith("[[") and text.endswith("]]"):
        ph = [c for c in text[2:-2] if c in id_map]
        audio.append(voice.phoneme_ids_to_audio(voice.phonemes_to_ids(ph), cfg))
        text, accented = "watson", True
    else:
        for sent in voice.phonemize(text):
            ph = accentize(sent, id_map, rng) if accented else sent
            audio.append(voice.phoneme_ids_to_audio(voice.phonemes_to_ids(ph), cfg))
    wav = np.concatenate(audio).astype(np.float32)
    if wav.dtype != np.float32 or np.abs(wav).max() > 1.5:   # int16-scaled output
        wav = wav / 32768.0
    warp = rng.uniform(0.9, 1.1)                              # pitch + formant + rate shift
    wav = soxr.resample(wav, voice.config.sample_rate * warp, SR)
    wav = np.clip(wav / (np.abs(wav).max() + 1e-9) * rng.uniform(0.3, 0.9), -1, 1)

    path = os.path.join(out_dir, label, f"{idx:06d}.wav")
    sf.write(path, wav, SR, subtype="PCM_16")
    speaker = f"{name}:{spk if spk is not None else 0}"
    row = dict(dataset="Piper", split=split, audio_path=path, label=label, speaker_id=speaker,
               duration_s=round(len(wav) / SR, 3), transcript_normalized=text.lower(),
               accent_sim=int(accented))
    row.update(extra)   # extra columns carried through from a --text-csv row (e.g. intent, h_*)
    return row


MULTI = {n for n in VOICE_WEIGHTS if n not in {"en_US-lessac-medium", "en_US-ryan-medium",
                                                "en_US-amy-medium", "en_GB-alba-medium"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "data/raw/Piper"))
    ap.add_argument("--manifest", default=os.path.join(REPO_ROOT, "manifests/piper_synthetic.csv"))
    ap.add_argument("--per-label", type=int, default=400)
    ap.add_argument("--per-gap-label", type=int, default=1500,
                    help="for labels with little/no real audio")
    ap.add_argument("--gap-labels", nargs="*", default=["TEMPERATURE", "CALL", "STOP", "PAUSE", "NEXT"])
    ap.add_argument("--oos", type=int, default=2000)
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--p-accent", type=float, default=0.5)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply all counts (smoke tests)")
    ap.add_argument("--templates", choices=["base", "round2", "wake", "none"], default="base")
    ap.add_argument("--text-csv", help="also read every row of this CSV (columns text,label) once")
    ap.add_argument("--wake-pos", type=int, default=4000)
    ap.add_argument("--wake-neg", type=int, default=3000)
    ap.add_argument("--round2-per-label", type=int, default=400)
    ap.add_argument("--round2-boost", nargs="*", default=["TEMPERATURE"],
                    help="labels that get 2x clips in round 2")
    ap.add_argument("--bench-per-prompt", type=int, default=0,
                    help="also read every benchmark prompt verbatim this many times")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out = os.path.abspath(args.out)

    missing = [n for n in VOICE_WEIGHTS if not os.path.exists(os.path.join(VOICE_DIR, n + ".onnx"))]
    assert not missing, f"missing voices in {VOICE_DIR}: {missing}"

    jobs, idx = [], 0
    rng = random.Random(args.seed)
    def add(label, n, fixed=None, extra=None):
        nonlocal idx
        os.makedirs(os.path.join(args.out, label), exist_ok=True)
        for _ in range(max(1, int(n * args.scale))):
            split = "dev" if rng.random() < args.dev_frac else "train"
            jobs.append((idx, label, split, args.seed * 1_000_003 + idx, args.out, args.p_accent,
                         args.templates, fixed, extra or {}))
            idx += 1

    if args.templates == "base":
        for label in T:
            add(label, args.oos if label == "OUT_OF_SCOPE" else
                args.per_gap_label if label in args.gap_labels else args.per_label)
    elif args.templates == "none":
        pass
    elif args.templates == "wake":
        for label in T_WAKE:
            add(label, args.wake_pos if label == "WATSON" else args.wake_neg)
    else:
        for label in T2:
            n = args.oos if label == "OUT_OF_SCOPE" else args.round2_per_label
            add(label, n * (2 if label in args.round2_boost else 1))
    if args.text_csv:
        with open(args.text_csv) as f:
            for r in csv.DictReader(f):
                extra = {k: v for k, v in r.items() if k not in ("text", "label")}
                add(r["label"], 1, fixed=r["text"], extra=extra)
    if args.bench_per_prompt:
        with open(BENCH_PROMPTS) as f:
            for p in csv.DictReader(f):
                add(p["label"], args.bench_per_prompt, fixed=speakable(p["text"]))

    print(f"{len(jobs):,} clips, {args.workers} workers -> {args.out}")
    t0, rows = time.time(), []
    with Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(synth, jobs, chunksize=16)):
            rows.append(row)
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1:,}/{len(jobs):,} ({time.time() - t0:.0f}s)", flush=True)
    rows.sort(key=lambda r: r["audio_path"])
    base_fields = ["dataset", "split", "audio_path", "label", "speaker_id", "duration_s",
                   "transcript_normalized", "accent_sim"]
    extra_fields, seen = [], set(base_fields)
    for r in rows:                          # union of any extra (--text-csv passthrough) columns
        for k in r:
            if k not in seen:
                seen.add(k)
                extra_fields.append(k)
    with open(args.manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=base_fields + extra_fields, restval="")
        w.writeheader()
        w.writerows(rows)
    hours = sum(r["duration_s"] for r in rows) / 3600
    print(f"wrote {len(rows):,} rows ({hours:.1f}h) -> {args.manifest} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
