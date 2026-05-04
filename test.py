from omnivoice import OmniVoice
import soundfile as sf
import torch

model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0",
    dtype=torch.float16
)
# Apple Silicon users: use device_map="mps" instead

audio = model.generate(
    text="""Ciao, questa è una voce di riferimento in italiano.

Sto parlando in modo naturale, con un tono rilassato ma chiaro, come in una conversazione reale.

Ogni frase è pronunciata con attenzione, lasciando piccole pause tra le idee, così che il discorso risulti fluido e facile da seguire.

CavadaLabs aiuta le aziende a usare l’intelligenza artificiale senza complicazioni, trasformando processi complessi in strumenti semplici e concreti.

L’obiettivo è rendere la tecnologia accessibile, utile e davvero integrata nel lavoro quotidiano.

Questa voce deve essere stabile, coerente e piacevole da ascoltare, senza cambi improvvisi di tono o velocità.""",
    instruct="female, low pitch",
    language_id="it",
)

# If you don't want to input `ref_text` manually, you can directly omit the `ref_text`.
# The model will use Whisper ASR to auto-transcribe it.

sf.write("out.wav", audio[0], 24000)