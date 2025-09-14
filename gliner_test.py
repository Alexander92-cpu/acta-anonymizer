from gliner import GLiNER
import torch
from torch.amp import autocast
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")
model.save_pretrained("gliner_Med")

# Load locally and move to GPU
loaded_model = GLiNER.from_pretrained("gliner_Med", load_tokenizer=True, local_files_only=True)
loaded_model.to(device)
print("model loaded to GPU successfully")

text = """
În cadrul evaluării performanței angajatului Ion Popescu , născut la 15 . 03 . 1985 în Chișinău , de naționalitate moldoveană , cu CNP 2850315123456 și domiciliul în str . Ștefan cel Mare 45 , MD - 2001 , Chișinău , care activează ca inginer software în domeniul IT , s - a constatat atingerea obiectivelor stabilite pentru trimestrul curent , demonstrând competențe avansate în gestionarea proiectelor complexe , fapt confirmat de feedback - ul pozitiv al managerului direct , exprimat în evaluarea formală transmisă prin email la ion . popescu @ gmail . com , în timp ce planul de dezvoltare propus include participarea la cursuri specializate și îmbunătățirea abilităților de leadership , cu mențiunea că angajatul , căsătorit și absolvent cu studii superioare , poate accesa resursele financiare necesare prin contul bancar IBAN MD24AG000000225100013104 , iar pentru contact rapid este disponibil la numărul de telefon mobil 069123456 , toate aceste elemente fiind centralizate în raportul administrativ care include și seria buletinului 0123456789 , numărul poliței de asigurare medicală AM1234567890 și numărul pașaportului MD1234567 .
"""

labels = [
  "Full Name",
  "Personal Numeric Code (CNP)",
  "Date of Birth",
  "Gender",
  "Nationality",
  "Spoken Language",
  "Address",
  "Work Address",
  "Mobile Phone",
  "Landline Phone",
  "Email",
  "Postal Code",
  "Birth City",
  "Birth Country",
  "Profession",
  "Activity / Occupation",
  "Employer",
  "Income",
  "Marital Status",
  "Education",
  "IBAN",
  "Bank Account",
  "Card Number",
  "Passport",
  "Identity Card",
  "License Number",
  "Health Insurance",
  "Blood Type",
  "Allergies",
  "Medical Conditions",
  "IP Address",
  "Username",
  "Device ID",
  "Biometric Data",
  "Contract Number",
  "Plate Number",
  "Digital Account",
  "Crypto Wallet",
  "Alternate Account Number",
  "Segment",
  "Politically Exposed Person (PEP)",
  "FATCA Status"
]

with autocast("cuda"):
    entities = loaded_model.predict_entities(text, labels, threshold=0.3)

for entity in entities:
    print(entity["text"], "=>", entity["label"])
