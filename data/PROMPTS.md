# Synthetic data generation prompts

Paste the output into `data/extra_train.csv` (columns: `text,label`, where 1 = scam and 0 = legit), then run `python src/make_data.py`.

**Prompt used (model: ______):**

> Generate 100 realistic but entirely FICTIONAL SMS/WhatsApp messages that someone in Sri Lanka might receive. Write 50 scams (fake bank OTP requests, parcel/customs fees, job offers, Aswesuma/Samurdhi payments, lottery wins, investment schemes, fake electricity disconnection) and 50 legitimate messages (real-looking bank OTP alerts that say "do not share", family chats, class notices, delivery updates). Mix English, Sinhala script, Tamil script and Singlish (romanised Sinhala). Use ONLY fake bank names like "DemoBank", phone numbers as 07X-XXXXXXX and links ending in .example. Output CSV with columns text,label (1=scam, 0=legit), and no other text.
