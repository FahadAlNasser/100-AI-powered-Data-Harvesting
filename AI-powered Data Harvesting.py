import requests   # Request to download PDF from websites
import pdfplumber  # Extracts data from PDF
import re   # Reqular expression. It helps with text cleaning, finding numbers, or specific words
import torch #Implemented with AI-powered text summarization
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM   #AI-powered summarization
import textwrap  # Process and analyze text by formatting it for better readability
from textblob import TextBlob  # Process and analyze text to determine if it is positive, negative, or neutral
from collections import Counter # Counting repretitve keywords and synonyms
import nltk   # Helps with processing text

device = 0 if torch.cuda.is_available() else -1   # If GPU is available, then use GPU. Otherwise, use CPU

nltk.download('punkt')   # Ensures a well structred text processing

# Dictionary method to include company's name with its sustainability report link
Sustainability_Report = {
    "Aramco": "https://www.aramco.com/-/media/publications/corporate-reports/sustainability-reports/report-2023/english/2023-saudi-aramco-sustainability-report-full-en.pdf",
    "STC": "https://www.stc.com/content/dam/groupsites/en/pdf/stc-sustainability-report2023englishV2.pdf",
    "Microsoft": "https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/msc/documents/presentations/CSR/Microsoft-2024-Environmental-Sustainability-Report.pdf"
}

# A combination of dictionary and list for the AI to recognize keywords and synonyms
related_keywords = {
    "climate change": ["global warming", "climate crisis"],
    "carbon emissions": ["CO2 emissions", "carbon footprint"],
    "water waste": ["water convservation", "water management"],
    "energy efficiency": ["energy savings", "power efficiency"],
    "renewable energy": ["green energy", "clean energy"],
    "fossil fuel": ["coal", "oil", "natural gas"],
    "pollution": ["contamination", "environmental damage"],
    "greenhouse gases": ["GHG", "carbon dioxide"],
    "sustainability strategy": ["eco plan", "sustainable development"],
    "environmental impact": ["eco footprint", "climate impact"],
    "resource management": ["natural resource planning", "conservation efforts"]
}

# Combines every keywords and synonyms into a single list for better detection
keywords = set()
for key, related in related_keywords.items():
  keywords.add(key)
  keywords.update(related)

# AI model that is able to read text and generate summaries
AI_model = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(AI_model)
model = AutoModelForSeq2SeqLM.from_pretrained(AI_model)
summarization = pipeline("summarization", model=model, tokenizer=tokenizer, device=device)

# Process to download the sustainability report from the provided link
def installing_pdf(company_name, link):
  pdf_link = f"{company_name}_Sustainability_Report.pdf"
  response = requests.get(link)
  with open(pdf_link, "wb") as file:
    file.write(response.content)
  print(f"{pdf_link} has been downloaded successfully")
  return pdf_link

# A function that works to extract the mentioned keywords and synonyms and stores which pages they appear on
def harvesting_essential_text(pdf_path):
  related_sections = []
  buffer, page_ref = [], []

  with pdfplumber.open(pdf_path) as pdf:
    for page_number, page in enumerate(pdf.pages, start=1):
      text = page.extract_text()
      if text:
        paragraphs = text.split("\n")
        for p in paragraphs:
          if any(re.search(rf"\b{kw}\b", p, re.IGNORECASE) for kw in keywords):
            buffer.append(p)
            page_ref.append(page_number)

            if len(buffer) >= 3:
              well_structured_text = " ".join(buffer)
              related_sections.append((well_structured_text, sorted(set(page_ref))))
              buffer, page_ref = [], []

  if buffer:
    related_sections.append((" ".join(buffer), sorted(set(page_ref))))
  return related_sections

# Making sure that the text is clean and readable
def enhance_readability(text):
  text = re.sub(r'\s+', ' ', text)
  text = re.sub(r'[^a-zA-Z0-9.,!?;\'"()\s]', '', text)
  return text.strip()

# Generating AI text with a clear summary without cutting off sentences
def ai_generated(text, chunk_size=1024):
  text = enhance_readability(text)
  input_tokens = tokenizer.encode(text, truncation=True, max_length=chunk_size)
  input_length = len(input_tokens)

  if input_length < 50:
    return text

  max_length = min(int(0.7 * input_length), 300)
  min_length = max(30, max_length - 100)

  summary = summarization(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
  return summary.strip()

# Counting repetitive keywords and synonyms to determines if the text is positive, negative, or neutral
def impact_of_sustainability(text):
  words = text.lower().split()
  kw_counts = Counter(word for word in words if word in keywords)

  sentiment_score = TextBlob(text).sentiment.polarity
  sentiment = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
  return kw_counts, sentiment

# Make certain that the summary is well-structured
def better_structured_summary(summary):
  summary = summary.replace("\n", " ")
  summary = re.sub(r'\s+', ' ', summary)
  return textwrap.fill(summary, width=100)

# Inspect each company's PDF, pull out important information, report it, and analyze impact
company_summary = {}

for company, link in Sustainability_Report.items():
  print(f"\n {company} Sustainability Report")
  pdf_path = installing_pdf(company, link)

  print("Harvesting important paragraphs...")
  related_sections = harvesting_essential_text(pdf_path)

  if not related_sections:
    print(f"There is no relevant sustainability content found in {company}'s report.")
    continue

  comprhensive_summary = ""
  full_text = ""

  for text, pages in related_sections:
    summary = ai_generated(text)
    formatted_pages = ", ".join(map(str, pages))
    comprhensive_summary += f"(Pages {forpages}) {summary}\n\n"
    full_text += " " + text
  kw_counts, sentiment = impact_of_sustainability(full_text)
  company_summary[company] = {
      "summary": comprhensive_summary,
      "keyword_count": kw_counts,
      "sentiment": sentiment
  }
  print(f" This is an AI-Generated Report for {company}:\n{comprhensive_summary[:500]}...\n")

# Displaying the final report with a summary, keyword inspecting, and sentiment
print("\n" + "=" * 100)
print("Compiling AI Sustainability Record")

for company, data in company_summary.items():
  formatted_summary = better_structured_summary(data["summary"])
  print(f"\n{company} Sustainability Report's Summary:\n{formatted_summary}")
  print("\nAnalyzing repetitive keywords and synonyms:")
  for kew, count in data["keyword_count"].items():
    print(f" - {kew}: {count} times")
  print(f"\n Sentiment Inspection: {data['sentiment']}")

print("\n" + "=" * 100)
