#from pymilvus import Collection, utility, Connections
import json
import re
from processing import processing_pipeline
import random
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient

import config
from utils import call_llm

# ------------------------------- Tasks should be updated in the future -------------------------------


def upload_to_milvus(doc: object, file_id: str, model: SentenceTransformer):
    """
    Uploads document chunks to Milvus after processing.
    Skips upload if file_id already exists in the collection.
    """
    collection_name = "file_embeddings"

    try:
        print("Connecting to Milvus...")
        client = MilvusClient(
            uri=f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}"
        )
        print("Successfully connected to Milvus.")
        
        # 1. Check if the collection exists
        if not client.has_collection(collection_name):
            print(f"Error: Collection '{collection_name}' does not exist.")
            return

        # 2. Query Milvus to see if any records with this file_id already exist
        # Important: String fields need quotes in the filter expression
        filter_expr = f"file_id == '{file_id}'"
        
        print(f"Checking for existing records with file_id: '{file_id}'...")
        existing_records = client.query(
            collection_name=collection_name,
            filter=filter_expr,
            output_fields=["file_id"],
            limit=1
        )

        # 3. Check the result. If the list is not empty, the file exists
        if existing_records:
            print(f"File '{file_id}' already exists in the Milvus collection. Skipping upload.")
            return
        
        print(f"File '{file_id}' not found. Starting processing pipeline...")
        
        # 4. If the file doesn't exist, run the full processing pipeline
        processed_data = processing_pipeline(doc, file_id, model)

        # 5. Insert the newly processed data
        if processed_data:
            print(f"Uploading {len(processed_data)} chunks to Milvus...")
            
            # Insert data (MilvusClient handles flushing automatically)
            insert_result = client.insert(
                collection_name=collection_name,
                data=processed_data
            )
            
            print(f"Upload complete. Inserted {insert_result['insert_count']} records.")
        else:
            print("Processing pipeline returned no data to upload.")

    except Exception as e:
        print(f"An error occurred during the Milvus operation: {e}")
        import traceback
        traceback.print_exc()
  

def summary(file):
    """
    Summarizes a legal PDF document using a local Ollama model.
    
    Args:
        pdf_path (str): Path to the PDF file.
        model (str): Ollama model name (e.g. "llama3", "OLLAMA_MODEL").
    
    Returns:
        str
    """

    if "qwen" in config.MODEL.lower():
        introduction = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. From now on, you will recieve instructions in Polish only. Your answers HAVE TO be in Polish only as well."
    else:
        introduction = "Jesteś programem AI specjalizującym się w streszczaniu dokumentów." 
    
    try:
        page_summaries = []
        
        if len(file) > 15:

            for i, page in enumerate(file):
                text = page.get_text().strip()

                if not text:
                    print(f"Skipping empty page {i+1}")
                    continue

                print(f"Sumarizing page {i+1}/{len(file)}...")

                system = f"""
                {introduction}

                **ZADANIE:**
                Twoim zadaniem jest wygenerowanie **zwięzłego i formalnego streszczenia** pojedynczej strony dokumentu.

                **WYMAGANIA:**
                - **Język:** wyłącznie polski, formalny, rzeczowy.
                - **Treść:** Zachowaj wszystkie kluczowe informacje prawne, daty, nazwy stron i istotne postanowienia.
                - **Styl:** Streszczenie musi być obiektywne, bez żadnych komentarzy, interpretacji, opinii, wstępów ani zakończeń.
                - **Format:** Zwróć tylko czysty tekst streszczenia.
                - **Długość:** Streszczenie powinno być jak najkrótsze, ale zawierać wszystkie niezbędne detale. Staraj się zmieścić w około 50-70 słowach.
                """

                options = {
                    "temperature": 0.5,
                    "top_p": 0.5,
                    "num_predict": 1000,
                    "repeat_penalty": 1.1
                }

                output = call_llm(
                    endpoint="task",
                    input=text,
                    system_message=system,
                    options=options
                )

                output.raise_for_status()
                json_output = output.json()        
                
                summary = json_output.get("response", "")

                page_summaries.append(summary)

        print("Generowanie końcowego streszczenia...")

        if len(page_summaries) > 0:
            text = "\n\n".join(summary for summary in page_summaries)
        else:
            text = "\n\n".join(page.get_text().strip() for page in file)

        system = f"""
        {introduction}

        **ZADANIE:** 
        Twoim zadaniem jest wygenerowanie **jednego, zwięzłego i formalnego streszczenia całego dokumentu**, bazując na dostarczonych fragmentach lub całym tekście.

        **WYMAGANIA:**
        - **Język:** wyłącznie polski, formalny, rzeczowy.
        - **Treść:** Ujmij wszystkie najważniejsze punkty, ustalenia, daty i strony z całego dokumentu.
        - **Styl:** Streszczenie musi być obiektywne, bez żadnych komentarzy, interpretacji, opinii, wstępów ani zakończeń.
        - **Format:** Zwróć tylko jeden, spójny akapit. 
        - **Długość:** Całe streszczenie powinno być zwięzłe.

        **PRZYKŁAD FORMATU ODPOWIEDZI:**
        Dokument dotyczy umowy z dnia XX.YY.ZZZZ pomiędzy Firmą A a Firmą B, obejmującej zakres prac P i Q. Ustalono termin realizacji do DD.MM.RRRR oraz warunki płatności określone w paragrafie X.
        """
        
        options = {
            "temperature": 0.5,
            "top_p": 0.5,
            "num_predict": 1000,
            "repeat_penalty": 1.1
        }

        summary = call_llm(
            endpoint="task",
            input=text,
            system_message=system,
            options=options
        )

    except Exception as e:
        print(f"Error generating summary: {e}")

    return summary

def summary_formatter(summary: str, max_retries: int = 5) -> str:
    """
    Converts a given text into a single, cohesive paragraph using a self-correction loop.
    It will retry up to 'max_retries' times if the output contains markdown.

    Args:
        summary (str): The text to be reformatted.
        max_retries (int): The maximum number of times to loop the formatting process.

    Returns:
        str: The reformatted text in paragraph form.
    """

    if "qwen" in config.MODEL.lower():
        introduction = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. From now on, you will recieve instructions in Polish only. Your answers HAVE TO be in Polish only as well."
    else:
        introduction = "Jesteś programem AI specjalizującym się w weryfikowaniu i poprawianiu formatowania tekstu."

    if not summary or not summary.strip():
        return "" 

    current_text = summary
    
    for i in range(max_retries):
        print(f"Formatting attempt {i + 1}/{max_retries}...")
        try:
            system = f"""
            {introduction}
            
            **ZADANIE:** 
            Twoim **JEDYNYM ZADANIEM** jest przekształcenie dostarczonego tekstu w **jeden, spójny akapit**, bez żadnych dodatków.

            **WYMAGANIA KRYTYCZNE (BEZWZGLĘDNIE OBOWIĄZUJĄCE):**
            1.  **Odpowiedź musi być WYŁĄCZNIE CZYSTYM TEKSTEM JEDNEGO AKAPITU.**
            2.  **NIE DODAJ ABSOLUTNIE NICZEGO INNEGO:**
                * **Żadnych wstępów** (np. "Oto tekst w formie akapitu:", "Twoja prośba została zrealizowana:").
                * **Żadnych powitań, zakończeń, pytań, komentarzy, wyjaśnień, ani zdań wprowadzających.**
                * **Żadnych znaków formatowania Markdown jak `**` lub `*`**.
                * **Żadnych podziałów linii**, ani nagłówków.
            3.  Zachowaj **całą istotną treść** z oryginalnego tekstu.
            4.  Połącz wszystkie zdania w jedną, płynną całość.
            5.  Odpowiedź musi być **wyłącznie w języku polskim**.

            **!!!WAŻNE!!!**
            Jeżeli otrzymany tekst już jest w formie jednego akapitu i **NIE ZAWIERA ŻADNEGO FORMATOWANIA MARKDOWN**, zwróć go **BEZ ŻADNYCH ZMIAN**.

            **PRZYKŁAD ODPOWIEDZI, KTÓRA JEST POPRAWNA (dokładnie taki format, bez dodatków):**
            Powyższy protokół dotyczy przeglądu technicznego wózka jezdniowego. Wykonano go 29 kwietnia 2025. Eksploatującym jest ORLEN S.A. Oddział PGNIG w Sanoku. Urządzenie to wózek typu EV-717. Wynik badania był pozytywny, a następny termin to kwiecień 2026.
            """

            options = { 
                "temperature": 0.5, 
                "top_p": 0.5, 
                "num_predict": 1000, 
                "repeat_penalty": 1.1 
            }

            output = call_llm(
                endpoint="task",
                input=summary,
                system_message=system,
                options=options
            )

            # --- The Correction Check ---
            if "**" not in output:
                print("Formatting successful.")
                return output # Return the clean text
            else:
                print("Formatting failed, markdown detected. Retrying...")
                current_text = output # The failed output becomes the new input

        except Exception as e:
            print(f"Error during formatting attempt {i + 1}: {e}")
            return current_text # Return the last known text on error

    # If the loop finishes without a clean result, return the last attempt
    print(f"Could not format text cleanly after {max_retries} attempts.")
    return current_text


def category_subcategory(summary, categories_json):
    """
    Assigns a legal document to one category and one of its subcategories using Ollama.

    Args:
        summary (str): Summaries of each page of the document.
        categories_json (dict): JSON of categories
        model (str): Name of the Ollama model to use.

    Returns:
        str
    """

    if "qwen" in config.MODEL.lower():
        introduction = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. From now on, you will recieve instructions in Polish only. Your answers HAVE TO be in Polish only as well."
    else:
        introduction = "Jesteś programem AI specjalizującym się w kategoryzowaniu dokumentów."

    try:
        system = f"""
        {introduction}

        **ZADANIE**
        Twoim zadaniem jest przypisanie dostarczonego podsumowania dokumentu do **najlepiej pasującej kategorii głównej** z listy. 
        Następnie, **jeśli wybrana kategoria posiada podkategorie**, przypisz dokument również do jednej z jej podkategorii.

        **DOSTĘPNE KATEGORIE I PODKATEGORIE:**
        {json.dumps(categories_json, ensure_ascii=False, indent=2)}

        **WYMAGANIA ODPOWIEDZI:**
        - Odpowiedź musi być **wyłącznie w formacie JSON**.
        - JSON musi zawierać **dokładnie** dwa klucze: `"category"` i `"subcategory"`.
        - **Jeśli wybrana kategoria główna nie ma zdefiniowanych podkategorii, wartość dla klucza `"subcategory"` musi wynosić `null`.**
        - Wartości kluczy muszą być **dokładnymi nazwami** kategorii i podkategorii z listy DOSTĘPNYCH OPCJI.
        - **NIE WOLNO** dodawać żadnego tekstu przed, po, ani poza formatem JSON (bez wstępów, wyjaśnień, ani innych słów).
        """
    
        options = {
            "temperature": 0.5,
            "top_p": 0.5,
            "num_predict": 1000,
            "repeat_penalty": 1.1
        }

        categories = call_llm(
            endpoint="task",
            input=summary,
            system_message=system,
            options=options
        )

        print("Ukończono przydzielanie kategorii/podkategorii.")

    except Exception as e:
        print(f"Error assigning category/subcategory: {e}")

    return categories

def department_assignment(summary, departments_json): 
    """
    Assigns a document to the most appropriate department based on content.

    Args:
        pages_summaries (list): List of strings, each representing a page summary.
        departments_json (dict): Dictionary with departaments names and descriptions:
        model (str): Name of the Ollama model to use.

    Returns:
        str
    """

    if "qwen" in config.MODEL.lower():
        introduction = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. From now on, you will recieve instructions in Polish only. Your answers HAVE TO be in Polish only as well."
    else:
        introduction = "Jesteś programem AI specjalizującym się w przypisaniu dokumentów do departamentów."

    try:
        system = f"""
        {introduction}
        
        **ZADANIE**
        Twoim zadaniem jest przypisanie podsumowania dokumentu do **najbardziej odpowiedniego departamentu** z poniższej listy.

        **DOSTĘPNE DEPARTAMENTY:**
        {json.dumps(departments_json, ensure_ascii=False, indent=2)}

        **WYMAGANIA ODPOWIEDZI:**
        - Odpowiedź musi być **wyłącznie nazwą jednego departamentu** z listy `DOSTĘPNE DEPARTAMENTY`.
        - **Nie wolno** dodawać żadnego tekstu przed, po, ani poza nazwą departamentu (bez wstępów, wyjaśnień, znaków interpunkcyjnych, cudzysłowów, ani innych słów).
        - **Zawsze** musisz wybrać jedną z podanych nazw, nawet jeśli dopasowanie nie jest idealne. W takim przypadku wybierz najbardziej prawdopodobny departament.
        """

        options = {
            "temperature": 0.5,
            "top_p": 0.5,
            "num_predict": 1000,
            "repeat_penalty": 1.1
        }

        department = call_llm(
            endpoint="task",
            input=summary,
            system_message=system,
            options=options
        )

        print("Ukończono przydzielanie departamentu.")

    except Exception as e:
        print(f"Error assigning department: {e}")

    return department

def orlen_department_extraction(summary): 
    """
    Assigns a document to the most appropriate department based on content.

    Args:
        pages_summaries (list): List of strings, each representing a page summary.
        departments_json (dict): Dictionary with departaments names and descriptions:
        model (str): Name of the Ollama model to use.

    Returns:
        str
    """

    if "qwen" in config.MODEL.lower():
        introduction = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. From now on, you will recieve instructions in Polish only. Your answers HAVE TO be in Polish only as well."
    else:
        introduction = "Jesteś programem AI specjalizującym się w przypisaniu dokumentów do departamentów."

    try:
        DEPARTMENT_NAMES_CANONICAL = [
            "ODDZIAŁ W SANOKU",
            "ODDZIAŁ W ZIELONEJ GÓRZE",
            "ODDZIAŁ W ODOLANOWIE",
            "ODDZIAŁ GEOLOGII I EKSPLOATACJI",
            "LABORATORIUM POMIAROWO-BADAWCZE",
            "RATOWNICZA STACJA GÓRNICTWA OTWOROWE"
        ]

        system = f"""
        {introduction}
        
        **ZADANIE**
        Twoim jedynym zadaniem jest analiza dostarczonego tekstu wiadomości i wskazanie nazwy departamentu nadawcy z poniższej listy.

        **DOSTĘPNE DEPARTAMENTY (Wybierz TYLKO JEDNĄ nazwę):**
        [{" , ".join(DEPARTMENT_NAMES_CANONICAL)}]

        **WYMAGANIA ODPOWIEDZI:**
        - MUSISZ ZWRÓCIĆ TYLKO NAZWĘ JEDNEGO Z DEPARTAMENTÓW WSKAZANYCH POWYŻEJ. Nic więcej.
        - Nie dodawaj żadnych wstępów, wyjaśnień, znaków interpunkcyjnych, cudzysłowów, ani innych słów.
        - Zawsze musisz wybrać jedną z podanych nazw, nawet jeśli nie jesteś pewien – wybierz wtedy najbardziej prawdopodobną.
        - Odpowiedź musi być w dokładnej oryginalnej formie jak na liście (z zachowaniem spacji i wielkich liter).
        """

        options = {
            "temperature": 0.5,
            "top_p": 0.5,
            "num_predict": 1000,
            "repeat_penalty": 1.1
        }

        departament = call_llm(
            endpoint="task",
            input=summary,
            system_message=system,
            options=options
        )

        print("Ukończono ustalanie nadawcy wiadomości.")

    except Exception as e:
        print(f"Error assigning Orlen department: {e}")

    return departament

def base_extraction(summary):
    """
    Extracts a specific piece of legal information from the full document text using Ollama.
    
    Args:
        text (str): Full text of the document (not summary).
        info_request (str): What to extract (e.g. "Data podpisania umowy").
        model (str): Ollama model name.
        ollama_host (str): Remote/local Ollama server URL.
    
    Returns:
        str
    """

    if "qwen" in config.MODEL.lower():
        introduction = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. From now on, you will recieve instructions in Polish only. Your answers HAVE TO be in Polish only as well."
    else:
        introduction = "Jesteś programem AI specjalizującym się w ekstrakcji informacji z dokumentów."

    try:
        system = f"""
        {introduction}
        
        **ZADANIE** 
        Twoim zadaniem jest zidentyfikowanie i wyodrębnienie najważniejszych danych, faktów i postanowień z tekstu.

        **WYMAGANIA ODPOWIEDZI:**
        - Odpowiedź musi być **po polsku**, w formie **zwięzłego, czytelnego akapitu**.
        - **Nie wolno** dodawać żadnych wstępów, zakończeń, komentarzy ani zbędnych słów. Zwróć **tylko** wyodrębnione informacje.
        - Wyodrębnij daty, nazwy stron, kwoty, terminy, numery referencyjne, kluczowe zobowiązania oraz inne istotne detale, które charakteryzują dokument.
        - Zachowaj formatowanie i interpunkcję niezbędną do czytelności wyodrębnionych danych.
        """

        options = {
            "temperature": 0.2,
            "top_k": 10,
            "top_p": 0.5,
            "num_predict": 1000,
            "repeat_penalty": 1.1
        }

        extracted_info = call_llm(
            endpoint="task",
            input=summary,
            system_message=system,
            options=options
        )
        print("Ukończono ekstrakcję informacji.")

    except Exception as e:
        print(f"Error extracting info: {e}")

    return extracted_info

def base_extraction_formatter(raw_extraction_text):
    """
    Formats raw extraction text into a plain text list of key-value pairs using Ollama.

    Args:
        raw_extraction_text (str): The verbose text output containing extracted information.

    Returns:
        str: Formatted text as a plain list of key-value pairs (e.g., "Klucz: Wartość\nKlucz2: Wartość2").
    """

    if "qwen" in config.MODEL.lower():
        introduction = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. From now on, you will recieve instructions in Polish only. Your answers HAVE TO be in Polish only as well."
    else:
        introduction = "Jesteś programem AI specjalizującym się w formatowaniu raportów z ekstrakcji informacji."

    if not raw_extraction_text or not raw_extraction_text.strip():
        return ""

    try:
        system = f"""
        {introduction}
        
        **ZADANIE** 
        Twoim zadaniem jest przekształcenie dostarczonego tekstu w listę par klucz-wartość, gdzie każda para jest w nowej linii.

        **OCZEKIWANE POLA I ICH FORMAT (Wypełnij, jeśli informacja jest dostępna w dostarczonym tekście):**
        - `Typ dokumentu`: (np. 'Umowa', 'Protokół', 'Decyzja')
        - `Data dokumentu`: (format RRRR-MM-DD, np. '2025-04-15')
        - `Strony`: (np. 'ORLEN S.A., Firma X')
        - `Przedmiot`: (krótki opis przedmiotu dokumentu/umowy)
        - `Wartość`: (kwota, jeśli dotyczy, np. '10000 PLN')
        - `Numer referencyjny`: (np. 'N4713000973')

        **WYMAGANIA KRYTYCZNE (BEZWZGLĘDNIE OBOWIĄZUJĄCE):**
        1.  **Odpowiedź musi być WYŁĄCZNIE listą par klucz-wartość, jedna para na linię.**
        2.  Format każdej linii: `Klucz: Wartość` (np. `Data dokumentu: 2025-04-29`).
        3.  Wypełnij tylko te pola, dla których informacja JEST ZAWARTA w dostarczonym tekście. Jeśli brak informacji, **pomiń całą linię dla tego klucza**.
        4.  Użyj **dokładnych nazw kluczy** jak powyżej (`Typ dokumentu`, `Data dokumentu`, itd.).
        5.  **Nie dodawaj żadnych wstępów, zakończeń, nagłówków, wyjaśnień, ani innych słów poza listą par.**
        6.  Język odpowiedzi: polski.

        **PRZYKŁAD WYJŚCIA (dokładnie taki format, bez dodatków):**
        Typ dokumentu: Protokół
        Data dokumentu: 2025-04-29
        Strony: ORLEN S.A. Oddział PGNIG w Sanoku
        Przedmiot: Przegląd techniczny wózka jezdniowego
        Numer referencyjny: N4713000973
        """

        options = {
            "temperature": 0.2,
            "top_k": 30, 
            "top_p": 0.3, 
            "num_predict": 500, 
            "repeat_penalty": 1.1
        }

        formatted_output = call_llm(
            endpoint="task",
            input=raw_extraction_text,
            system_message=system,
            options=options
        )

        print("Ukończono formatowanie do ekstrakcji informacji.")
        return formatted_output

    except Exception as e:
        print(f"Error formatting to key-value pairs: {e}")
        return raw_extraction_text 

def check_confidential(summary):
    """
    Checks whether a legal document contains confidential information using Ollama.
    
    Args:
        text (str): Full text of the document (not summary).
        model (str): Ollama model name.
        ollama_host (str): Base URL of the Ollama server.
    
    Returns:
        str
    """

    if "qwen" in config.MODEL.lower():
        introduction = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. From now on, you will recieve instructions in Polish only. Your answers HAVE TO be in Polish only as well."
    else:
        introduction = "Jesteś programem AI specjalizującym się w identyfikowaniu informacji wrażliwych w dokumentach."
    
    try:
        system = f"""
        {introduction}
        
        **ZADANIE** 
        Your answers HAVE TO be in Polish only as well. Twoim zadaniem jest **ściśle określenie, czy dostarczony tekst dokumentu zawiera jakiekolwiek informacje wrażliwe**.

        **INFORMACJE WRAŻLIWE OBEJMUJĄ:**
        - Dane osobowe (np. imiona, nazwiska, adresy, daty urodzenia, PESEL, NIP, numery dowodów osobistych)
        - Dane kontaktowe (np. adresy e-mail, numery telefonów)
        - Dane finansowe (np. numery rachunków bankowych, kwoty wynagrodzeń, dane kart płatniczych)
        - Numery identyfikacyjne (np. numery umów, sygnatury akt sądowych, numery rejestracyjne pojazdów)
        - Informacje medyczne, poufne strategie biznesowe, tajemnice handlowe.

        **WYMAGANIA ODPOWIEDZI:**
        - Odpowiedz **WYŁĄCZNIE JEDNYM SŁOWEM**: `TAK` (jeśli zawiera) lub `NIE` (jeśli nie zawiera).
        - **Nie wolno** dodawać żadnych dodatkowych słów, wyjaśnień, interpunkcji ani komentarzy.
        """

        options = {
            "temperature": 0.2,
            "top_k": 10,
            "top_p": 0.5,
            "num_predict": 1000,
            "repeat_penalty": 1.1
        }

        contains_confidential = call_llm(
            endpoint="task",
            input=summary,
            system_message=system,
            options=options
        )

        print("Ukończono sprawdzenie informacji wrażliwych.")

    except Exception as e:
        print(f"Error checking for confidential info: {e}")

    return contains_confidential

def other(prompt):
    """
    Sends a custom prompt to an Ollama model and returns the response.
    
    Args:
        prompt (str): Prompt to send to the model.
        model (str): Name of the Ollama model to use.
    
    Returns:
        str: Model's response text.
    """
    try:
        options = {
            "temperature": 0.2,
            "top_k": 10,
            "top_p": 0.5,
            "num_predict": 1000,
            "repeat_penalty": 1.1
        }

        response = call_llm(
            endpoint="task",
            input=prompt,
            options=options
        )

        print("Ukończono task 'OTHER'.")

    except Exception as e:
        print(f"Error generating response: {e}")

    return response

def suggested_action(summary):
    """
    Sends a custom prompt to an Ollama model and returns the response.
    
    Args:
        text (str): document to send to the model.
        model (str): Name of the Ollama model to use.
    
    Returns:
        str: Model's response text.
    """

    if "qwen" in config.MODEL.lower():
        introduction = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. From now on, you will recieve instructions in Polish only. Your answers HAVE TO be in Polish only as well."
    else:
        introduction = "Jesteś programem AI specjalizującym się w proponowaniu działań na podstawie treści dokumentów."

    try:
        system = f"""
        {introduction}
        
        **ZADANIE**
        Your answers HAVE TO be in Polish only as well. Twoim zadaniem jest przeanalizowanie dostarczonego tekstu dokumentu i zaproponowanie konkretnych, adekwatnych działań, które należy podjąć w związku z jego treścią.

        **WYMAGANIA ODPOWIEDZI:**
        - Odpowiedź musi być **po polsku**, w formie **jednego, formalnego akapitu**.
        - Propozycje działań powinny być **jasne, zwięzłe i praktyczne**.
        - **Nie wolno** dodawać żadnych wstępów, zakończeń, komentarzy ani zbędnych słów. Zwróć **tylko** rekomendowane działania.
        - Jeśli dokument nie wymaga oczywistych działań, wskaż, że dokument nie wymaga konkretnych akcji lub zawiera informacje do wiadomości.
        """
        
        options = {
            "temperature": 0.2,
            "top_k": 10,
            "top_p": 0.5,
            "num_predict": 1000,
            "repeat_penalty": 1.1
        }

        suggested_action = call_llm(
            endpoint="task",
            input=summary,
            system_message=system,
            options=options
        )

        print("Ukończono generowanie zaleceń.")

    except Exception as e:
        print(f"Error generating suggestions: {e}")

    return suggested_action

def orlen_department_extraction_2(input_string):
    """
    Extracts department name from input text based on first occurrence.
    Uses flexible matching to handle variations like "ODDZIAŁ PGNIG W SANOKU".
    
    Args:
        input_string (str): Text from document to analyze
        
    Returns:
        str: Name of the department that appears first in the text (with spaces)
    """
    
    # Department patterns for flexible matching - key identifying parts
    department_patterns = [
        ("SANOKU", "ODDZIAŁWSANOKU"),
        ("ZIELONEJGÓRZE", "ODDZIAŁWZIELONEJGÓRZE"),
        ("ODOLANOWIE", "ODDZIAŁWODOLANOWIE"), 
        ("GEOLOGIIIEKSPLOATACJI", "ODDZIAŁGEOLOGIIIEKSPLOATACJI"),
        ("LABORATORIUMPOMIAROWOBADAWCZE", "POMIAROWOBADAWCZE"),
        ("RATOWNICZASTACJAGÓRNICTWAOTWOROWE", "STACJAGÓRNICTWA")
    ]
    
    # Maximum normalization - keep only letters and convert to uppercase
    normalized_input = re.sub(r'[^a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]', '', input_string).upper()
    
    # Find first occurrence of each department pattern
    first_occurrences = []
    
    for pattern, dept_name in department_patterns:
        # Find position of pattern in text
        position = normalized_input.find(pattern)
        if position != -1:  # If found
            first_occurrences.append((position, dept_name))
    
    # If any departments were found, return the one that appears first
    if first_occurrences:
        # Sort by position and return the department name that appears first
        first_occurrences.sort(key=lambda x: x[0])
        return first_occurrences[0][1]
    
    # If no departments found, print message and return random one
    print("best guess")
    department_names = [dept[1] for dept in department_patterns]
    return random.choice(department_names)