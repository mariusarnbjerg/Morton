import pandas as pd
import re

# --- READ EXCEL FILE ---
file_path = r"C:/Users/Mariu/Downloads/Endelige_udtræk_INC6299351.xlsx"
out_path=r"C:/Users/Mariu/Downloads/separate_data.xlsx"
sheet_name = "Anæstesiprætilsynsnotatet"


df = pd.read_excel(file_path, sheet_name=sheet_name)

# --- SETTINGS ---
text_column = "Tekst"

sections = [
    "Anamnese",
    "Neuro/Psyk - øvrigt",
    "Respiratorisk",
    "Rygestatus",
    "Kardiovaskulært",
    "GI/Lever/Nyre",
    "Endo/Andet",
    "Bevægeapparat",
    "Objektiv undersøgelse",
    "Neurologisk",
    "Højde og vægt",
    "Højde",
    "Vægt",
    "BMI",
    "Luftvej",
    "Mallampati",
    "Mundåbning",
    "Underbid",
    "TM afstand",
    "Tandstatus",
    "Abdominalt",
    "Ryg",
    "Plan for anæstesi",
    "ASA",
    "Planlagt anæstesitype",
    "Induktion",
    "Vedligehold",
    "Luftvejsplan",
    "Monitorering",
    "Samtykke",
    "Andet",
    "Performance Score"
]

def extract_sections(text):
    if pd.isna(text):
        return {}

    text = str(text)

    pattern = "|".join([re.escape(section + ":") for section in sections])
    matches = list(re.finditer(pattern, text))

    result = {}

    for i, match in enumerate(matches):
        section_name = match.group().replace(":", "").strip()
        start = match.end()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        value = text[start:end].strip()

        # Handles duplicate headings, e.g. Kardiovaskulært appearing twice
        if section_name in result:
            counter = 2
            new_name = f"{section_name}_{counter}"

            while new_name in result:
                counter += 1
                new_name = f"{section_name}_{counter}"

            result[new_name] = value
        else:
            result[section_name] = value

    return result

# --- EXTRACT TEXT SECTIONS ---
parsed = df[text_column].apply(extract_sections)

parsed_df = pd.DataFrame(parsed.tolist())

# --- COMBINE ORIGINAL DATA WITH NEW COLUMNS ---
final_df = pd.concat([df, parsed_df], axis=1)

# Optional: replace missing values with empty cells
final_df = final_df.fillna("NULL")

# --- DISPLAY FIRST 100 ROWS ---
print(final_df.head(100))

# # --- EXPORT CLEANED DATA ---
# final_df.to_csv("cleaned_data.csv", index=False, encoding="utf-8-sig")
final_df.to_excel(out_path, index=False)