import os
import json
import argparse
import spacy
nlp = spacy.load('en_core_web_sm')

parser = argparse.ArgumentParser(description="reformat the json files for OOD data")
parser.add_argument("--input_path", type=str, default="/home/data/zwanggy/APSI_data/OOD_data/SMILE_json", help="input json dir")
parser.add_argument("--output_path", type=str, default="/home/data/zwanggy/APSI_data/OOD_data/smile.json", help="output json file")
parser.add_argument("--text_key", type=str, default="@text")
parser.add_argument("--source", type=str, default="DeScript", help="the dataset name")

args = parser.parse_args()

cleaned_script_list, global_id = [], 0
for file in os.listdir(args.input_path):
    full_path = os.path.join(args.input_path, file)
    data = json.load(open(full_path))["scripts"]["script"]
    title = file.split(".")[0].split("_")
    title[0] = nlp(title[0])[0].lemma_

    title = "How to " + " ".join(title)
    if not title.endswith("?"):
        title += "?"
    print(title)
    for script in data:
        cleaned_script = {"id": global_id, "title": title, "source": args.source}
        subevent_list = []
        for subevent in script["item"]:
            subevent_list.append((int(subevent["@slot"]), subevent[args.text_key]))
        subevent_list.sort(key=lambda x: x[0])
        subevent_list = [e[1] for e in subevent_list]
        cleaned_script["subevents"] = subevent_list
        cleaned_script_list.append(cleaned_script)
        global_id += 1

with open(args.output_path, "w") as fout:
    for script in cleaned_script_list:
        fout.write(json.dumps(script) + "\n")



