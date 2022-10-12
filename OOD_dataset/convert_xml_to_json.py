import os

input_path = "/home/zwanggy/large_files/APSI_data/OOD_data/DeScript_LREC2016"
output_path = "/home/zwanggy/large_files/APSI_data/OOD_data/DeScript_json"

for file in os.listdir(input_path):
    full_input_path = os.path.join(input_path, file)
    json_file = ".".join(file.split(".")[:-1]) + ".json"
    json_file = json_file.replace("_GOOD", "")
    json_file = json_file.replace("_filtered", "")
    json_file = json_file.replace("_flitered", "")
    json_file = json_file.replace(".new", "")
    json_file = json_file.replace(" ", "_")
    full_output_path = os.path.join(output_path, json_file)
    os.system(f"xml2json -t xml2json -o \"{full_output_path}\" \"{full_input_path}\"")
