import pandas as pd
import numpy as np
# Define the path to the merged dataset
csv_path = "merged_final_dataset_0529.csv" #path to output csv files

# Define model names and type list
model_names = [ "gpt-4-vision-preview", "gpt4o", "gemini-1-5","gemini-pro-vision","claude-3-vision", 
               'llavav-next-110b', 'llavav-next-34b', 'VILA-40b','Yi-VL-34b', 'Intern-VL-1-5', 'Cogvlm2', "DeepSeekVL", "MiniCPMV", "HPT_AIR"]
type_list = ["Adjective", "Adjectives modify different objects", "Complement modifies different part", "Emotional expression", "High level expression", "Literal and Extended Meanings", "Noun", "Noun_word", "Passive and active", "Same question, different images, different meaning", "Who does take the action"]

target_list = [ "Adjective","Noun_word","Emotional expression","Adjectives modify different objects","Complement modifies different part","Who does take the action","Passive and active","High level expression","Same question, different images, different meaning", "Noun", "Literal and Extended Meanings"]



print(len(type_list))
# Read the merged dataset
df_now = pd.read_csv(csv_path, encoding='latin1')
print(df_now.columns)

# Sort the DataFrame by the 'class' column

df_now['class'] = df_now['class'].str.replace('High level expression v2','High level expression')
df_now['class'] = df_now['class'].str.replace('Adjectives modify different objects','Coordination Ambiguity')
df_now['class'] = df_now['class'].str.replace('Complement modifies different part','Coordination Ambiguity')
df_now['class'] = df_now['class'].str.replace('Who does take the action','Attachment Ambiguity')
df_now['class'] = df_now['class'].str.replace('Literal and Extended Meanings','Idiom')
df_now['class'] = df_now['class'].str.replace('Same question, different images, different meaning','Pragmatic Ambiguity')
df_now['class'] = df_now['class'].str.replace('Passive and active','Structural Ambiguity')
df_now['class'] = df_now['class'].str.replace('High level expression','Pragmatic Ambiguity')
df_now['class'] = df_now['class'].str.replace('Emotional expression','Verb')
df_now['class'] = df_now['class'].str.replace('Noun_word','Noun')
df_now = df_now.sort_values(by=['class', 'caption'])
#df_now.to_csv("consolidated_results_0602.csv", index=True)
#print("Results have been consolidated into consolidated_results.csv")
set(df_now['class'])


model_names = ["gpt4o", 'llavav-next-34b']


type_list = ["Adjective", "Attachment Ambiguity", "Coordination Ambiguity", "Idiom", "Noun", "Pragmatic Ambiguity","Structural Ambiguity","Verb"]

target_list = ["Adjective","Noun","Verb", "Attachment Ambiguity", "Coordination Ambiguity","Structural Ambiguity","Pragmatic Ambiguity","Idiom"]

result_data = []

for model_name in model_names:
    count = 0
    index = 0
    prev = "Adjective"
    count_all = [0] * 8
    total_all = [0] * 8
    tol = 0
    
    for i in range(0, len(df_now) - 1, 2):  # step by 2
        
        if i + 1 >= len(df_now):  # Skip if there's no pair
            break
        
        row1, row2 = df_now.iloc[i], df_now.iloc[i + 1]
        #print(row2['file_name'])
        ans1, ans2 = row1[model_name], row2[model_name]
        #print(i,row1['file_name'],row2['file_name'] )
        if type(ans1) is not str or type(ans2) is not str or "No Response" in ans1 or "No Response" in ans2:
            continue

        tol += 1
        if row1['class'] != prev:
            prev = row1['class']
            index += 1
        total_all[index] += 1
        correct1 = 0
        correct2 = 1
        #parse result for each model specifically
        if model_name == "llavav-next-34b":
            ans1 = ans1.split("assistant")[1].split("|")[0].strip()
            ans2 = ans2.split("assistant")[1].split("|")[0].strip()
        elif model_name == "llavav-next-110b" or model_name == "llavav-next-110b-0523":
            ans1 = ans1[2:-2].split("|")[0].split(":")[0].split(":")[0].split("*")[-1].strip().lstrip('*')
            ans2 = ans2[2:-2].split("|")[0].split(":")[0].split(":")[0].split("*")[-1].strip().lstrip('*')
        elif model_name == "gemini-1-5":
            ans1 = ans1.split("|")[0].split(":")[0].split(":")[0].split("*")[-1].strip().lstrip('*')
            ans2 = ans2.split("|")[0].split(":")[0].split(":")[0].split("*")[-1].strip().lstrip('*')
        elif model_name == 'HPT_AIR':
            ans1 = ans1.replace("Answer:", "").split("|")[0].split(":")[0].strip()
            ans2 = ans2.replace("Answer:", "").split("|")[0].split(":")[0].strip()
        elif model_name == 'VILA_3b':
            ans1 = ans1.split("|")[0].split(":")[0].strip('.').strip()
            ans2 = ans2.split("|")[0].split(":")[0].strip('.').strip()
            if len(ans1.split("answer is")) > 1:
                ans1 = ans1.split("answer is")[1].strip()
            if len(ans2.split("answer is")) > 1:
                ans2 = ans2.split("answer is")[1].strip()
        elif model_name == 'VILA_8b':
            ans1 = ans1.split("|")[0].split(":")[0].strip('.').strip()
            ans2 = ans2.split("|")[0].split(":")[0].strip('.').strip()
        elif model_name == "gpt4o":
            ans1 = ans1.split("|")[0].split(":")[0].strip('.').strip()
            ans2 = ans2.split("|")[0].split(":")[0].strip('.').strip()
            if(len(ans1) > 0):
                ans1 = ans1[0]
            if(len(ans2) > 0):
                ans2 = ans2[0]
        elif model_name == "Intern-VL-1-5":
            ans1 = ans1.split("|")[0].split(":")[0].strip()[0]
            ans2 = ans2.split("|")[0].split(":")[0].strip()[0]
        else:
            ans1 = ans1.split("|")[0].split(":")[0].strip()[0]
            ans2 = ans2.split("|")[0].split(":")[0].strip()[0]
        correct1 = ans1.upper() == row1['answer'].strip().upper()
        correct2 = ans2.upper() == row2['answer'].strip().upper()
        
         
        if correct1 and correct2:
            count += 1
            count_all[index] += 1

    type_to_count = dict(zip(type_list, count_all))
    type_to_total = dict(zip(type_list, total_all))
    new_count = [type_to_count.get(item, None) for item in target_list]
    new_total = [type_to_total.get(item, None) for item in target_list]
    count_all = new_count[:]
    total_all = new_total[:]
    result_list = [count_all[i] / total_all[i] if total_all[i] > 0 else 0 for i in range(8)]
    result_list.append(sum(count_all[:3]) / sum(total_all[:3]))
    result_list.append(sum(count_all[3:6]) / sum(total_all[3:6]))
    result_list.append(sum(count_all[6:8]) / sum(total_all[6:8]))
    result_list.append(sum(count_all) / sum(total_all))
    result_data.append(result_list)
    print(model_name.replace("-", "_")+"_Amibiguity_Accuracy", " = ", result_list)

columns = target_list[:] + ["Lexical", "Syntactic", "Semantic", "Overall"]
print(len(columns), len(result_data))
result_df = pd.DataFrame(result_data, columns=columns, index = model_names)
result_df.to_csv("consolidated_results_0602.csv", index=True)

print("Results have been consolidated into consolidated_results.csv")
